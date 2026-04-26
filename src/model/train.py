import os
import sys
import time
import math
import glob
import shutil
import argparse
from tqdm import tqdm
from collections import defaultdict

import wandb

import torch
import torch.nn as nn

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from model_factory import get_shared_parser
from core_utils import (
    setup_global_environment, 
    setup_wandb, 
    run_inference, 
    load_checkpoint,
    get_food101_loaders
)
from Cae import UNet2d, weights_init
from core_utils import to_wandb



def generate_exp_name(args, wandb_id):
    """
    Génère une chaîne de caractères unique servant d'identifiant pour l'expérience en cours.
    Le nom est construit dynamiquement en concaténant les hyperparamètres clés du modèle 
    et un suffixe court issu de l'identifiant de session.

    Args:
        args: Objet contenant les paramètres de configuration.
        wandb_id: Identifiant unique de la session de suivi.

    Returns:
        Le nom formaté de l'expérience.
    """
    abbrv = {'image_size': 'sz', 'hidden_dims': 'dims', 'skip_mode': 'skip', 'lr': 'lr'}
    exp_parts = ["AE"]

    # On boucle sur les paramètres clés présents dans args
    for p in ['image_size', 'hidden_dims', 'skip_mode', 'lr']:
        val = getattr(args, p)
        name = abbrv.get(p, p)
        if isinstance(val, list): val = "x".join(map(str, val))
        exp_parts.append(f"{name}{val}")

    exp_parts.append(wandb_id[:6])
    return "-".join(exp_parts)


def configure_optimizers(model, args):
    """
    Configure l'optimiseur et le planificateur de taux d'apprentissage pour l'entraînement.
    Sépare les paramètres du modèle pour appliquer le déclin de poids de manière sélective, 
    en l'excluant systématiquement pour les biais et les couches de normalisation.

    Args:
        model: Le réseau de neurones à optimiser.
        args: Paramètres incluant le taux d'apprentissage et le déclin de poids.

    Returns:
        Un tuple contenant l'optimiseur configuré et le planificateur de type Cosine Annealing.
    """
    blacklist_modules = (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)

    decay = set()
    no_decay = set()

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith('bias'):
                # Tous les biais sont exclus
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_modules):
                # Les poids des modules de normalisation sont exclus
                no_decay.add(fpn)
            else:
                # Tout le reste (Conv2d, ConvTranspose2d, etc.) subit le decay
                decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": args.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    return optimizer, scheduler


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler, 
    device,
    epoch,
    total_epochs,
    use_amp,
    amp_dtype,
    accum_steps
):
    """
    Exécute une itération complète d'entraînement (une époque) sur le jeu de données fourni.
    Gère le passage avant, le calcul de la perte globale reconstruction, la rétropropagation via précision mixte (AMP), 
    l'écrêtage des gradients (gradient clipping) pour la stabilité du dictionnaire, et l'approximation en temps réel de la santé de l'espace latent.

    Args:
        model (torch.nn.Module): Le modèle à entraîner.
        dataloader (torch.utils.data.DataLoader): Le chargeur de données d'entraînement.
        optimizer (torch.optim.Optimizer): L'optimiseur pour la mise à jour des poids.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Le planificateur mis à jour à chaque étape ou époque.
        criterion (callable): La fonction de coût calculant l'erreur de reconstruction (ex: L1Loss).
        scaler (torch.amp.GradScaler): Le gestionnaire d'échelle de gradient pour prévenir le sous-dépassement (underflow) en AMP.
        device (torch.device): Le périphérique matériel actif.
        epoch (int): L'index de l'époque courante.
        total_epochs (int): Le nombre total d'époques prévues.
        use_amp (bool): Activation de la précision mixte automatique.
        amp_dtype (torch.dtype): Le format de précision mixte ciblé.
        accum_steps (int): Nombre d'étapes d'accumulation de gradient avant la mise à jour des poids.

    Returns:
        dict: Dictionnaire consolidé contenant les métriques moyennes d'entraînement calculées sur l'époque loss.
    """
    model.train()
    loop = tqdm(dataloader, desc=f"ep {epoch}/{total_epochs} [train]", dynamic_ncols=False)
    
    epoch_metrics = defaultdict(float)
    count = 0
    optimizer.zero_grad(set_to_none=True)

    for i, (x, _) in enumerate(loop):
        x = x.to(device, non_blocking=True)

        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp, dtype=amp_dtype):
            predictions = model(x)
            loss = criterion(predictions, x) / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            # Clip gradients : Standard rigoureux pour éviter les pics dans le VQ-VAE
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_val = loss.item() * accum_steps
        if math.isfinite(loss_val):
            epoch_metrics['loss'] += loss_val

            count += 1
            loop.set_postfix(loss=f"{loss_val:.4f}", avg=f"{epoch_metrics['loss']/count:.4f}")

    for k in epoch_metrics:
        epoch_metrics[k] /= max(1, count)
        
    return dict(epoch_metrics)


def run_training_loop(
    args,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    scaler, 
    device,
    use_amp,
    amp_dtype,
    exp_name,
    start_epoch,
    best_val_loss,
    fold=-1
):
    """
    Orchestre la boucle principale d'entraînement et d'évaluation sur plusieurs époques.
    Coordonne l'exécution de l'entraînement, déclenche les validations périodiques, gère la sauvegarde conditionnelle 
    des meilleurs états (checkpoints) selon le score de validation, implémente l'arrêt prématuré (early stopping) 
    et assure la synchronisation de tous les artefacts et métriques avec la plateforme de suivi.

    Args:
        args (argparse.Namespace): Configurations globales et hyperparamètres.
        model (torch.nn.Module): L'architecture réseau à entraîner et évaluer.
        train_loader (torch.utils.data.DataLoader): Chargeur pour les données d'entraînement.
        val_loader (torch.utils.data.DataLoader): Chargeur pour les données de validation.
        optimizer (torch.optim.Optimizer): Optimiseur de la descente de gradient.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Planificateur d'apprentissage.
        criterion (callable): Fonction de perte de référence.
        scaler (torch.amp.GradScaler): Gestionnaire AMP pour l'entraînement.
        device (torch.device): Périphérique matériel d'exécution.
        use_amp (bool): État de la précision mixte.
        amp_dtype (torch.dtype): Type scalaire utilisé pour l'AMP.
        exp_name (str): L'identifiant textuel de l'expérience pour nommer les sauvegardes.
        start_epoch (int): L'époque de départ (supérieure à 1 en cas de reprise).
        best_val_loss (float): Le meilleur score historique à battre pour sauvegarder un nouveau checkpoint.
        fold (int, optional): Indice du pli de validation croisée, si applicable. Par défaut: -1 (inactif).
    """
    prefix = f"fold_{fold}/" if fold != -1 else ""
    wandb.define_metric(f"{prefix}val/pr_auc", step_metric=f"{prefix}epoch")
    wandb.define_metric(f"{prefix}train/loss", step_metric=f"{prefix}epoch")

    stagnation_counter = 0
    best_model_path = None
    accum_steps = max(1, args.batch_size_theoric // args.batch_size_accumulat)
    best_val_loss = float('inf') if best_val_loss < 0 else best_val_loss
    total_start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics_dict = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, 
            device, epoch, args.epochs, use_amp, amp_dtype, accum_steps, args.codebook_size
        )

        metrics = {
            f"{prefix}epoch": epoch,
            f"{prefix}train/lr": optimizer.param_groups[0]['lr'], 
        }
        for k, v in train_metrics_dict.items():
            metrics[f"{prefix}train/{k}"] = v

        if epoch >= args.val_start_epoch:
            val_loss, samples, is_valid = run_inference(model, val_loader, criterion, device, use_amp, amp_dtype)

            if is_valid:
                metrics[f"{prefix}val/loss"] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    stagnation_counter = 0

                    for f in glob.glob(os.path.join(args.checkpoint_dir, f"best_model_{exp_name}_ep*.pt")):
                        try: os.remove(f)
                        except OSError: pass

                    best_model_path = os.path.join(args.checkpoint_dir, f"best_model_{exp_name}_ep{epoch}.pt")
                    checkpoint = {
                        'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                        'best_score': best_val_loss
                    }
                    if scaler: checkpoint['scaler_state_dict'] = scaler.state_dict()
                    torch.save(checkpoint, best_model_path)
                    
                    wandb.run.summary["best_val_loss"] = best_val_loss
                    orig, recon = samples
                    wandb.log({
                        "reconstructions": [wandb.Image(to_wandb(o), caption="original") for o in orig] + 
                                        [wandb.Image(to_wandb(r), caption="reconstruction") for r in recon]
                    }, step=epoch)

                else:
                    stagnation_counter += 1
                    if stagnation_counter >= args.patience:
                        wandb.log(metrics)
                        break

        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch, 'model_state_dict': model.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss
            }
            if scaler: checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"backup_{exp_name}_ep{epoch}.pt"))

        scheduler.step()
        wandb.log(metrics)

    total_train_time_hours = (time.time() - total_start_time) / 3600.0
    wandb.run.summary["total_train_time_hours"] = total_train_time_hours

    if best_model_path and os.path.exists(best_model_path):
        shutil.copy2(best_model_path, os.path.join(wandb.run.dir, os.path.basename(best_model_path)))
        artifact_name = f"model_Unet-{wandb.run.id}" + (f"-fold{fold}" if fold != -1 else "")
        artifact = wandb.Artifact(artifact_name, type='model')
        artifact.add_file(os.path.join(wandb.run.dir, os.path.basename(best_model_path)))
        wandb.log_artifact(artifact)



def run(args):
    """
    Initialise les composants structurels et lance la procédure principale d'entraînement.
    Configure l'environnement matériel, initialise le réseau et ses poids, prépare les 
    flux de données et restaure éventuellement un état précédent à partir d'un fichier 
    avant de déléguer la gestion à la boucle d'entraînement.

    Args:
        args: Paramètres de configuration extraits de la ligne de commande.
    """
    # 1. Setup Global
    device, use_amp, amp_dtype = setup_global_environment(args)

    # 2. WandB (Gère la reprise d'ID automatiquement)
    wandb_id = wandb.util.generate_id()
    resume_mode = "allow"
    id_file = os.path.join(args.checkpoint_dir, "wandb_run_id.txt")

    if args.resume_from and os.path.exists(id_file):
        with open(id_file, "r") as f: wandb_id = f.read().strip() or wandb_id
        resume_mode = "must"

    model = UNet2d(
        input_dim=args.in_channels,
        output_dim=args.out_channels,
        hidden_dims=args.hidden_dims,
        kernel_size=args.kernel_size,
        padding_mode=args.padding_mode,
        skip_mode=args.skip_mode,
        upsampling_mode=args.upsampling_mode,
        dropout=args.dropout,
    )

    model.apply(weights_init)
    model = model.to(device)

    exp_name = generate_exp_name(args, wandb_id)

    setup_wandb(args, job_type="train", run_name=f"run_{exp_name}", wandb_id=wandb_id, resume_mode=resume_mode)
    with open(id_file, "w") as f: f.write(wandb.run.id)

    # 3. Dataloaders
    train_loader, val_loader, _ = get_food101_loaders(
        data_dir=args.data,
        batch_size=args.batch_size_accumulat,
        image_size=args.image_size,
    )

    # 4. Optimiseur & Loss
    optimizer, scheduler = configure_optimizers(model, args)

    criterion = nn.L1Loss()
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16)) if use_amp else None

    # 6. Reprise
    start_epoch, best_score = load_checkpoint(
        args.resume_from if args.resume_from else "", 
        model, 
        device, 
        optimizer, 
        scaler,
        scheduler=scheduler
    )
    model = torch.compile(model)

    if not args.resume_from and wandb.run.summary.get("best_val_loss"):
        best_score = float(wandb.run.summary["best_val_loss"])

    # 7. Lancement
    run_training_loop(args, model, train_loader, val_loader, optimizer, scheduler, criterion, scaler, device, use_amp, amp_dtype, exp_name, start_epoch, best_score)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Script d'entraînement standard", parents=[get_shared_parser()])
    parser.add_argument('--data', type=str, default='/Vrac/phd/data')

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_start_epoch', type=int, default=10)

    # Arguments Système
    parser.add_argument('--resume_from', type=str, default=None, 
                        help="Chemin vers un fichier .pt pour reprendre l'entraînement")

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
