import os
import numpy as np
import torch
import wandb
from tqdm import tqdm
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import math



def get_food101_loaders(data_dir, batch_size, image_size):
    """
    Prépare et retourne les chargeurs de données (DataLoaders) pour le jeu de données Food101.
    Applique des transformations distinctes pour l'entraînement (avec augmentation) et 
    l'évaluation (pures), tout en garantissant une séparation stratifiée stricte sans fuite de données.

    Args:
        data_dir (str): Chemin vers le répertoire de stockage du jeu de données.
        batch_size (int): Nombre d'échantillons par lot.
        image_size (int): Dimension cible pour le redimensionnement spatial des images.

    Returns:
        tuple: Un tuple contenant les trois DataLoaders (train_loader, val_loader, test_loader).
    """
    print(f"Préparation des données Food101...")

    # 1. Définition des transformations
    train_transforms = T.Compose([
        T.Resize(image_size), T.RandomCrop(image_size), T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15), T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    val_transforms = T.Compose([
        T.Resize(image_size), T.CenterCrop(image_size),
        T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # 2. L'astuce pour éviter le bug des transformations
    # On crée une version "augmentée" et une version "pure" du dataset global
    print("Vérification du dataset...")
    train_part_aug = datasets.Food101(root=data_dir, split='train', download=True, transform=train_transforms)
    test_part_aug = datasets.Food101(root=data_dir, split='test', download=True, transform=train_transforms)
    full_dataset_aug = ConcatDataset([train_part_aug, test_part_aug])

    train_part_pure = datasets.Food101(root=data_dir, split='train', download=True, transform=val_transforms)
    test_part_pure = datasets.Food101(root=data_dir, split='test', download=True, transform=val_transforms)
    full_dataset_pure = ConcatDataset([train_part_pure, test_part_pure])

    # 3. Récupération des labels pour le split stratifié
    all_targets = train_part_pure._labels + test_part_pure._labels
    all_indices = list(range(len(full_dataset_pure)))

    # 4. Splitting AVEC random_state=42 (Garantit l'absence de data leak)
    keep_idx, keep_targets = all_indices, all_targets

    train_idx, temp_idx, _, temp_targets = train_test_split(keep_idx, keep_targets, test_size=0.30, stratify=keep_targets, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_targets, random_state=42)

    # 5. Création des Subsets avec la BONNE transformation
    train_dataset = Subset(full_dataset_aug, train_idx) # Seulement Train a la Data Augmentation
    val_dataset = Subset(full_dataset_pure, val_idx)    # Val est pur
    test_dataset = Subset(full_dataset_pure, test_idx)  # Test est pur

    # 6. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Tailles des splits -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader



def setup_global_environment(args):
    """
    Initialise l'environnement matériel, la gestion de la mémoire et la précision mixte (AMP).
    Configure l'allocation mémoire de PyTorch pour limiter la fragmentation, 
    crée les dossiers de sauvegarde nécessaires et configure le périphérique cible. 

    Args:
        args (argparse.Namespace): Objet contenant les configurations globales du système.

    Returns:
        tuple: Un tuple contenant :
            - device (torch.device): Le périphérique de calcul alloué.
            - use_amp (bool): État d'activation de la précision mixte.
            - amp_dtype (torch.dtype): Le type de tenseur utilisé pour l'AMP.
    """
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        device = torch.device(f"cuda:{args.gpu}")
        use_amp = not getattr(args, 'not_use_amp', False)

        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        if use_amp:
            print(f"[INIT] AMP activé ({'BFloat16' if amp_dtype == torch.bfloat16 else 'Float16'})")

        torch.backends.cudnn.benchmark = getattr(args, 'use_static_padding', False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    else:
        device, use_amp, amp_dtype = torch.device("cpu"), False, torch.float32
        print("[INIT] Mode CPU forcé.")

    return device, use_amp, amp_dtype


def setup_wandb(args, job_type, run_name, group=None, wandb_id=None, resume_mode="allow", tags=None):
    """
    Initialise une session Weights & Biases (WandB) de manière modulaire et isolée.
    Configure un environnement en mode hors ligne pour le suivi des expériences et 
    gère la création des répertoires de cache temporaires.

    Args:
        args (argparse.Namespace): Paramètres globaux de l'expérience.
        job_type (str): Le type de tâche exécutée.
        run_name (str): Le nom d'affichage de l'expérience sur le tableau de bord.
        group (str, optional): Le nom du groupe pour regrouper plusieurs exécutions.
        wandb_id (str, optional): Identifiant unique de l'exécution pour la reprise.
        resume_mode (str, optional): Stratégie de reprise de l'exécution.
        tags (list of str, optional): Liste de balises supplémentaires pour filtrer les expériences.

    Returns:
        str: L'identifiant unique (ID) initialisé pour la session.
    """
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()

    base_wandb_path = os.path.join(args.output, "wandb_logs")
    os.makedirs(base_wandb_path, exist_ok=True)

    os.environ["WANDB_MODE"] = "offline" 
    os.environ["WANDB_DIR"] = base_wandb_path
    os.environ["WANDB_CACHE_DIR"] = os.path.join(base_wandb_path, "cache")
    os.environ["TMPDIR"] = os.path.join(base_wandb_path, "tmp")
    os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    default_tags = ["offline"]
    if tags: default_tags.extend(tags)

    wandb.init(
        project="phd",
        group=group,
        job_type=job_type,
        name=run_name,
        config=vars(args),
        id=wandb_id,
        resume=resume_mode,
        tags=list(set(default_tags))
    )
    return wandb_id


@torch.no_grad()
def run_inference(model, dataloader, criterion, device, use_amp, amp_dtype, desc="[Inférence]"):
    """
    Exécute une boucle d'inférence universelle sans calcul de gradient.
    Gère la précision mixte (AMP), calcule la perte moyenne et surveille la validité numérique 
    des prédictions pour détecter les valeurs anormales (NaN/Inf).

    Args:
        model (torch.nn.Module): Le modèle d'apprentissage profond à évaluer.
        dataloader (torch.utils.data.DataLoader): Le chargeur de données fournissant les lots.
        criterion (callable): La fonction de coût (Loss) pour évaluer les reconstructions.
        device (torch.device): Le périphérique matériel utilisé pour l'inférence.
        use_amp (bool): État d'activation de la précision mixte.
        amp_dtype (torch.dtype): Type de données utilisé par l'AMP pour l'optimisation.
        desc (str, optional): Description textuelle affichée sur la barre de progression. Par défaut: "[Inférence]".

    Returns:
        tuple: Un tuple de résultats contenant :
            - avg_loss (float): La perte moyenne évaluée sur l'ensemble du jeu de données.
            - samples (tuple): Un lot d'images originales et leurs reconstructions respectives (origs, recons).
            - is_valid (bool): Indicateur confirmant l'absence de valeurs numériques corrompues (NaN/Inf).
    """
    model.eval()
    total_loss = 0.0
    count = 0
    is_valid = True
    sample_originals, sample_reconstructions = [], []

    for images, _ in tqdm(dataloader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp, dtype=amp_dtype):
            reconstructed, _, _ = model(images)
            loss = criterion(reconstructed, images)
            if not torch.isfinite(loss):
                is_valid = False

        total_loss += loss.item()
        count += 1

        if len(sample_originals) == 0:
            sample_originals = images[:8].cpu()
            sample_reconstructions = reconstructed[:8].cpu()

    return total_loss / max(1, count), (sample_originals, sample_reconstructions), is_valid



def calculate_metrics_batch(origs_01, recons_01):
    """
    Calcule conjointement les métriques d'évaluation standard (PSNR, SNR, MSE, L1) pour un lot d'images.
    Les tenseurs fournis doivent être préalablement normalisés dans l'intervalle [0, 1] 
    pour garantir la validité physique et mathématique des résultats en décibels (dB).

    Args:
        origs_01 (torch.Tensor): Tenseur contenant le lot d'images originales (min: 0.0, max: 1.0).
        recons_01 (torch.Tensor): Tenseur contenant le lot d'images reconstruites (min: 0.0, max: 1.0).

    Returns:
        tuple: Un tuple contenant les quatre métriques calculées sous forme de scalaires :
            - psnr_val (float): Peak Signal-to-Noise Ratio mesuré en décibels (dB).
            - snr_val (float): Signal-to-Noise Ratio mesuré en décibels (dB).
            - mse_val (float): Erreur quadratique moyenne (L2 Loss).
            - l1_val (float): Erreur absolue moyenne (L1 Loss).
    """
    mse_val = torch.mean((origs_01 - recons_01) ** 2).item()
    l1_val = torch.mean(torch.abs(origs_01 - recons_01)).item()
    signal_power = torch.mean(origs_01 ** 2).item()
    
    psnr_val = 20 * math.log10(1.0) - 10 * math.log10(mse_val) if mse_val > 0 else float('inf')
    snr_val = 10 * math.log10(signal_power / mse_val) if mse_val > 0 else float('inf')
    
    return psnr_val, snr_val, mse_val, l1_val


@torch.no_grad()
def run_inference_eval(model, dataloader, device, use_amp, amp_dtype, codebook_size):
    """
    Exécute une itération complète d'évaluation SOTA (State of the Art) sur un jeu de données.
    Intègre le calcul optimisé des métriques perceptuelles et génératives (FID, LPIPS, MS-SSIM) 
    ainsi qu'un suivi global et rigoureux de l'état de la quantification vectorielle (RVQ), 
    utilisant l'accumulation mathématique pour éviter toute saturation mémoire.

    Args:
        model (torch.nn.Module): Le modèle d'autoencodeur à évaluer.
        dataloader (torch.utils.data.DataLoader): Le chargeur de données d'évaluation (Test/Val).
        device (torch.device): Le périphérique matériel alloué pour les opérations matricielles.
        use_amp (bool): Activation de la précision mixte pour accélérer l'inférence.
        amp_dtype (torch.dtype): Le format de précision mixte ciblé (ex: torch.bfloat16).
        codebook_size (int): La capacité maximale du dictionnaire latent pour le calcul de l'utilisation.

    Returns:
        tuple: Un tuple regroupant les résultats analytiques et visuels :
            - metrics (dict): Dictionnaire exhaustif contenant toutes les métriques moyennes (l1, mse, psnr, snr, lpips, ms_ssim, fid) ainsi que la santé détaillée du codebook (active_pct, perplexity).
            - samples (tuple): Tenseurs contenant les 8 premières images originales et reconstruites du premier lot pour la visualisation.
    """
    from collections import defaultdict
    model.eval()

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    tot_l1, tot_mse, tot_psnr, tot_snr = 0.0, 0.0, 0.0, 0.0
    tot_lpips, tot_msssim = 0.0, 0.0
    count = 0
    tot_rvq_losses = defaultdict(float)

    # Accumulateur global pour la santé du codebook
    global_bincounts = None

    sample_originals, sample_reconstructions = [], []

    for images, _ in tqdm(dataloader, desc="[EVAL]", leave=False):
        images = images.to(device, non_blocking=True)

        with torch.amp.autocast('cuda' if use_amp else 'cpu', enabled=use_amp, dtype=amp_dtype):
            reconstructed, _, rvq_info = model(images)

        origs_01 = (images * 0.5 + 0.5).clamp(0, 1).float()
        recons_01 = (reconstructed * 0.5 + 0.5).clamp(0, 1).float()

        # Métriques standard
        b_psnr, b_snr, b_mse, b_l1 = calculate_metrics_batch(origs_01, recons_01)
        tot_psnr += b_psnr; tot_snr += b_snr; tot_mse += b_mse; tot_l1 += b_l1
    
        # Métriques perceptuelles
        tot_lpips += lpips_metric(recons_01, origs_01).item()
        tot_msssim += ms_ssim_metric(recons_01, origs_01).item()

        # Accumulation FID
        fid_metric.update(origs_01, real=True)
        fid_metric.update(recons_01, real=False)

        # Accumulation rigoureuse RVQ
        if 'metrics' in rvq_info:
            for k, v in rvq_info['metrics'].items():
                tot_rvq_losses[k] += v

        if 'indices' in rvq_info:
            indices = rvq_info['indices'] # Forme attendue : (B, num_quantizers, H, W)
            num_quantizers = indices.shape[1]

            # Initialisation de la matrice des comptes (sur GPU pour la vitesse, en float64 pour éviter l'overflow)
            if global_bincounts is None:
                global_bincounts = torch.zeros((num_quantizers, codebook_size), dtype=torch.float64, device=device)

            # Accumulation des occurrences niveau par niveau
            for q in range(num_quantizers):
                flat_idx = indices[:, q, :, :].reshape(-1)
                batch_bincount = torch.bincount(flat_idx, minlength=codebook_size).to(torch.float64)
                global_bincounts[q] += batch_bincount

        count += 1
        if len(sample_originals) == 0:
            sample_originals = images[:8].cpu()
            sample_reconstructions = reconstructed[:8].cpu()

    metrics = {
        "l1": tot_l1 / count,
        "mse": tot_mse / count,
        "psnr": tot_psnr / count,
        "snr": tot_snr / count,
        "lpips": tot_lpips / count,
        "ms_ssim": tot_msssim / count,
        "fid": fid_metric.compute().item()
    }

    # Pertes moyennes RVQ
    for k, v in tot_rvq_losses.items():
        metrics[f"rvq_{k}"] = v / count

    # Résolution globale de l'état du codebook (sans jamais avoir stocké les millions d'indices)
    if global_bincounts is not None:
        for q in range(global_bincounts.shape[0]):
            counts = global_bincounts[q]

            # 1. Utilisation active : Combien de codes ont un compte > 0 sur tout le dataset ?
            active_pct = (torch.sum(counts > 0).item() / codebook_size) * 100.0

            # 2. Perplexité exacte de la distribution globale
            probs = counts / counts.sum()
            entropy = -torch.sum(probs[probs > 0] * torch.log(probs[probs > 0]))
            perplexity = torch.exp(entropy).item()

            metrics[f"rvq_lvl_{q}/active_pct"] = active_pct
            metrics[f"rvq_lvl_{q}/perplexity"] = perplexity

    return metrics, (sample_originals, sample_reconstructions)


def calculate_codebook_metrics_per_level(all_indices, codebook_size):
    """
    Évalue mathématiquement la santé de l'espace latent pour chaque niveau de quantification (RVQ).
    Calcule le pourcentage d'utilisation active du dictionnaire (Active Codes) et la perplexité 
    de la distribution spatiale pour diagnostiquer un éventuel effondrement du codebook.

    Args:
        all_indices (torch.Tensor): Tenseur global contenant les indices discrets utilisés, de forme (B, num_quantizers, H, W).
        codebook_size (int): La taille théorique maximale du dictionnaire.

    Returns:
        dict: Un dictionnaire associant à chaque niveau de quantification (lvl_i) ses métriques de santé fonctionnelle (active_pct et perplexity).
    """
    metrics = {}
    num_levels = all_indices.shape[1]
    
    for i in range(num_levels):
        flat_indices = all_indices[:, i, :, :].reshape(-1)
        unique_indices = torch.unique(flat_indices)
        active_pct = (len(unique_indices) / codebook_size) * 100.0

        counts = torch.bincount(flat_indices, minlength=codebook_size).float()
        probs = counts / counts.sum()
        entropy = -torch.sum(probs[probs > 0] * torch.log(probs[probs > 0]))
        perplexity = torch.exp(entropy).item()

        metrics[f"lvl_{i}/active_pct"] = active_pct
        metrics[f"lvl_{i}/perplexity"] = perplexity
        
    return metrics


def load_checkpoint(checkpoint_path, model, device, optimizer=None, scaler=None, scheduler=None):
    """
    Restaure l'état d'un modèle et de son environnement d'entraînement à partir d'un point de sauvegarde.
    Gère le nettoyage des préfixes de compilation PyTorch lors de l'injection des poids.

    Args:
        checkpoint_path (str): Le chemin vers le fichier de sauvegarde.
        model (torch.nn.Module): Le modèle cible pour le chargement des poids.
        device (torch.device): Le périphérique cible pour le chargement.
        optimizer (torch.optim.Optimizer, optional): L'optimiseur à restaurer.
        scaler (torch.amp.GradScaler, optional): Le gestionnaire d'échelle de gradient à restaurer.
        scheduler (torch.optim.lr_scheduler, optional): Le planificateur de taux d'apprentissage à restaurer.

    Returns:
        tuple: Un tuple contenant :
            - start_epoch (int): L'époque à laquelle reprendre l'exécution.
            - best_score (float): Le meilleur score de validation précédemment enregistré.
    """
    if not os.path.exists(checkpoint_path):
        return 1, -1.0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.get('model_state_dict', checkpoint).items()}
    model.load_state_dict(state_dict, strict=False)

    start_epoch = checkpoint.get('epoch', 0) + 1
    best_score = checkpoint.get('best_score', checkpoint.get('best_val_loss', -1.0))

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return start_epoch, best_score



def plot_images(images, titles, save_path):
    """
    Génère et sauvegarde une grille visuelle à partir d'une liste de tenseurs d'images ou de tableaux NumPy.
    Applique la dé-normalisation nécessaire pour un affichage correct.

    Args:
        images (list): Liste de tenseurs PyTorch ou de tableaux NumPy représentant les images.
        titles (list of str): Titres associés à chaque image de la grille.
        save_path (str): Chemin complet de destination pour la sauvegarde du fichier image.
    """
    f, ax = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        ax = [ax]

    for i in range(len(images)):
        img_tensor = images[i]

        if hasattr(img_tensor, 'permute'):
            # Dé-normalisation (-1, 1 -> 0, 1) et passage en (H, W, C)
            img = (img_tensor.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
        else:
            img = img_tensor

        ax[i].imshow(img, cmap="gray" if img.shape[-1] == 1 else None, interpolation='nearest')
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    # Création du dossier si nécessaire et sauvegarde
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(f)


def to_wandb(img_tensor):
    """
    Convertit un tenseur d'image PyTorch en un format compatible avec l'affichage sur Weights & Biases.
    Effectue une permutation des dimensions pour passer au format standard, une dé-normalisation 
    des valeurs de pixels et une conversion finale en tableau NumPy.

    Args:
        img_tensor: Le tenseur de l'image à traiter.

    Returns:
        Tableau NumPy représentant l'image traitée pour la visualisation.
    """
    return (img_tensor.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().float().numpy()


def plot_results(curves, labels, save_path):
    """
    Génère et sauvegarde un graphique représentant l'évolution des courbes de métriques.

    Args:
        curves (list of list of float): Liste contenant les séries de données à tracer.
        labels (list of str): Liste des étiquettes de légende associées à chaque courbe.
        save_path (str): Chemin de destination pour la sauvegarde du graphique.
    """
    f = plt.figure()
    for i in range(len(curves)):
        plt.plot(curves[i], label=labels[i])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(f)


@torch.no_grad()
def visualize_latent_interpolation(model, dataloader, device, save_path, skip_mode, num_steps=8):
    """
    Réalise une interpolation linéaire dans l'espace latent entre deux images distinctes 
    et sauvegarde la progression visuelle de la reconstruction.

    Args:
        model (torch.nn.Module): Le modèle autoencodeur utilisé.
        dataloader (torch.utils.data.DataLoader): Chargeur de données pour extraire les images sources.
        device (torch.device): Périphérique de calcul.
        save_path (str): Chemin de sauvegarde de l'image générée.
        skip_mode (str): Configuration des connexions résiduelles pour le décodage.
        num_steps (int, optional): Nombre d'étapes intermédiaires pour l'interpolation.

    Returns:
        list: Liste des tenseurs d'images générés lors de l'interpolation.
    """
    model.eval()
    batch, _ = next(iter(dataloader))
    img_a, img_b = batch[0:1].to(device), batch[1:2].to(device)

    z_a, skips_a = model.encode(img_a)
    z_b, skips_b = model.encode(img_b)

    alphas = np.linspace(0, 1, num_steps)
    interpolation_results = []

    for alpha in alphas:
        z_interp = (1 - alpha) * z_a + alpha * z_b
        # On utilise les skips de l'image A si on n'est pas en skip_mode="none"
        res = model.decode(z_interp, skips_a if skip_mode != "none" else None)
        interpolation_results.append(res.cpu().squeeze(0))

    plot_images(interpolation_results, [f"t={a:.2f}" for a in alphas], save_path)
    return interpolation_results


@torch.no_grad()
def sample_from_latent(model, dataloader, device, save_path, skip_mode, num_samples=8):
    """
    Génère un ensemble de nouvelles images en échantillonnant des vecteurs de bruit aléatoires 
    directement dans l'espace latent, puis sauvegarde les résultats visuels.

    Args:
        model (torch.nn.Module): Le modèle générateur/décodeur.
        dataloader (torch.utils.data.DataLoader): Chargeur de données utilisé pour identifier la dimension latente.
        device (torch.device): Périphérique de calcul.
        save_path (str): Chemin de sauvegarde de la grille d'images générées.
        skip_mode (str): Configuration des connexions résiduelles.
        num_samples (int, optional): Nombre total d'images indépendantes à générer.

    Returns:
        list: Liste des tenseurs d'images artificiellement générées.
    """
    model.eval()

    batch, _ = next(iter(dataloader))
    dummy_z, _ = model.encode(batch[0:1].to(device))
    latent_shape = dummy_z.shape[1:] # ex: (1024, 16, 16)

    # Génération du bruit avec la bonne shape
    z_random = torch.randn(num_samples, *latent_shape).to(device)

    # Décodage
    # Note : Si skip_mode="concat" ou "add", générer du bruit est très dur car 
    # le décodeur attend aussi des tenseurs spatiaux de haute résolution (les skips).
    # On passe None et on espère que le réseau gère (ou on force skip_mode="none").
    generated = model.decode(z_random, skip_features=None)

    gen_list = [generated[i].cpu() for i in range(num_samples)]
    plot_images(gen_list, [f"Sample {i}" for i in range(num_samples)], save_path)

    return gen_list
