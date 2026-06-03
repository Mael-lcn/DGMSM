import os
import math
import json
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import get_shared_parser
from flowMatching.ConditionalFlowMolecule import ConditionalFlowMolecule
from flowMatching.flow_matching import FlowMatchingEngine
from flowMatching.Evaluation import (
    plot_trajectory, TrajectoryMetrics, circular_mae,
    compute_ramachandran_jsd, plot_ramachandran,
    evaluate_vampnet_kinetics, plot_timescales_comparison
)
from AlanineDipeptideChunkDataset import AlanineDipeptideChunkDataset 



def autoregressive_rollout(model, diffusion_engine, initial_context, chunk_size, num_ar_steps, device):
    """
    Génère une longue trajectoire en réinjectant ses propres prédictions comme contexte.
    
    Args:
        initial_context: Tenseur de forme (B, 2, context_length)
        chunk_size: Nombre de pas générés à chaque itération du Flow Matching
        num_ar_steps: Nombre total de chunks à générer
    """
    model.eval()
    B, C, context_length = initial_context.shape
    current_context = initial_context.clone()
    
    # On stocke toute la trajectoire générée
    generated_trajectory = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_ar_steps), desc="Génération Auto-régressive"):
            shape = (B, C, chunk_size)
            
            # 1. Génération du prochain chunk
            pred_chunk = diffusion_engine.p_sample_loop(
                model=model, shape=shape, mamba_context=current_context, device=device
            )
            generated_trajectory.append(pred_chunk.cpu())
            
            # 2. Mise à jour du contexte (Glissement de la fenêtre)
            # On colle l'ancien contexte et la nouvelle prédiction, et on garde les 'context_length' derniers pas
            full_sequence = torch.cat([current_context, pred_chunk], dim=2)
            current_context = full_sequence[:, :, -context_length:]

    # Concaténation finale (B, 2, num_ar_steps * chunk_size)
    return torch.cat(generated_trajectory, dim=2).numpy()


def main():
    parser = get_shared_parser()
    # Ajout d'arguments spécifiques à l'inférence longue
    parser.add_argument("--ar_steps", type=int, default=50, help="Nombre de chunks à générer en auto-régression")
    parser.add_argument("--vampnet_path", type=str, default="", help="Chemin vers le modèle VAMPnet (.pt) pour l'évaluation cinétique")
    parser.add_argument("--baseline_path", type=str, default="", help="Chemin vers les timescales cibles (.pkl)")
    parser.add_argument("--model_path", type=str, default="../../../checkpoints/model_best_thermo_model.pt", help="Chemin vers les timescales cibles (.pkl)")

    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("ERREUR : Spécifiez --model_path pour évaluer le Flow Matching !")

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nLancement de l'évaluation globale sur : {device}")

    # 1. Chargement du modèle de Diffusion (Flow Matching + Mamba)
    model = ConditionalFlowMolecule(
        mamba_in_channels=2, flow_in_channels=4, context_dim=args.mamba_dim,
        mamba_layers=args.mamba_layers, flow_channels=args.flow_channels, flow_blocks=args.flow_blocks
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    diffusion_engine = FlowMatchingEngine(euler_steps=args.euler_steps)

    test_dataset = AlanineDipeptideChunkDataset(args.test_data, args.context_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("\n--- PHASE 1 : Évaluation Locale (Teacher-Forcing) ---")
    metrics = TrajectoryMetrics(max_eval_steps=3) 

    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(test_loader, desc="Test Court-Terme")):
            gt_batch = batch["GT"].to(device)
            cond_batch = batch["input"].to(device)
            
            # On ne fait qu'une seule prédiction par batch pour calculer la MAE
            pred_batch = diffusion_engine.p_sample_loop(
                model=model, shape=(gt_batch.shape[0], 2, gt_batch.shape[2]), mamba_context=cond_batch, device=device
            )
            metrics.update(gt_batch, pred_batch)

            if b_idx == 0:
                for i in range(min(4, gt_batch.shape[0])):
                    mae_spec = circular_mae(gt_batch[i].unsqueeze(0), pred_batch[i].unsqueeze(0)).item() * (180.0/math.pi)
                    plot_trajectory(gt_batch[i], pred_batch[i], mae_spec, os.path.join(args.log_dir, f"test_short_term_{i}.png"))

    res_short_term = metrics.compute()

    print("\n--- PHASE 2 : Rollout Auto-régressif (Long-Terme) ---")

    # On prend juste le TOUT PREMIER contexte du dataset comme graine
    seed_batch = next(iter(test_loader))
    initial_context = seed_batch["input"].to(device)
    chunk_size = seed_batch["GT"].shape[2] # Taille d'un chunk (ex: 10)

    # Génération de la trajectoire longue (shape: [Batch, 2, num_ar_steps * chunk_size])
    long_pred_trajs = autoregressive_rollout(
        model, diffusion_engine, initial_context, chunk_size, args.ar_steps, device
    )

    # Pour comparer la thermodynamique, on prend un échantillon équivalent de la vraie donnée
    long_gt_trajs = np.concatenate([b["GT"].numpy() for _, b in zip(range(args.ar_steps), test_loader)], axis=2)

    print("\nCalcul de la Thermodynamique (JSD)...")
    jsd_score = compute_ramachandran_jsd(long_gt_trajs, long_pred_trajs)
    plot_ramachandran(torch.tensor(long_gt_trajs), torch.tensor(long_pred_trajs), jsd_score, os.path.join(args.log_dir, "ramachandran_ar_eval.png"))

    timescale_errors = {}
    if args.vampnet_path and args.baseline_path:
        print("\n--- PHASE 3 : Évaluation Cinétique (VAMPnet) ---")
        try:
            # Chargement de la baseline et du juge
            vampnet_juge = torch.load(args.vampnet_path, map_location=device)
            with open(args.baseline_path, "rb") as f:
                baseline_data = pickle.load(f)
                target_timescales = baseline_data["timescales"]

            # VAMPnet lit nos trajectoires générées auto-régressivement
            gen_timescales = evaluate_vampnet_kinetics(long_pred_trajs, vampnet_juge, target_timescales, lag_time=1)

            if gen_timescales is not None:
                plot_timescales_comparison(target_timescales, gen_timescales, os.path.join(args.log_dir, "kinetics_timescales.png"))

                # Formatage pour le JSON
                n_t = min(len(target_timescales), len(gen_timescales))
                for i in range(n_t):
                    t_cible = float(target_timescales[i])
                    t_gen = float(gen_timescales[i])
                    timescale_errors[f"process_{i+1}"] = {
                        "target": t_cible, "generated": t_gen,
                        "error_relative_percent": abs(t_cible - t_gen) / t_cible * 100
                    }
        except Exception as e:
            print(f"Erreur lors de l'évaluation VAMPnet : {e}")
    else:
        print("\n[!] Paramètres --vampnet_path ou --baseline_path manquants. Évaluation cinétique ignorée.")

    report = {
        "1_local_dynamics_short_term": {
            "circular_mae_deg": float(res_short_term.get('MAE_circ_deg', 0)),
            "circular_mse_rad": float(res_short_term.get('MSE_circ', 0))
        },
        "2_thermodynamics_long_term": {
            "autoregressive_steps": args.ar_steps,
            "ramachandran_jsd": float(jsd_score)
        },
        "3_kinetics_vampnet": timescale_errors
    }

    report_path = os.path.join(args.log_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "="*50)
    print("ÉVALUATION TERMINÉE AVEC SUCCÈS")
    print("="*50)
    print(f"1. Précision Solveur ODE (MAE) : {report['1_local_dynamics_short_term']['circular_mae_deg']:.2f}°")
    print(f"2. Fidélité Thermodynamique (JSD) : {jsd_score:.4f}")
    if timescale_errors:
        err_moy = np.mean([v["error_relative_percent"] for v in timescale_errors.values()])
        print(f"3. Fidélité Cinétique VAMPnet (Erreur Moyenne) : {err_moy:.1f}%")
    print("="*50)
    print(f"Rapports et graphiques sauvegardés dans : {args.log_dir}")

if __name__ == "__main__":
    main()
