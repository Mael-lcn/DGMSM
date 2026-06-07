import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from config import get_shared_parser
from flowMatching.ConditionalFlowMolecule import ConditionalFlowMolecule
from flowMatching.flow_matching import FlowMatchingEngine
from AlanineDipeptideChunkDataset import AlanineDipeptideChunkDataset 

from flowMatching.Evaluation import (
    Phase1Evaluator, run_phase_2_eval, fit_global_tica, fit_global_kmeans,
    to_sincos_flat, build_trans_matrix, compute_stationary_from_T
)
from flowMatching.EnergyGuidance import EnergyGuidance



def autoregressive_sampling_dynamic(model, diffusion_engine, initial_contexts, chunk_size, max_ar_steps, device, energy_guide=None, alpha=0.1):
    model.eval()
    B, C, context_length = initial_contexts.shape
    current_context = initial_contexts.clone()
    generated_trajectory = []

    with torch.no_grad():
        for _ in range(max_ar_steps):
            shape = (B, C, chunk_size)
            # On passe le guide ici au solveur
            pred_chunk = diffusion_engine.p_sample_loop(
                model=model, shape=shape, mamba_context=current_context, device=device,
                energy_guide=energy_guide, alpha=alpha 
            )
            generated_trajectory.append(pred_chunk.cpu().numpy())

            full_sequence = torch.cat([current_context, pred_chunk], dim=2)
            current_context = full_sequence[:, :, -context_length:]

    return np.concatenate(generated_trajectory, axis=2)


def load_continuous_simulation(file_pattern):
    """Charge la simulation réelle (Ground Truth) pour la Phase 2 et TICA."""
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"Aucune vraie simulation trouvée pour : {file_pattern}")
    continuous_trajs = []
    for file_path in files:
        with np.load(file_path) as npz_file:
            for key in npz_file.files:
                data = npz_file[key]
                traj = np.transpose(data).reshape(1, 2, -1)
                continuous_trajs.append(traj)
    return continuous_trajs


def main():
    parser = get_shared_parser()
    parser.add_argument("--max_ar_steps", type=int, default=100, help="Nombre de blocs concaténés (Phase 2)")
    parser.add_argument("--max_sampling_batches", type=int, default=10, help="Nombre de batchs testés (Phase 2)")
    parser.add_argument("--model_path", type=str, default="../../../output/logs/model_best_thermo_model.pt")
    parser.add_argument("--test_data", type=str, default="../../../output/dataset/test.npy")
    parser.add_argument("--true_sim_path", type=str, default="../../../data/*backbone-dihedrals.npz")
    parser.add_argument("--n_clusters", type=int, default=6, help="Conformations K-Means")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = ConditionalFlowMolecule(
        mamba_in_channels=4, flow_in_channels=4, context_dim=args.mamba_dim,
        mamba_layers=args.mamba_layers, flow_channels=args.flow_channels, flow_blocks=args.flow_blocks
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    diffusion_engine = FlowMatchingEngine(euler_steps=args.euler_steps)

    test_dataset = AlanineDipeptideChunkDataset(args.test_data, args.context_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("\n[Préparation] Chargement Ground Truth continu et Entraînement TICA...")
    true_continuous_trajs = load_continuous_simulation(args.true_sim_path)
    global_tica_model = fit_global_tica(true_continuous_trajs, lagtime=10)
    
    global_kmeans, fes_gt, xedges, yedges = fit_global_kmeans(
            true_continuous_trajs, 
            n_clusters=args.n_clusters, 
            save_dir=args.log_dir
        )

    energy_guide = EnergyGuidance(fes_gt, xedges, yedges)

    print("\n[Préparation] Calcul de la Cinétique Globale (Ground Truth)...")
    def discretize_global(t_list):
        dtrajs = []
        for traj in t_list:
            if isinstance(traj, torch.Tensor): traj = traj.cpu().numpy()
            B, _, T = traj.shape
            sc = to_sincos_flat(traj)
            cl = global_kmeans.predict(sc).reshape(B, T)
            dtrajs.extend([cl[i] for i in range(B)])
        return dtrajs

    gt_dtrajs_global = discretize_global(true_continuous_trajs)
    T_gt = build_trans_matrix(gt_dtrajs_global, args.n_clusters, lagtime=10)
    pi_gt = compute_stationary_from_T(T_gt)


    # ==========================================================
    # PHASE 1 : DYNAMIQUE LOCALE (1 CHUNK)
    # ==========================================================
    print("\n" + "="*60)
    print("PHASE 1 : ÉVALUATION SUR L'ENSEMBLE DES BLOCS COURTS")
    print("="*60)

    phase1_eval = Phase1Evaluator(max_mae_steps=3)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Génération Phase 1"):
            gt_batch, cond_batch = batch["GT"].to(device), batch["input"].to(device)
            pred_batch = diffusion_engine.p_sample_loop(
                model=model, shape=(gt_batch.shape[0], 2, gt_batch.shape[2]), mamba_context=cond_batch, device=device
            )
            phase1_eval.update(gt_batch, pred_batch)

    # On passe T_gt et pi_gt pour obtenir les nouvelles métriques comparatives
    metrics_p1 = phase1_eval.compute_and_log(
        step="eval", save_dir=args.log_dir, 
        global_kmeans=global_kmeans, tica_model=global_tica_model,
        T_gt=T_gt, pi_gt=pi_gt
    )

    # ==========================================================
    # PHASE 2 : DYNAMIQUE LONG TERME ET CINÉTIQUE
    # ==========================================================
    print("\n" + "="*60)
    print("PHASE 2 : SAMPLING LONG-TERME DYNAMIQUE")
    print("="*60)

    all_pred_trajs = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Génération Auto-régressive Phase 2")):
            if i >= args.max_sampling_batches: 
                break

            batch_traj = autoregressive_sampling_dynamic(
                model, diffusion_engine, batch["input"].to(device), 
                batch["GT"].shape[2], args.max_ar_steps, device,
                energy_guide=energy_guide, alpha=args.alpha
            )
            all_pred_trajs.append(batch_traj)

    # On passe T_gt et pi_gt à la phase 2
    metrics_p2 = run_phase_2_eval(
        true_continuous_trajs, all_pred_trajs, args.log_dir, 
        fes_gt=fes_gt, xedges=xedges, yedges=yedges,
        global_kmeans=global_kmeans, tica_model=global_tica_model,
        T_gt=T_gt, pi_gt=pi_gt
    )


    final_metrics = {**metrics_p1, **metrics_p2}
    df_metrics = pd.DataFrame([final_metrics])
    csv_path = os.path.join(args.log_dir, "comprehensive_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)

    # NOUVEAU : Affichage enrichi avec le Transport Optimal et la Thermo !
    print("\n" + "="*65)
    print("BILAN SOTA DÉFINITIF")
    print("="*65)
    print("--- [PHASE 1 : Dynamique Locale (16 pas)] ---")
    print(f"MAE Solveur ODE (3 pas)      : {metrics_p1.get('P1_MAE_Circ_Deg_3steps', 0):.2f}°")
    print(f"Distance Wasserstein (W1)    : {metrics_p1.get('P1_Wasserstein_Rad', 0):.4f} rad")
    print(f"MAE Matrice Transition       : {metrics_p1.get('P1_Transition_Matrix_MAE', 0):.4f}")
    print(f"MAE Dist. Stationnaire (pi)  : {metrics_p1.get('P1_Stationary_Dist_MAE', 0):.4f}")
    
    print("\n--- [PHASE 2 : Dynamique Macroscopique] ---")
    print(f"FES RMSE (Erreur Energie)    : {metrics_p2.get('P2_FES_RMSE_kcal_mol', 0):.2f} kcal/mol")
    print(f"Distance Wasserstein (W1)    : {metrics_p2.get('P2_Wasserstein_Rad', 0):.4f} rad")
    print(f"MAE Matrice Transition       : {metrics_p2.get('P2_Transition_Matrix_MAE', 0):.4f}")
    print(f"MAE Dist. Stationnaire (pi)  : {metrics_p2.get('P2_Stationary_Dist_MAE', 0):.4f}")
    print(f"Corrélation Pearson TICA     : {metrics_p2.get('P2_TICA_Pearson', 0):.3f}")
    
    # Affichage intelligent des temps d'implication (t2, t3, t4)
    t2_gt = metrics_p2.get('P2_t2_GT_Steps', 0)
    t2_pr = metrics_p2.get('P2_t2_Pred_Steps', 0)
    t3_gt = metrics_p2.get('P2_t3_GT_Steps', 0)
    t3_pr = metrics_p2.get('P2_t3_Pred_Steps', 0)
    
    print(f"\nTemps Relaxation t2 (GT/Pred): {t2_gt:.1f} / {t2_pr:.1f} pas")
    if t3_gt > 0:
        print(f"Temps Relaxation t3 (GT/Pred): {t3_gt:.1f} / {t3_pr:.1f} pas")
        
    print("\n" + "-"*65)
    print(f"-> Graphes (MFPT, CK-Test, Heatmaps) sauvegardés dans : {args.log_dir}")
    print("="*65)

if __name__ == "__main__":
    main()
