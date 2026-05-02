import os
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import get_shared_parser
from flowMatching.ConditionalFlowMolecule import ConditionalFlowMolecule
from flowMatching.flow_matching import FlowMatchingEngine
from flowMatching.Evaluation import plot_trajectory, TrajectoryMetrics, circular_mae

from AlanineDipeptideChunkDataset import AlanineDipeptideChunkDataset 



def main():
    # Chargement de la configuration
    parser = get_shared_parser()
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("ERREUR : On doit spécifier --model_path pour l'évaluation !")

    os.makedirs(args.eval_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nLancement de l'évaluation sur : {device}")

    # Création de l'architecture
    print(f"Chargement de l'architecture (Mamba Dim: {args.mamba_dim}, Flow Channels: {args.flow_channels})...")
    model = ConditionalFlowMolecule(
        mamba_in_channels=2,
        flow_in_channels=4,
        context_dim=args.mamba_dim,
        mamba_layers=args.mamba_layers,
        flow_channels=args.flow_channels,
        flow_blocks=args.flow_blocks
    )

    # Moteur d'inférence (Euler solver)
    diffusion_engine = FlowMatchingEngine(euler_steps=args.euler_steps)

    # Chargement des poids
    print(f"Chargement des poids depuis : {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # Préparation des données de test
    print("Chargement du Dataset de Validation/Test...")
    # On met limit_val à False si besoin, ou on utilise tout le fichier de val
    test_dataset = AlanineDipeptideChunkDataset(args.test_data, args.context_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers,
        drop_last=False
    )

    metrics = TrajectoryMetrics()

    # Boucle d'évaluation
    print(f"Début de l'inférence sur {len(test_dataset)} molécules (Steps d'Euler : {args.euler_steps})...")
    
    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(test_loader, desc="Évaluation")):
            gt_batch = batch["GT"].to(device)
            cond_batch = batch["input"].to(device)
            
            shape = (gt_batch.shape[0], 2, gt_batch.shape[2])
            
            # Génération
            pred_batch = diffusion_engine.p_sample_loop(
                model=model,
                shape=shape,
                mamba_context=cond_batch,
                device=device
            )

            # Calcul de la Loss
            metrics.update(gt_batch, pred_batch)

            if b_idx == 0:
                print("Génération des graphiques pour le premier batch...")
                # On sauvegarde jusqu'à 4 graphiques
                num_plots = min(4, gt_batch.shape[0])
                for i in range(num_plots):
                    # Calcule l'erreur spécifique pour CE graphe
                    mae_specifique = circular_mae(gt_batch[i].unsqueeze(0), pred_batch[i].unsqueeze(0)).item() * (180.0/math.pi)

                    plot_path = os.path.join(args.eval_dir, f"test_molecule_{i}.png")
                    plot_trajectory(gt_batch[i], pred_batch[i], mae_specifique, plot_path)

    res = metrics.compute()
    print("\n" + "="*50)
    print("RÉSULTATS DÉFINITIFS DE L'ÉVALUATION")
    print("="*50)
    print(f"Moyenne Erreur Circulaire (MAE) : {res['MAE_circ_deg']:.2f}°")
    print(f"Moyenne Erreur Quadratique (MSE) : {res['MSE_circ']:.4f}")
    print("="*50)
    print(f"Graphiques sauvegardés dans : {args.eval_dir}")


if __name__ == "__main__":
    main()
