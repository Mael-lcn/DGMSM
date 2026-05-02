import os

import torch
from torch.utils.data import DataLoader

from config import get_shared_parser
from flowMatching.ConditionalFlowMolecule import ConditionalFlowMolecule
from flowMatching.flow_matching import FlowMatchingEngine
from flowMatching.train_util import TrainLoop

from AlanineDipeptideChunkDataset import AlanineDipeptideChunkDataset 



def main():
    # Chargement de la configuration
    parser = get_shared_parser()
    args = parser.parse_args()

    # Configuration des variables d'environnement pour la boucle (logs)
    os.environ["OPENAI_LOGDIR"] = args.log_dir
    os.makedirs(args.log_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Lancement de l'entraînement sur : {device}")

    # Création du Cerveau et du Moteur
    print("Initialisation de l'architecture Mamba + Flow...")
    model = ConditionalFlowMolecule(
        mamba_in_channels=2,
        flow_in_channels=4,
        context_dim=args.mamba_dim,
        mamba_layers=args.mamba_layers,
        flow_channels=args.flow_channels,
        flow_blocks=args.flow_blocks
    )

    # Le moteur d'Optimal Transport
    diffusion_engine = FlowMatchingEngine(euler_steps=args.euler_steps)

    # Pipeline de données
    print("Chargement des trajectoires...")
    train_dataset = AlanineDipeptideChunkDataset(args.train_data, args.context_length)
    val_dataset = AlanineDipeptideChunkDataset(args.val_data, args.context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.workers,
        drop_last=True
    )

    # Un générateur infini pour la boucle d'entraînement
    def load_gen(loader):
        while True:
            yield from loader

    train_gen = load_gen(train_loader)

    # Lancement de la boucle d'entraînement
    print("Démarrage de la boucle d'entraînement...")
    loop = TrainLoop(
        model=model,
        device=device,
        diffusion=diffusion_engine,
        data=train_gen,
        val_loader=val_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint if args.resume_checkpoint else None,
        weight_decay=args.weight_decay,
        euler_steps=args.euler_steps
    )
    
    loop.run_loop(max_iter=args.max_iter)

if __name__ == "__main__":
    main()
