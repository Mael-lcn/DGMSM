import argparse



def get_shared_parser():
    """
    Parseur d'arguments centralisé pour le projet DGMSM (Mamba + Flow Matching).
    
    Cette configuration regroupe tous les paramètres nécessaires à l'entraînement,
    à l'architecture du modèle et aux procédures d'évaluation physique pour l'Alanine Dipeptide.
    """
    parser = argparse.ArgumentParser(
        description="Configuration SOTA : Extending Deep Generative Markov State Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1. PARAMÈTRES SYSTÈME ET CHEMINS (PATHS)
    group_sys = parser.add_argument_group("Système & Données")
    group_sys.add_argument("--gpu", type=int, default=0,
                           help="ID du GPU cible (ex: 0, 1).")
    group_sys.add_argument("--workers", type=int, default=4, 
                           help="Nombre de processus CPU pour le chargement des données.")
    group_sys.add_argument("--train_data", type=str, default="../../../output/dataset/train.npy", 
                           help="Chemin vers le fichier de trajectoires d'entraînement.")
    group_sys.add_argument("--val_data", type=str, default="../../../output/dataset/val.npy", 
                           help="Chemin vers le fichier de trajectoires de validation.")
    group_sys.add_argument("--log_dir", type=str, default="../../../output/logs/", 
                           help="Répertoire de sortie pour les checkpoints, les courbes et les plots.")

    # 2. HYPERPARAMÈTRES D'ENTRAÎNEMENT
    group_train = parser.add_argument_group("Optimisation & Entraînement")
    group_train.add_argument("--batch_size", type=int, default=64, 
                             help="Taille du batch global.")
    group_train.add_argument("--microbatch", type=int, default=64, 
                             help="Taille du microbatch pour l'accumulation de gradient (stabilité SOTA).")
    group_train.add_argument("--lr", type=float, default=5e-5, 
                             help="Taux d'apprentissage (Learning Rate) initial.")
    group_train.add_argument("--weight_decay", type=float, default=5e-4, 
                             help="Pénalité L2 appliquée sélectivement (hors biais/normes).")
    group_train.add_argument("--grad_clip", type=float, default=1.0, 
                             help="Seuil de clipping des gradients pour éviter les explosions (crucial pour SSM/Mamba).")
    group_train.add_argument("--max_iter", type=int, default=150000, 
                             help="Nombre total d'itérations d'entraînement.")
    group_train.add_argument("--resume_checkpoint", type=str, default="", 
                             help="Chemin vers un modèle .pt existant pour reprendre l'entraînement.")

    # 3. ARCHITECTURE DU MODÈLE (MAMBA + FLOW)
    group_model = parser.add_argument_group("Architecture")
    group_model.add_argument("--context_length", type=int, default=16, 
                             help="Longueur de la fenêtre passée (Input) pour l'encodeur Mamba.")
    group_model.add_argument("--mamba_dim", type=int, default=256, 
                             help="Dimension de l'espace latent (Mamba Hidden State).")
    group_model.add_argument("--mamba_layers", type=int, default=4, 
                             help="Nombre de blocs Mamba-2 empilés.")
    group_model.add_argument("--flow_channels", type=int, default=128, 
                             help="Largeur (canaux) des couches du FlowNetwork1D.")
    group_model.add_argument("--flow_blocks", type=int, default=4, 
                             help="Nombre de blocs résiduels dans le prédicteur de vitesse.")
    group_model.add_argument("--dropout", type=float, default=0.2, 
                             help="Taux de dropout (actif dans Mamba et FlowNetwork).")

    # 2. HYPERPARAMÈTRES D'ENTRAÎNEMENT
    group_eval = parser.add_argument_group("Évaluation & Inférence")
    group_eval.add_argument("--euler_steps", type=int, default=50, 
                            help="Nombre de pas d'intégration pour le solveur d'Euler à l'inférence.")
    group_eval.add_argument("--log_interval", type=int, default=5000, 
                            help="Fréquence de logging et d'affichage des courbes de loss.")
    group_eval.add_argument("--save_interval", type=int, default=10000, 
                            help="Fréquence de sauvegarde du checkpoint latest.")

    return parser
