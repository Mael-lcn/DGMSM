import argparse
import multiprocessing


def get_shared_parser():
    """Retourne un parser avec les arguments communs à Train et Eval."""
    parser = argparse.ArgumentParser(add_help=False)

    # --- 1. PARAMÈTRES SYSTÈME ET DONNÉES ---
    group_sys = parser.add_argument_group("Configuration Système & Fichiers")
    group_sys.add_argument(
        "--output",
        type=str,
        default="../../../output/",
        help="Dossier de sortie standard",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="../../../checkpoints/",
        help="Dossier où sauvegarder les poids (.pt)",
    )

    group_sys.add_argument(
        "-w",
        "--workers",
        type=int,
        default=min(max(4, multiprocessing.cpu_count() - 1), 8),
        help="Nombre de processus pour charger les données",
    )
    group_sys.add_argument(
        "--gpu", type=int, default=0, help="index du GPU a utiliser (config HPC)"
    )

    # --- 2. Paramètres du dataloading communs ---
    group_train = parser.add_argument_group(
        "Hyperparamètres Communs du Dataloader/Inférence"
    )
    group_train.add_argument(
        "--batch_size_theoric",
        type=int,
        default=64,
        help="Taille du batch de MAJ du gradient",
    )
    group_train.add_argument(
        "--batch_size_accumulat",
        type=int,
        default=64,
        help="Taille du batch d'inference",
    )

    group_train.add_argument(
        "--not_use_amp",
        action="store_false",
        default=False,
        help="Désactive l'Automatic Mixed Precision (AMP). Passe en FP32.",
    )

    # --- 3. ARCHITECTURE DU MODÈLE (U-Net Autoencodeur) ---
    group_model = parser.add_argument_group("3. Architecture du Modèle")
    group_model.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Taille de redimensionnement des images (H, W)",
    )
    group_model.add_argument(
        "--in_channels", type=int, default=3, help="Nombre de canaux en entrée (RGB)"
    )
    group_model.add_argument(
        "--out_channels",
        type=int,
        default=3,
        help="Nombre de canaux en sortie (Reconstruction RGB)",
    )

    # nargs='+' permet de passer une liste via le terminal
    group_model.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[64, 128, 256, 256, 128],
        help="Liste des dimensions cachées de l'encodeur/décodeur",
    )
    group_model.add_argument(
        "--kernel_size", type=int, default=3, help="Taille du kernel convolutif"
    )

    group_model.add_argument(
        "--padding_mode",
        type=str,
        default="same",
        choices=["same", "valid"],
        help="Mode de padding ('zeros' garde la taille de l'image intacte)",
    )
    group_model.add_argument(
        "--skip_mode",
        type=str,
        default="none",
        choices=["concat", "add", "none"],
        help="Mode des skip-connections. (Alerte : mettre 'none' pour un vrai Autoencodeur)",
    )
    group_model.add_argument(
        "--upsampling_mode",
        type=str,
        default="transpose",
        choices=["transpose", "bilinear"],
        help="Méthode d'upsampling dans le décodeur",
    )
    group_model.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Taux de dropout pour la régularisation (0.0 = désactivé)",
    )

    return parser
