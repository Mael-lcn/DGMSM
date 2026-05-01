import argparse
import multiprocessing



def get_shared_parser():
    parser = argparse.ArgumentParser(description="Mamba + Flow Matching")

    group_sys = parser.add_argument_group("Configuration")
    group_sys.add_argument("--train_data", type=str, default="", help="Chemin dataset d'entraînement")
    group_sys.add_argument("--val_data", type=str, default="", help="Chemin dataset de validation")
    group_sys.add_argument("--log_dir", type=str, default="./logs/", help="Dossier de sortie (logs et plots)")
    group_sys.add_argument("--model_path", type=str, default="", help="Chemin vers le modèle (.pt) pour l'inférence")
    group_sys.add_argument("--eval_dir", type=str, default="./eval_results/", help="Dossier de sortie pour les tests")
    group_sys.add_argument("-w", "--workers", type=int, default=min(max(2, multiprocessing.cpu_count() - 1), 4))

    group_train = parser.add_argument_group("Entraînement")
    group_train.add_argument("--batch_size", type=int, default=64)
    group_train.add_argument("--microbatch", type=int, default=16)
    group_train.add_argument("--lr", type=float, default=1e-4)
    group_train.add_argument("--weight_decay", type=float, default=1e-4)
    group_train.add_argument("--max_iter", type=int, default=100000)
    group_train.add_argument("--log_interval", type=int, default=1000)
    group_train.add_argument("--save_interval", type=int, default=5000)
    group_train.add_argument("--resume_checkpoint", type=str, default="")

    group_model = parser.add_argument_group("Architecture")
    group_model.add_argument("--mamba_dim", type=int, default=256)
    group_model.add_argument("--mamba_layers", type=int, default=4)
    group_model.add_argument("--flow_channels", type=int, default=128)
    group_model.add_argument("--flow_blocks", type=int, default=4)
    group_model.add_argument("--euler_steps", type=int, default=20)

    return parser
