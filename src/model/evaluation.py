import os
import sys
import time
import math
import argparse

import wandb

project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.abspath(project_root))

from model_factory import get_shared_parser
from core_utils import (
    setup_global_environment,
    setup_wandb,
    run_inference_eval,
    load_checkpoint,
    get_food101_loaders,
    visualize_latent_interpolation,
    sample_from_latent
)
from Cae import UNet2d
from core_utils import to_wandb



def calculate_psnr(mse, max_val=1.0):
    """
    Calcule le Peak Signal-to-Noise Ratio (PSNR) à partir de l'erreur quadratique moyenne (MSE).

    Args:
        mse (float): L'erreur quadratique moyenne calculée entre l'image originale et la reconstruction.
        max_val (float): La valeur maximale possible pour un pixel de l'image (intensité maximale théorique).

    Returns:
        float: La valeur du PSNR en décibels (dB), ou l'infini si le MSE est nul ou négatif.
    """
    if mse <= 0: return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse)

def run(args):
    """
    orchestration de l'évaluation avec injection dynamique des métriques d'espace latent.
    """
    device, use_amp, amp_dtype = setup_global_environment(args)
    exp_name = args.checkpoint.replace(".pt", "") if args.checkpoint else "evaluation_run"
    setup_wandb(args, job_type="test", run_name=f"test_{exp_name}")

    _, _, test_loader = get_food101_loaders(
        data_dir=args.data, batch_size=args.batch_size_accumulat, image_size=args.image_size
    )

    model = UNet2d(
        input_dim=args.in_channels, output_dim=args.out_channels, hidden_dims=args.hidden_dims,
        kernel_size=args.kernel_size, padding_mode=args.padding_mode, skip_mode=args.skip_mode,
        upsampling_mode=args.upsampling_mode, dropout=args.dropout, use_rvq=args.use_rvq,
        num_quantizers=args.num_quantizers, codebook_size=args.codebook_size
    ).to(device)

    load_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint), model, device)

    start_time = time.time()
    
    # passage de la taille du dictionnaire pour le calcul d'activité
    metrics, (sample_origs, sample_recons) = run_inference_eval(
        model, test_loader, device, use_amp, amp_dtype, args.codebook_size
    )

    inference_time = time.time() - start_time

    interp_path = os.path.join(args.output, "plots", f"{exp_name}_interpolation.png")
    sample_path = os.path.join(args.output, "plots", f"{exp_name}_random_samples.png")

    interp_images = visualize_latent_interpolation(
        model=model, dataloader=test_loader, device=device, 
        save_path=interp_path, skip_mode=args.skip_mode, num_steps=8
    )

    random_images = None
    if args.skip_mode == "none":
        random_images = sample_from_latent(
            model=model, dataloader=test_loader, device=device, 
            save_path=sample_path, skip_mode=args.skip_mode, num_samples=8
        )

    def to_wandb_img(tensor_list, caption_prefix):
        return [wandb.Image((to_wandb(img)), caption=f"{caption_prefix} {i}") for i, img in enumerate(tensor_list)]

    wandb_logs = {
        "test_metrics_base/l1_loss": metrics['l1'],
        "test_metrics_base/mse_loss": metrics['mse'],
        "test_metrics_base/psnr_db": metrics['psnr'],
        "test_metrics_base/snr_db": metrics['snr'],

        "test_metrics_sota/ms_ssim": metrics['ms_ssim'],
        "test_metrics_sota/lpips": metrics['lpips'],
        "test_metrics_sota/fid": metrics['fid'],
        
        "performance/inference_time_sec": inference_time,

        "visuel/1_originaux": to_wandb_img(sample_origs[:4], "original"),
        "visuel/2_reconstructions": to_wandb_img(sample_recons[:4], "recon"),
        "visuel/3_latent_interpolation": to_wandb_img(interp_images, "alpha"),
    }
    
    # Mapping rigoureux de toutes les données RVQ
    for k, v in metrics.items():
        if k.startswith("rvq_"):
            wandb_logs[f"test_latent_health/{k}"] = v
    
    # ajout dynamique des métriques rvq au dictionnaire de log
    for k, v in metrics.items():
        if k.startswith("rvq_"):
            wandb_logs[f"test_latent_health/{k}"] = v

    if random_images is not None:
        wandb_logs["visuel/4_random_sampling"] = to_wandb_img(random_images, "random")

    wandb.log(wandb_logs)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Évaluation SOTA de l'Autoencodeur", parents=[get_shared_parser()])
    parser.add_argument('--data', type=str, default='/Vrac/phd/data', help="Chemin vers Food101")
    parser.add_argument('-c', '--checkpoint', required=True, help="Nom du fichier .pt")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
