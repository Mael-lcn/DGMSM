import math
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import torch



def angular_difference(gt, pred):
    """
    Calcule la différence angulaire la plus courte sur le cercle.
    Garantit que la différence entre 179° et -179° est de 2°, et non 358°.
    Retourne un tenseur de différences comprises entre -pi et +pi.
    """
    return (pred - gt + math.pi) % (2 * math.pi) - math.pi

def circular_mse(gt, pred):
    """
    Compute Circular Mean Squared Error (MSE).
    gt et pred doivent être en radians.
    """
    diff = angular_difference(gt, pred)
    return torch.mean(diff ** 2)

def circular_mae(gt, pred):
    """
    Compute Circular Mean Absolute Error (MAE).
    Souvent plus interprétable physiquement que la MSE (erreur moyenne en radians).
    """
    diff = angular_difference(gt, pred)
    return torch.mean(torch.abs(diff))


def plot_trajectory(gt, pred, mae_score, save_path):
    """Génère un graphique de comparaison Vrai vs Prédit."""
    gt_cpu = gt.cpu().numpy()
    pred_cpu = pred.cpu().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    time_axis = range(gt_cpu.shape[1])

    # Angle Phi
    axs[0].plot(time_axis, gt_cpu[0], label="Vrai (GT)", color="#1f77b4", marker='o')
    axs[0].plot(time_axis, pred_cpu[0], label="Prédiction Flow", color="#d62728", marker='x', linestyle='dashed')
    axs[0].set_title("Angle Dièdre Phi (φ)")
    axs[0].set_ylabel("Angle (Radians)")
    axs[0].set_xlabel("Pas de temps")
    axs[0].set_ylim(-math.pi, math.pi)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Angle Psi
    axs[1].plot(time_axis, gt_cpu[1], label="Vrai (GT)", color="#1f77b4", marker='o')
    axs[1].plot(time_axis, pred_cpu[1], label="Prédiction Flow", color="#d62728", marker='x', linestyle='dashed')
    axs[1].set_title("Angle Dièdre Psi (ψ)")
    axs[1].set_xlabel("Pas de temps")
    axs[1].set_ylim(-math.pi, math.pi)
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    fig.suptitle(f"Évaluation Flow Matching | Erreur moyenne: {mae_score:.2f}°", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def compute_ramachandran_jsd(gt_cpu, pred_cpu, bins=60):
    """
    Calcule la Divergence de Jensen-Shannon (JSD) entre les distributions 2D
    de Ramachandran réelles et générées. Une JSD proche de 0 est parfaite.
    """
    # Extraction et aplatissement des angles Phi (index 0) et Psi (index 1)
    gt_phi, gt_psi = gt_cpu[:, 0, :].flatten(), gt_cpu[:, 1, :].flatten()
    pred_phi, pred_psi = pred_cpu[:, 0, :].flatten(), pred_cpu[:, 1, :].flatten()

    hist_range = [[-math.pi, math.pi], [-math.pi, math.pi]]

    # Création des histogrammes de probabilité
    hist_gt, _, _ = np.histogram2d(gt_phi, gt_psi, bins=bins, range=hist_range, density=True)
    hist_pred, _, _ = np.histogram2d(pred_phi, pred_psi, bins=bins, range=hist_range, density=True)

    p_gt = hist_gt.flatten()
    p_pred = hist_pred.flatten()

    # Normalisation
    p_gt /= p_gt.sum() + 1e-10
    p_pred /= p_pred.sum() + 1e-10

    return jensenshannon(p_gt, p_pred)


def plot_ramachandran(gt, pred, jsd_score, save_path):
    """
    Trace les cartes d'Énergie Libre (Ramachandran Plots).
    """
    gt_cpu = gt.cpu().numpy()
    pred_cpu = pred.cpu().numpy()

    gt_phi, gt_psi = gt_cpu[:, 0, :].flatten(), gt_cpu[:, 1, :].flatten()
    pred_phi, pred_psi = pred_cpu[:, 0, :].flatten(), pred_cpu[:, 1, :].flatten()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    axs[0].hist2d(gt_phi, gt_psi, bins=60, range=[[-math.pi, math.pi], [-math.pi, math.pi]], cmap='Blues', density=True)
    axs[0].set_title("Vrai (Ground Truth)")
    axs[0].set_xlabel("Phi (rad)")
    axs[0].set_ylabel("Psi (rad)")

    axs[1].hist2d(pred_phi, pred_psi, bins=60, range=[[-math.pi, math.pi], [-math.pi, math.pi]], cmap='Reds', density=True)
    axs[1].set_title("Généré (Flow Matching)")
    axs[1].set_xlabel("Phi (rad)")

    fig.suptitle(f"Ramachandran Plot | Divergence JSD : {jsd_score:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


class TrajectoryMetrics:
    """
    Garde en mémoire les statistiques des erreurs sur l'ensemble d'une époque (Validation/Test).
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.count = 0

    def update(self, gt, pred):
        """
        Met à jour les compteurs avec un nouveau batch.
        """
        batch_size = gt.size(0)
        
        # On multiplie par le batch_size pour avoir la somme totale de l'erreur
        self.mse_sum += circular_mse(gt, pred).item() * batch_size
        self.mae_sum += circular_mae(gt, pred).item() * batch_size
        self.count += batch_size

    def compute(self):
        """
        Retourne la moyenne des métriques.
        """
        if self.count == 0:
            return {"MSE_circ": 0.0, "MAE_circ_rad": 0.0, "MAE_circ_deg": 0.0}

        mean_mse = self.mse_sum / self.count
        mean_mae_rad = self.mae_sum / self.count
        
        # Conversion en degrés
        mean_mae_deg = mean_mae_rad * (180.0 / math.pi)

        return {
            "MSE_circ": mean_mse,
            "MAE_circ_rad": mean_mae_rad,
            "MAE_circ_deg": mean_mae_deg
        }

    def __repr__(self):
        metrics = self.compute()
        return f"Circular MSE: {metrics['MSE_circ']:.4f} | Circular MAE: {metrics['MAE_circ_deg']:.2f}°"
