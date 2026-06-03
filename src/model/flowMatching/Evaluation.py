import math
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import torch

import deeptime.markov as markov



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
    Calcule les statistiques d'erreurs (MSE/MAE) UNIQUEMENT sur le court-terme 
    pour éviter la divergence chaotique.
    """
    def __init__(self, max_eval_steps=3):
        self.max_eval_steps = max_eval_steps
        self.reset()

    def reset(self):
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.count = 0

    def update(self, gt, pred):
        # On ne prend que les 'max_eval_steps' premiers pas pour la MSE/MAE !
        # Shape attendue: (Batch, 2, L)
        steps = min(self.max_eval_steps, gt.shape[2])
        gt_short = gt[:, :, :steps]
        pred_short = pred[:, :, :steps]
        
        batch_size = gt.size(0)
        self.mse_sum += circular_mse(gt_short, pred_short).item() * batch_size
        self.mae_sum += circular_mae(gt_short, pred_short).item() * batch_size
        self.count += batch_size

    def compute(self):
        if self.count == 0: return {}
        mean_mae_rad = self.mae_sum / self.count
        return {
            "MSE_circ": self.mse_sum / self.count,
            "MAE_circ_deg": mean_mae_rad * (180.0 / math.pi)
        }

def evaluate_vampnet_kinetics(pred_trajs_cpu, vampnet_model, target_timescales, lag_time=1):
    """
    Utilise le VAMPnet pré-entraîné pour évaluer la cinétique des trajectoires générées.
    
    Args:
        pred_trajs_cpu: Numpy array des trajectoires générées (N, 2, L)
        vampnet_model: Le modèle PyTorch/Deeptime VAMPnet entraîné (Le Juge)
        target_timescales: Les temps implicites de la vraie physique
    """
    print("\n--- ÉVALUATION CINÉTIQUE VIA VAMPNET ---")
    
    # 1. Transformation Topologique (VAMPnet a besoin de sin/cos, pas des angles bruts)
    phi = pred_trajs_cpu[:, 0, :]
    psi = pred_trajs_cpu[:, 1, :]
    
    sincos_trajs = np.stack([
        np.sin(phi), np.cos(phi), 
        np.sin(psi), np.cos(psi)
    ], axis=-1).astype(np.float32) # Shape: (N, L, 4)

    # 2. VAMPnet lit les trajectoires
    dtrajs_gen = []
    for traj in sincos_trajs:
        # Probabilités (Soft assignments)
        probs = vampnet_model.transform(traj)
        # Assignations strictes (Hard assignments) pour le Modèle de Markov
        hard_labels = np.argmax(probs, axis=1)
        dtrajs_gen.append(hard_labels)

    # 3. Construction de la Matrice de Transition sur les données générées
    try:
        count_estimator = markov.TransitionCountEstimator(lagtime=lag_time, count_mode="sliding")
        counts_gen = count_estimator.fit(dtrajs_gen).fetch_model()

        msm_estimator = markov.msm.MaximumLikelihoodMSM(reversible=True)
        gen_model = msm_estimator.fit(counts_gen).fetch_model()

        gen_transition_matrix = gen_model.transition_matrix
        gen_timescales = gen_model.timescales()

        # 4. Affichage et Comparaison
        print(">> Comparaison des Temps Implicites (Implied Timescales) :")
        n_timescales = min(len(target_timescales), len(gen_timescales))

        for i in range(n_timescales):
            t_cible = target_timescales[i]
            t_gen = gen_timescales[i]
            erreur = abs(t_cible - t_gen) / t_cible * 100
            print(f"  - Processus lent n°{i+1} | Vrai: {t_cible:.2f} | Généré: {t_gen:.2f} -> Erreur relative: {erreur:.1f}%")

        print("\n>> Matrice de Transition Générée :")
        print(np.round(gen_transition_matrix, 3))

        return gen_timescales

    except Exception as e:
        print(f"Échec du Modèle de Markov. Les trajectoires générées ne visitent peut-être pas tous les états : {e}")
        return None
