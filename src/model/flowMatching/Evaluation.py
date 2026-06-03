import math
import numpy as np
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import torch

import deeptime.markov as markov
import deeptime.plots as dtplots



def angular_difference(gt, pred):
    """
    Calcule la différence angulaire la plus courte sur le cercle.
    Garantit que la différence entre 179° et -179° est de 2°, et non 358°.
    Retourne un tenseur de différences comprises entre -pi et +pi.
    """
    return (pred - gt + math.pi) % (2 * math.pi) - math.pi

def circular_mse(gt, pred):
    """Compute Circular Mean Squared Error (MSE)."""
    diff = angular_difference(gt, pred)
    return torch.mean(diff ** 2)

def circular_mae(gt, pred):
    """Compute Circular Mean Absolute Error (MAE)."""
    diff = angular_difference(gt, pred)
    return torch.mean(torch.abs(diff))

def plot_trajectory(gt, pred, mae_score, save_path):
    """Génère un graphique de comparaison temporel Vrai vs Prédit."""
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

class TrajectoryMetrics:
    """
    Calcule les statistiques d'erreurs (MSE/MAE) UNIQUEMENT sur le court-terme 
    (ex: les 3 premiers pas) pour éviter la divergence chaotique naturelle.
    """
    def __init__(self, max_eval_steps=3):
        self.max_eval_steps = max_eval_steps
        self.reset()

    def reset(self):
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.count = 0

    def update(self, gt, pred):
        # On ne prend que les 'max_eval_steps' premiers pas !
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

# =====================================================================
# 2. MÉTRIQUES THERMODYNAMIQUES (STATIQUE / ÉQUILIBRE)
# =====================================================================

def compute_ramachandran_jsd(gt_cpu, pred_cpu, bins=60):
    """Calcule la Divergence de Jensen-Shannon (JSD) sur les Ramachandran."""
    gt_phi, gt_psi = gt_cpu[:, 0, :].flatten(), gt_cpu[:, 1, :].flatten()
    pred_phi, pred_psi = pred_cpu[:, 0, :].flatten(), pred_cpu[:, 1, :].flatten()

    hist_range = [[-math.pi, math.pi], [-math.pi, math.pi]]
    hist_gt, _, _ = np.histogram2d(gt_phi, gt_psi, bins=bins, range=hist_range, density=True)
    hist_pred, _, _ = np.histogram2d(pred_phi, pred_psi, bins=bins, range=hist_range, density=True)

    p_gt = hist_gt.flatten()
    p_pred = hist_pred.flatten()
    p_gt /= p_gt.sum() + 1e-10
    p_pred /= p_pred.sum() + 1e-10

    return jensenshannon(p_gt, p_pred)

def plot_ramachandran(gt, pred, jsd_score, save_path):
    """Trace les cartes d'Énergie Libre 2D (Ramachandran Plots)."""
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

def plot_1d_free_energy(gt_cpu, pred_cpu, save_path, bins=100):
    """
    Trace les profils d'Énergie Libre 1D le long de Phi et Psi.
    (Inspiré de Shen et al. 2025 pour voir la hauteur des barrières d'énergie).
    """
    gt_phi, gt_psi = gt_cpu[:, 0, :].flatten(), gt_cpu[:, 1, :].flatten()
    pred_phi, pred_psi = pred_cpu[:, 0, :].flatten(), pred_cpu[:, 1, :].flatten()

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    for ax, gt_data, pred_data, angle_name in zip(axs, [gt_phi, gt_psi], [pred_phi, pred_psi], ["Phi", "Psi"]):
        hist_gt, edges = np.histogram(gt_data, bins=bins, range=[-math.pi, math.pi], density=True)
        hist_pred, _ = np.histogram(pred_data, bins=bins, range=[-math.pi, math.pi], density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        
        fe_gt = -np.log(hist_gt + 1e-10)
        fe_pred = -np.log(hist_pred + 1e-10)
        
        fe_gt -= np.min(fe_gt)
        fe_pred -= np.min(fe_pred)
        
        ax.plot(centers, fe_gt, label='Vrai (GT)', color='#1f77b4', linewidth=2)
        ax.plot(centers, fe_pred, label='Généré (Flow)', color='#d62728', linestyle='dashed', linewidth=2)
        
        ax.set_title(f"Énergie Libre le long de {angle_name}")
        ax.set_xlabel(f"Angle {angle_name} (rad)")
        ax.set_ylabel("Énergie Libre (kT)")
        
        y_max = max(np.percentile(fe_gt, 95), np.percentile(fe_pred, 95)) + 2
        ax.set_ylim(0, y_max)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =====================================================================
# 3. MÉTRIQUES CINÉTIQUES (DYNAMIQUE LONG-TERME VIA VAMPNET)
# =====================================================================

def evaluate_vampnet_kinetics(pred_trajs_cpu, vampnet_model, target_timescales, lag_time=1):
    """
    Utilise le VAMPnet pré-entraîné pour évaluer la cinétique des trajectoires générées.
    """
    print("\n--- ÉVALUATION CINÉTIQUE VIA VAMPNET ---")
    
    # 1. Transformation Topologique
    phi = pred_trajs_cpu[:, 0, :]
    psi = pred_trajs_cpu[:, 1, :]
    sincos_trajs = np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1).astype(np.float32)

    # 2. VAMPnet lit les trajectoires
    dtrajs_gen = []
    for traj in sincos_trajs:
        probs = vampnet_model.transform(traj)
        hard_labels = np.argmax(probs, axis=1)
        dtrajs_gen.append(hard_labels)

    # 3. Construction de la Matrice de Transition générée
    try:
        count_estimator = markov.TransitionCountEstimator(lagtime=lag_time, count_mode="sliding")
        counts_gen = count_estimator.fit(dtrajs_gen).fetch_model()

        msm_estimator = markov.msm.MaximumLikelihoodMSM(reversible=True)
        gen_model = msm_estimator.fit(counts_gen).fetch_model()

        gen_transition_matrix = gen_model.transition_matrix
        gen_timescales = gen_model.timescales()

        # 4. Affichage console
        print(">> Comparaison des Temps Implicites (Implied Timescales) :")
        n_timescales = min(len(target_timescales), len(gen_timescales))

        for i in range(n_timescales):
            t_cible = target_timescales[i]
            t_gen = gen_timescales[i]
            erreur = abs(t_cible - t_gen) / t_cible * 100
            print(f"  - Processus lent n°{i+1} | Vrai: {t_cible:.2f} | Généré: {t_gen:.2f} -> Erreur relative: {erreur:.1f}%")

        print("\n>> Matrice de Transition Générée :")
        print(np.round(gen_transition_matrix, 3))

        # IMPORTANT : Retourne les timescales ET le modèle pour le CK-test
        return gen_timescales, gen_model

    except Exception as e:
        print(f"Échec du Modèle de Markov. Les trajectoires générées ne visitent peut-être pas tous les états : {e}")
        return None, None

def plot_timescales_comparison(target_timescales, gen_timescales, save_path):
    """Génère un graphique en barres logarithmiques des temps de relaxation."""
    if gen_timescales is None or len(gen_timescales) == 0:
        return

    n_timescales = min(len(target_timescales), len(gen_timescales))
    targets = target_timescales[:n_timescales]
    gens = gen_timescales[:n_timescales]
    
    x = np.arange(n_timescales)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    
    rects1 = ax.bar(x - width/2, targets, width, label='Vrai (Ground Truth)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, gens, width, label='Généré (Flow Matching)', color='#d62728')

    ax.set_ylabel('Temps Implicite / Relaxation (pas)', fontsize=12)
    ax.set_title('Comparaison de la Cinétique Globale (Implied Timescales)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Processus lent {i+1}' for i in range(n_timescales)])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.3)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def perform_ck_test(gen_model, save_path, mlags=5):
    """
    Exécute le Test de Chapman-Kolmogorov.
    Preuve ultime que la dynamique générée est markovienne (Wu et al. 2018).
    """
    print("Exécution du Test de Chapman-Kolmogorov...")
    try:
        ck_test_results = gen_model.ck_test(mlags=mlags)
        
        fig, axes = dtplots.plot_ck_test(ck_test_results)
        fig.suptitle("Test de Chapman-Kolmogorov (Validation Cinétique)", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Graphique CK-test sauvegardé : {save_path}")
        return True
    except Exception as e:
        print(f"Impossible de réaliser le CK-test : {e}")
        return False
