import os
import math
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, wasserstein_distance
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
import deeptime.markov as markov
from deeptime.decomposition import TICA
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



# =====================================================================
# FONCTIONS UTILITAIRES DE BASE
# =====================================================================
def angular_difference(gt, pred):
    """Calcule la différence angulaire en respectant la périodicité [-pi, pi]."""
    return (pred - gt + math.pi) % (2 * math.pi) - math.pi

def circular_mae(gt, pred):
    """MAE Circulaire."""
    return torch.mean(torch.abs(angular_difference(gt, pred)))

def to_sincos_flat(trajs):
    """Convertit les angles en espace topologique (Batch*Time, 4) pour K-Means."""
    if isinstance(trajs, torch.Tensor):
        trajs = trajs.cpu().numpy()
    phi, psi = trajs[:, 0, :], trajs[:, 1, :]
    return np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1).reshape(-1, 4)

def flatten_traj_list(traj_list):
    """Aplatit une liste de trajectoires (Batch, 2, Time) en (N_points, 2)."""
    all_points = []
    for traj in traj_list:
        if isinstance(traj, torch.Tensor): traj = traj.cpu().numpy()
        all_points.append(np.transpose(traj, (0, 2, 1)).reshape(-1, 2))
    return np.concatenate(all_points, axis=0)

def compute_fes(phi, psi, bins=60, temp=300):
    """Calcule la Surface d'Energie Libre (FES) en kcal/mol."""
    kb = 0.001987 # kcal/(mol K)
    hist, xedges, yedges = np.histogram2d(phi, psi, bins=bins, range=[[-math.pi, math.pi], [-math.pi, math.pi]], density=True)
    hist = np.clip(hist, 1e-10, None)
    fes = -kb * temp * np.log(hist)
    fes = fes - np.min(fes)
    return fes, xedges, yedges

def extract_time_series(traj_list):
    """Convertit (Batch, 2, Time) en liste de séries temporelles (Time, 4) pour Deeptime."""
    ts_list = []
    for traj in traj_list:
        if isinstance(traj, torch.Tensor): traj = traj.cpu().numpy()
        B, _, _ = traj.shape
        for i in range(B):
            phi, psi = traj[i, 0, :], traj[i, 1, :]
            ts_list.append(np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1))
    return ts_list


# =====================================================================
# UTILITAIRES DE CINÉTIQUE
# =====================================================================
def build_trans_matrix(dtrajs, n_clusters, lagtime=10):
    """Construit une matrice de transition de taille fixe."""
    counts = np.zeros((n_clusters, n_clusters))
    for traj in dtrajs:
        for t in range(len(traj) - lagtime):
            counts[traj[t], traj[t+lagtime]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums!=0)

def compute_stationary_from_T(T):
    """
    Extrait la distribution stationnaire robuste.
    Gère mathématiquement les états non visités sans crasher.
    """
    # 1. Identifier les états visités
    visited = T.sum(axis=1) > 0
    if not np.any(visited):
        return np.ones(len(T)) / len(T)
        
    # 2. Extraire la sous-matrice valide
    T_sub = T[visited][:, visited]
    
    # 3. Rendre parfaitement stochastique
    row_sums = T_sub.sum(axis=1, keepdims=True)
    T_sub = np.divide(T_sub, row_sums, out=np.zeros_like(T_sub), where=row_sums!=0)
    T_sub[T_sub.sum(axis=1) == 0] = 1.0 / len(T_sub) # Sécurité finale
    
    # 4. Calcul des vecteurs propres
    vals, vecs = np.linalg.eig(T_sub.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi_sub = np.real(vecs[:, idx])
    pi_sub = np.maximum(pi_sub, 0)
    
    if pi_sub.sum() > 0:
        pi_sub /= pi_sub.sum()
    else:
        pi_sub = np.ones(len(pi_sub)) / len(pi_sub)
        
    # 5. Reconstruire le vecteur complet
    pi = np.zeros(len(T))
    pi[visited] = pi_sub
    return pi


# =====================================================================
# FONCTIONS GRAPHIQUES GLOBALES
# =====================================================================
def plot_trajectory(gt, pred, mae_score, save_path):
    gt_cpu = gt.cpu().numpy()
    pred_cpu = pred.cpu().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    time_axis = range(gt_cpu.shape[1])

    axs[0].plot(time_axis, gt_cpu[0], label="Vrai (GT)", color="#1f77b4", marker='o')
    axs[0].plot(time_axis, pred_cpu[0], label="Prédiction Flow", color="#d62728", marker='x', linestyle='dashed')
    axs[0].set_title("Angle Dièdre Phi (φ)")
    axs[0].set_xlabel("Pas de temps")
    axs[0].set_ylim(-math.pi, math.pi)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

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

def plot_ramachandran(gt_phi, gt_psi, pred_phi, pred_psi, jsd_score, w1_score, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    bins = 60
    hist_range = [[-math.pi, math.pi], [-math.pi, math.pi]]
    
    axs[0].hist2d(gt_phi, gt_psi, bins=bins, range=hist_range, cmap='Blues', density=True)
    axs[0].set_title("Vrai (Ground Truth)")
    axs[0].set_xlabel("Phi (rad)")
    axs[0].set_ylabel("Psi (rad)")

    axs[1].hist2d(pred_phi, pred_psi, bins=bins, range=hist_range, cmap='Reds', density=True)
    axs[1].set_title("Généré (Flow Matching)")
    axs[1].set_xlabel("Phi (rad)")

    fig.suptitle(f"Ramachandran Plot | JSD: {jsd_score:.4f} | W1: {w1_score:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_kmeans_centroids(fes_gt, xedges, yedges, kmeans_model, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    levels = np.linspace(0, 15, 25)
    c = ax.contourf(X, Y, fes_gt.T, levels=levels, cmap='RdYlBu_r', extend='max', alpha=0.85)
    fig.colorbar(c, ax=ax, label='Énergie Libre (kcal/mol)')

    centers_4d = kmeans_model.cluster_centers_
    centers_phi = np.arctan2(centers_4d[:, 0], centers_4d[:, 1])
    centers_psi = np.arctan2(centers_4d[:, 2], centers_4d[:, 3])

    ax.scatter(centers_phi, centers_psi, c='white', marker='X', s=200, edgecolor='black', linewidth=2, zorder=5, label='Centres K-Means++')
    for i, (phi, psi) in enumerate(zip(centers_phi, centers_psi)):
        ax.annotate(f"État {i}", (phi + 0.15, psi + 0.15), color='black', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.8), zorder=6)

    ax.set_title("Vérification : États K-Means++ sur la FES", fontsize=14)
    ax.set_xlabel("Phi (rad)")
    ax.set_ylabel("Psi (rad)")
    ax.set_xlim(-math.pi, math.pi)
    ax.set_ylim(-math.pi, math.pi)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_stationary_distributions(pi_gt, pi_pred, phase_name, save_path):
    n_states = len(pi_gt)
    x = np.arange(n_states)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(x - width/2, pi_gt, width, label='Vrai (Ground Truth)', color='#1f77b4')
    ax.bar(x + width/2, pi_pred, width, label='Généré (Flow Matching)', color='#d62728')
    ax.set_ylabel('Probabilité Stationnaire')
    ax.set_xlabel('États Métastables')
    ax.set_title(f"Thermodynamique : Distribution Stationnaire ({phase_name})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"État {i}" for i in range(n_states)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_transition_matrix_diff(T_gt, T_pred, phase_name, save_path):
    diff = T_pred - T_gt
    max_val = max(0.1, np.max(np.abs(diff)))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    cax = ax.imshow(diff, cmap='RdBu_r', vmin=-max_val, vmax=max_val) 
    fig.colorbar(cax, label='Différence (Pred - GT)')
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            val = diff[i, j]
            if abs(val) > 0.01:
                color = "white" if abs(val) > (max_val / 2) else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color=color, fontsize=9)
    ax.set_title(f"Erreur de Cinétique : Matrice de Transition ({phase_name})")
    ax.set_xlabel("État d'arrivée (j)")
    ax.set_ylabel("État de départ (i)")
    ax.set_xticks(np.arange(diff.shape[1]))
    ax.set_yticks(np.arange(diff.shape[0]))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_mfpt_comparison(msm_gt, msm_pred, n_clusters, save_path):
    """Affiche la Heatmap des Temps de Premier Passage Moyens (MFPT) en échelle log."""
    mfpt_gt = np.zeros((n_clusters, n_clusters))
    mfpt_pred = np.zeros((n_clusters, n_clusters))
    
    for j in range(n_clusters):
        try:
            t_gt = msm_gt.mfpt(j)
            for i in range(n_clusters): mfpt_gt[i, j] = t_gt[i] if i != j else 0
        except: pass
        try:
            t_pred = msm_pred.mfpt(j)
            for i in range(n_clusters): mfpt_pred[i, j] = t_pred[i] if i != j else 0
        except: pass

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    im1 = axs[0].imshow(np.log10(mfpt_gt + 1), cmap='viridis')
    axs[0].set_title("MFPT Ground Truth (log10 pas)")
    axs[0].set_xlabel("Cible (j)"); axs[0].set_ylabel("Origine (i)")
    fig.colorbar(im1, ax=axs[0])
    
    im2 = axs[1].imshow(np.log10(mfpt_pred + 1), cmap='viridis')
    axs[1].set_title("MFPT Généré (log10 pas)")
    axs[1].set_xlabel("Cible (j)")
    fig.colorbar(im2, ax=axs[1])

    plt.suptitle("Comparaison des Temps de Premier Passage Moyens (MFPT)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =====================================================================
# ENTRAÎNEMENT DES BASES DE RÉFÉRENCE (TICA / KMEANS)
# =====================================================================
def fit_global_tica(gt_trajs_list, lagtime=10):
    gt_ts = extract_time_series(gt_trajs_list)
    tica_model = TICA(lagtime=lagtime, dim=2).fit(gt_ts).fetch_model()
    return tica_model

def fit_global_kmeans(gt_trajs_list, n_clusters=6, save_dir=None): 
    gt_flat = flatten_traj_list(gt_trajs_list)
    fes_gt, xedges, yedges = compute_fes(gt_flat[:, 0], gt_flat[:, 1])
    
    phi_gt, psi_gt = gt_flat[:, 0], gt_flat[:, 1]
    all_sincos = np.stack([np.sin(phi_gt), np.cos(phi_gt), np.sin(psi_gt), np.cos(psi_gt)], axis=-1)
    
    print(f"[K-Means] Recherche des {n_clusters} états via 50 initialisations sur la densité des données...")
    global_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=50, random_state=42).fit(all_sincos)
    print(f"[K-Means] Modèle ajusté avec succès (Meilleure Inertie : {global_kmeans.inertia_:.2f})")
    
    if save_dir is not None:
        plot_path = os.path.join(save_dir, "phase0_kmeans_centroids.png")
        plot_kmeans_centroids(fes_gt, xedges, yedges, global_kmeans, plot_path)
        
    return global_kmeans, fes_gt, xedges, yedges


# =====================================================================
# PHASE 1 : DYNAMIQUE LOCALE (CHUNKS COURTS)
# =====================================================================
class Phase1Evaluator:
    def __init__(self, max_mae_steps=3):
        self.max_mae_steps = max_mae_steps
        self.all_gt_chunks = []
        self.all_pred_chunks = []

    def update(self, gt, pred):
        self.all_gt_chunks.append(gt.cpu())
        self.all_pred_chunks.append(pred.cpu())
        
    def reset(self):
        self.all_gt_chunks = []
        self.all_pred_chunks = []

    def compute_and_log(self, step=None, save_dir=None, global_kmeans=None, tica_model=None, T_gt=None, pi_gt=None):
        full_gt_tensor = torch.cat(self.all_gt_chunks, dim=0)
        full_pred_tensor = torch.cat(self.all_pred_chunks, dim=0)
        gt_np = full_gt_tensor.numpy()
        pred_np = full_pred_tensor.numpy()
        N_chunks, _, T_chunk = gt_np.shape

        if T_gt is None or pi_gt is None:
            n_clusters = global_kmeans.n_clusters
            T_gt = np.eye(n_clusters)
            pi_gt = np.ones(n_clusters) / n_clusters

        # 1. MAE Circulaire
        gt_short = full_gt_tensor[:, :, :min(self.max_mae_steps, T_chunk)]
        pred_short = full_pred_tensor[:, :, :min(self.max_mae_steps, T_chunk)]
        mae_deg = circular_mae(gt_short, pred_short).item() * (180.0 / math.pi)

        # 2. Thermodynamique Locale : JSD et Wasserstein (W1)
        gt_phi, gt_psi = gt_np[:, 0, :].flatten(), gt_np[:, 1, :].flatten()
        pred_phi, pred_psi = pred_np[:, 0, :].flatten(), pred_np[:, 1, :].flatten()
        
        hist_range = [[-math.pi, math.pi], [-math.pi, math.pi]]
        h_gt, _, _ = np.histogram2d(gt_phi, gt_psi, bins=60, range=hist_range, density=True)
        h_pred, _, _ = np.histogram2d(pred_phi, pred_psi, bins=60, range=hist_range, density=True)
        p = h_gt.flatten() / (h_gt.sum() + 1e-10)
        q = h_pred.flatten() / (h_pred.sum() + 1e-10)
        jsd_p1 = float(jensenshannon(p, q))

        w1_phi = wasserstein_distance(gt_phi, pred_phi)
        w1_psi = wasserstein_distance(gt_psi, pred_psi)
        w1_p1 = float((w1_phi + w1_psi) / 2.0)

        if save_dir is not None and step is not None:
            plot_trajectory(full_gt_tensor[0], full_pred_tensor[0], mae_deg, os.path.join(save_dir, f"traj_step_{step}.png"))
            plot_ramachandran(gt_phi, gt_psi, pred_phi, pred_psi, jsd_p1, w1_p1, os.path.join(save_dir, f"rama_step_{step}.png"))

        # 3. Cinétique : Matrice et Distribution
        pred_sincos = to_sincos_flat(pred_np)
        n_clusters = global_kmeans.n_clusters
        pred_dtrajs = global_kmeans.predict(pred_sincos).reshape(N_chunks, T_chunk)
        
        T_pred = build_trans_matrix(pred_dtrajs, n_clusters, lagtime=10)
        pi_pred = compute_stationary_from_T(T_pred)
        
        trans_mae = float(np.mean(np.abs(T_gt - T_pred)))
        statio_mae = float(np.mean(np.abs(pi_gt - pi_pred)))

        if save_dir is not None and step is not None:
            plot_transition_matrix_diff(T_gt, T_pred, "Phase 1", os.path.join(save_dir, f"phase1_trans_diff_step_{step}.png"))
            plot_stationary_distributions(pi_gt, pi_pred, "Phase 1", os.path.join(save_dir, f"phase1_stationary_step_{step}.png"))

        metrics = {
            "P1_MAE_Circ_Deg_3steps": mae_deg,
            "P1_JSD_Distribution": jsd_p1,
            "P1_Wasserstein_Rad": w1_p1,
            "P1_Transition_Matrix_MAE": trans_mae,
            "P1_Stationary_Dist_MAE": statio_mae
        }

        # 4. TICA Locale
        if tica_model is not None:
            gt_sincos = to_sincos_flat(gt_np)
            gt_tica = tica_model.transform(gt_sincos)
            pred_tica = tica_model.transform(pred_sincos)

            h_gt_tica, x_edges, y_edges = np.histogram2d(gt_tica[:, 0], gt_tica[:, 1], bins=50, density=True)
            h_pred_tica, _, _ = np.histogram2d(pred_tica[:, 0], pred_tica[:, 1], bins=[x_edges, y_edges], density=True)

            jsd_tica = float(jensenshannon(h_gt_tica.flatten() / (h_gt_tica.sum() + 1e-10), h_pred_tica.flatten() / (h_pred_tica.sum() + 1e-10)))
            pearson_r, _ = pearsonr(h_gt_tica.flatten(), h_pred_tica.flatten())

            metrics["P1_TICA_JSD"] = jsd_tica
            metrics["P1_TICA_Pearson"] = float(pearson_r)

        return metrics


# =====================================================================
# PHASE 2 : SAMPLING LONG-TERME & THERMODYNAMIQUE
# =====================================================================
def plot_fes_comparison(fes_gt, fes_pred, xedges, yedges, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    levels = np.linspace(0, 10, 20)
    c1 = axs[0].contourf(X, Y, fes_gt.T, levels=levels, cmap='jet', extend='max')
    axs[0].set_title("FES Ground Truth (MD)")
    axs[0].set_xlabel("Phi"); axs[0].set_ylabel("Psi")
    c2 = axs[1].contourf(X, Y, fes_pred.T, levels=levels, cmap='jet', extend='max')
    axs[1].set_title("FES Générée (Flow Matching)")
    axs[1].set_xlabel("Phi")
    fig.colorbar(c1, ax=axs, orientation='vertical', label='Free Energy (kcal/mol)')
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_autocorrelation(gt_trajs, pred_trajs, save_path):
    def calc_acf(trajs, max_lag=500):
        acf_all = []
        for traj in trajs:
            T = traj.shape[2]
            lag_limit = min(max_lag, T-1)
            phi = traj[:, 0, :]
            acf = [np.mean(np.cos(phi[:, :-l] - phi[:, l:])) if l > 0 else 1.0 for l in range(lag_limit)]
            acf_all.append(acf)
        min_len = min([len(a) for a in acf_all])
        return np.mean([a[:min_len] for a in acf_all], axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(calc_acf(gt_trajs), label='Ground Truth', color='blue', linewidth=2)
    plt.plot(calc_acf(pred_trajs), label='Generated', color='red', linestyle='--', linewidth=2)
    plt.axhline(0, color='black', linestyle=':')
    plt.xlabel('Lag Time (steps)')
    plt.ylabel('Autocorrelation (Phi)')
    plt.title('Décroissance de l\'Autocorrélation')
    plt.legend()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_implied_timescales_spectrum(gt_dtrajs, pred_dtrajs, save_path, max_lag=20):
    """Trace le spectre complet des 3 temps d'implication les plus lents (t2, t3, t4)."""
    lags = np.arange(1, max_lag + 1, max(1, max_lag//10))
    t_gt = {0: [], 1: [], 2: []} 
    t_pred = {0: [], 1: [], 2: []}
    
    for lag in lags:
        try:
            m_gt = markov.TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit(gt_dtrajs).fetch_model()
            ts_gt = markov.msm.MaximumLikelihoodMSM(reversible=True).fit(m_gt).fetch_model().timescales()
            for k in range(3): t_gt[k].append(ts_gt[k] if len(ts_gt) > k else 0)
            
            m_pred = markov.TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit(pred_dtrajs).fetch_model()
            ts_pred = markov.msm.MaximumLikelihoodMSM(reversible=True).fit(m_pred).fetch_model().timescales()
            for k in range(3): t_pred[k].append(ts_pred[k] if len(ts_pred) > k else 0)
        except:
            for k in range(3): t_gt[k].append(0); t_pred[k].append(0)

    plt.figure(figsize=(8, 5), dpi=150)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    for k in range(3):
        plt.plot(lags, t_gt[k], marker='o', color=colors[k], label=f'GT $t_{k+2}$')
        plt.plot(lags, t_pred[k], marker='x', linestyle='--', color=colors[k], label=f'Pred $t_{k+2}$')
    
    plt.xlabel('Lag time $\\tau$ (steps)')
    plt.ylabel('Implied Timescales (steps)')
    plt.yscale('log') # Echelle LOG indispensable
    plt.title('Spectre des Implied Timescales ($t_2, t_3, t_4$)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_trajectory_on_fes(fes, xedges, yedges, sample_trajs, cluster_centers, save_path):
    n_trajs = len(sample_trajs)
    fig, axs = plt.subplots(1, n_trajs, figsize=(4.5 * n_trajs, 5), dpi=200, sharex=True, sharey=True)
    if n_trajs == 1: axs = [axs] 
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    levels = np.linspace(0, 15, 25) 

    for idx, (ax, sample_traj) in enumerate(zip(axs, sample_trajs)):
        c = ax.contourf(X, Y, fes.T, levels=levels, cmap='RdYlBu_r', extend='max', alpha=0.5)
        if isinstance(sample_traj, torch.Tensor): sample_traj = sample_traj.cpu().numpy()
        
        phi_t, psi_t = sample_traj[0, :], sample_traj[1, :]
        time_steps = np.arange(len(phi_t))

        ax.plot(phi_t, psi_t, color='black', linewidth=0.3, alpha=0.3, zorder=2)
        sc = ax.scatter(phi_t, psi_t, c=time_steps, cmap='plasma', s=10, alpha=0.8, zorder=3, edgecolors='none')

        ax.scatter(phi_t[0], psi_t[0], color='lime', marker='o', s=80, zorder=5, edgecolor='black', linewidth=1.2, label='Début (t=0)')
        ax.scatter(phi_t[-1], psi_t[-1], color='red', marker='s', s=80, zorder=5, edgecolor='black', linewidth=1.2, label='Fin (t=max)')

        if cluster_centers is not None:
            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='white', marker='X', s=90, edgecolor='black', linewidth=1.2, zorder=4)

        ax.set_title(f"Traj. {idx+1}", fontsize=11)
        ax.set_xlabel("Phi (rad)")
        if idx == 0: ax.set_ylabel("Psi (rad)")
        ax.set_xlim(-math.pi, math.pi); ax.set_ylim(-math.pi, math.pi)

    cbar = fig.colorbar(c, ax=axs, orientation='vertical', fraction=0.015, pad=0.02)
    cbar.set_label('Énergie Libre (kcal/mol)')
    plt.suptitle("Dynamique Long-Terme tracée sur la FES Générée", fontsize=15, y=1.05)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def run_phase_2_eval(gt_trajs_list, pred_trajs_list, save_dir, fes_gt, xedges, yedges, global_kmeans, tica_model=None, T_gt=None, pi_gt=None):
    print("\n[Phase 2] Évaluation Thermodynamique, Cinétique Avancée (SOTA)...")
    metrics = {}

    if T_gt is None or pi_gt is None:
        T_gt = np.eye(global_kmeans.n_clusters)
        pi_gt = np.ones(global_kmeans.n_clusters) / global_kmeans.n_clusters

    gt_flat = flatten_traj_list(gt_trajs_list)
    pred_flat = flatten_traj_list(pred_trajs_list)
    
    # 1. FES et Thermodynamique Macroscopique
    fes_pred, xedges_p, yedges_p = compute_fes(pred_flat[:, 0], pred_flat[:, 1])
    plot_fes_comparison(fes_gt, fes_pred, xedges, yedges, os.path.join(save_dir, "phase2_FES_comparison.png"))
    
    accessible_mask = fes_gt < 10.0 
    metrics["P2_FES_RMSE_kcal_mol"] = float(np.sqrt(np.mean((fes_gt[accessible_mask] - fes_pred[accessible_mask])**2)))

    # Distance Wasserstein Macroscopique
    w1_phi = wasserstein_distance(gt_flat[:, 0], pred_flat[:, 0])
    w1_psi = wasserstein_distance(gt_flat[:, 1], pred_flat[:, 1])
    metrics["P2_Wasserstein_Rad"] = float((w1_phi + w1_psi) / 2.0)

    plot_autocorrelation(gt_trajs_list, pred_trajs_list, os.path.join(save_dir, "phase2_autocorrelation.png"))

    # 2. TICA
    if tica_model is not None:
        gt_ts = extract_time_series(gt_trajs_list)
        pred_ts = extract_time_series(pred_trajs_list)
        gt_tica_flat = np.concatenate([tica_model.transform(ts) for ts in gt_ts], axis=0)
        pred_tica_flat = np.concatenate([tica_model.transform(ts) for ts in pred_ts], axis=0)
        
        h_gt_tica, x_e_t, y_e_t = np.histogram2d(gt_tica_flat[:, 0], gt_tica_flat[:, 1], bins=50, density=True)
        h_pred_tica, _, _ = np.histogram2d(pred_tica_flat[:, 0], pred_tica_flat[:, 1], bins=[x_e_t, y_e_t], density=True)

        metrics["P2_TICA_JSD"] = float(jensenshannon(h_gt_tica.flatten() / (h_gt_tica.sum() + 1e-10), h_pred_tica.flatten() / (h_pred_tica.sum() + 1e-10)))
        metrics["P2_TICA_Pearson"] = float(pearsonr(h_gt_tica.flatten(), h_pred_tica.flatten())[0])

    # 3. Cinétique : Markov, MFPT, CK-Test
    print("[Phase 2] Modèles MSM, MFPT et Chapman-Kolmogorov...")
    def discretize(t_list):
        dtrajs = []
        for traj in t_list:
            if isinstance(traj, torch.Tensor): traj = traj.cpu().numpy()
            sc = to_sincos_flat(traj)
            dtrajs.extend([global_kmeans.predict(sc).reshape(traj.shape[0], traj.shape[2])[i] for i in range(traj.shape[0])])
        return dtrajs

    gt_dtrajs = discretize(gt_trajs_list)
    pred_dtrajs = discretize(pred_trajs_list)
    lag_time = 10
    n_clusters = global_kmeans.n_clusters

    # Matrices et Stationnaires
    T_pred = build_trans_matrix(pred_dtrajs, n_clusters, lagtime=lag_time)
    pi_pred = compute_stationary_from_T(T_pred)
    metrics["P2_Transition_Matrix_MAE"] = float(np.mean(np.abs(T_gt - T_pred)))
    metrics["P2_Stationary_Dist_MAE"] = float(np.mean(np.abs(pi_gt - pi_pred)))
    plot_transition_matrix_diff(T_gt, T_pred, "Phase 2", os.path.join(save_dir, "phase2_trans_diff.png"))
    plot_stationary_distributions(pi_gt, pi_pred, "Phase 2", os.path.join(save_dir, "phase2_stationary_dist.png"))

    # Deeptime MSM pour MFPT, Timescales, Stationnaire et CK-Test SOTA
    try:
        # 1. Estimateurs de comptage
        count_gt = markov.TransitionCountEstimator(lagtime=lag_time, count_mode="sliding").fit(gt_dtrajs).fetch_model()
        count_pred = markov.TransitionCountEstimator(lagtime=lag_time, count_mode="sliding").fit(pred_dtrajs).fetch_model()

        # 2. Extraction des modèles
        msm_gt = markov.msm.MaximumLikelihoodMSM(reversible=True).fit(count_gt).fetch_model()
        msm_pred = markov.msm.MaximumLikelihoodMSM(reversible=True).fit(count_pred).fetch_model()

        # On crée des vecteurs de zéros de la taille totale (n_clusters)
        pi_gt_full = np.zeros(n_clusters)
        pi_gt_full[msm_gt.active_set] = msm_gt.stationary_distribution

        pi_pred_full = np.zeros(n_clusters)
        pi_pred_full[msm_pred.active_set] = msm_pred.stationary_distribution

        # On écrase l'ancienne valeur avec la vraie métrique
        metrics["P2_Stationary_Dist_MAE"] = float(np.mean(np.abs(pi_gt_full - pi_pred_full)))
        plot_stationary_distributions(pi_gt_full, pi_pred_full, "Phase 2 (MSM)", os.path.join(save_dir, "phase2_stationary_dist_msm.png"))

        # 3. Spectres cinétiques
        ts_gt = msm_gt.timescales()
        ts_pred = msm_pred.timescales()
        for k in range(3):
            metrics[f"P2_t{k+2}_GT_Steps"] = float(ts_gt[k]) if len(ts_gt) > k else 0.0
            metrics[f"P2_t{k+2}_Pred_Steps"] = float(ts_pred[k]) if len(ts_pred) > k else 0.0
            
        plot_mfpt_comparison(msm_gt, msm_pred, n_clusters, os.path.join(save_dir, "phase2_MFPT_comparison.png"))

        try:
            import deeptime.plots as dplt
            
            models_ck = []
            lags_ck = [lag_time * i for i in range(1, 6)] # Test sur lagtime x1, x2, x3, x4, x5
            for lag in lags_ck:
                c_model = markov.TransitionCountEstimator(lagtime=lag, count_mode="sliding").fit(pred_dtrajs).fetch_model()
                m_model = markov.msm.MaximumLikelihoodMSM(reversible=True).fit(c_model).fetch_model()
                models_ck.append(m_model)

            # Appel du test sur le modèle principal en lui donnant les autres modèles
            ck_pred = msm_pred.ck_test(models_ck, n_metastable_sets=n_clusters)
            
            fig = dplt.plot_ck_test(ck_pred) 
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "phase2_CK_test_pred.png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[Info] Test CK ignoré (non-convergé ou erreur de plot) : {e}")

    except Exception as e:
        print(f"[Avertissement] Le modèle MSM SOTA n'a pas convergé : {e}")

    plot_implied_timescales_spectrum(gt_dtrajs, pred_dtrajs, os.path.join(save_dir, "phase2_implied_timescales.png"))

    # 4. Tracé des Trajectoires
    all_individual_trajs = [batch_traj[b] for batch_traj in pred_trajs_list for b in range(batch_traj.shape[0])]
    np.random.seed(42) 
    sampled_trajectories = [all_individual_trajs[idx] for idx in np.random.choice(len(all_individual_trajs), min(5, len(all_individual_trajs)), replace=False)]

    centers_4d = global_kmeans.cluster_centers_
    centers_2d = np.column_stack((np.arctan2(centers_4d[:, 0], centers_4d[:, 1]), np.arctan2(centers_4d[:, 2], centers_4d[:, 3])))

    plot_trajectory_on_fes(fes_pred, xedges_p, yedges_p, sampled_trajectories, centers_2d, os.path.join(save_dir, "phase2_trajectory_dynamics.png"))

    return metrics
