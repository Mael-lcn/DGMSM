import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity



def plot_free_energy_landscape(train_file_path, bandwidth=0.2):
    """
    Calcule et trace uniquement le paysage d'énergie libre (KDE) 
    d'une trajectoire moléculaire.
    """
    print(f"Chargement des données : {train_file_path}")
    data = np.load(train_file_path)
    angles = data.reshape(-1, 2)

    # 2. Calcul de la KDE continue
    print(f"Calcul de la densité (bandwidth={bandwidth})...")
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(angles)

    # Création de la grille d'évaluation
    phi_grid = np.linspace(-np.pi, np.pi, 100)
    psi_grid = np.linspace(-np.pi, np.pi, 100)
    X, Y = np.meshgrid(phi_grid, psi_grid)
    grid_coords = np.vstack([X.ravel(), Y.ravel()]).T

    log_dens = kde.score_samples(grid_coords)
    free_energy = -log_dens.reshape(X.shape)

    # Normalisation pour que le minimum d'énergie soit à 0
    free_energy = free_energy - np.min(free_energy)

    # 5. Visualisation (Un seul graphique centré)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Niveaux et contours
    levels = np.linspace(0, np.percentile(free_energy, 95), 15)
    im = ax.contourf(X, Y, free_energy, levels=levels, cmap='viridis_r', extend='max')
    contours = ax.contour(X, Y, free_energy, levels=levels, colors='black', linewidths=0.5, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

    # Esthétique du graphique
    ax.set_title("Paysage d'Énergie Libre de l'Alanine Dipeptide", fontsize=14, pad=15)
    ax.set_xlabel("Phi (rad)", fontsize=12)
    ax.set_ylabel("Psi (rad)", fontsize=12)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)

    fig.colorbar(im, ax=ax, label='Énergie Libre relative $\Delta F$ (kT)')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_free_energy_landscape("../../../output/dataset/train.npy", bandwidth=0.2)
