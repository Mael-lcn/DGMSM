import math
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter



class EnergyGuidance:
    def __init__(self, fes_grid, xedges, yedges, sigma=1.0):
        """
        Initialise le champ de force avec lissage gaussien de la FES.
        """
        # 1. Lissage de la grille FES
        fes_smooth = gaussian_filter(fes_grid, sigma=sigma)

        # 2. Calculer les centres des bins
        self.phi_centers = (xedges[:-1] + xedges[1:]) / 2.0
        self.psi_centers = (yedges[:-1] + yedges[1:]) / 2.0
        
        dphi = self.phi_centers[1] - self.phi_centers[0]
        dpsi = self.psi_centers[1] - self.psi_centers[0]
        
        # 3. Calcul du gradient sur la grille LISSÉE
        grad_phi, grad_psi = np.gradient(fes_smooth, dphi, dpsi)

        # 4. Interpolateurs
        self.interp_grad_phi = RegularGridInterpolator(
            (self.phi_centers, self.psi_centers), grad_phi, bounds_error=False, fill_value=0.0
        )
        self.interp_grad_psi = RegularGridInterpolator(
            (self.phi_centers, self.psi_centers), grad_psi, bounds_error=False, fill_value=0.0
        )

    def get_guidance_force(self, phi_tensor, psi_tensor, alpha=0.01):
        """
        Retourne la direction vers la vallée (force unitaire).
        """
        phi_np = phi_tensor.detach().cpu().numpy()
        psi_np = psi_tensor.detach().cpu().numpy()

        phi_wrapped = (phi_np + math.pi) % (2 * math.pi) - math.pi
        psi_wrapped = (psi_np + math.pi) % (2 * math.pi) - math.pi
        pts = np.stack([phi_wrapped.flatten(), psi_wrapped.flatten()], axis=-1)

        # Force brute (-nabla F)
        f_phi_raw = -self.interp_grad_phi(pts).reshape(phi_np.shape)
        f_psi_raw = -self.interp_grad_psi(pts).reshape(psi_np.shape)

        # Calcul de la norme pour normaliser
        norm = np.sqrt(f_phi_raw**2 + f_psi_raw**2 + 1e-8)

        # Normalisation (Direction unitaire)
        unit_f_phi = f_phi_raw / norm
        unit_f_psi = f_psi_raw / norm

        # On multiplie par alpha (alpha est maintenant une vitesse constante)
        force_phi_t = torch.tensor(unit_f_phi * alpha, dtype=torch.float32, device=phi_tensor.device)
        force_psi_t = torch.tensor(unit_f_psi * alpha, dtype=torch.float32, device=phi_tensor.device)

        return force_phi_t, force_psi_t
