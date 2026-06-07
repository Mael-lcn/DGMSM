import math
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter



class EnergyGuidance:
    def __init__(self, fes_grid, xedges, yedges, sigma=1.0):
        """
        Initialise avec le lissage gaussien et les interpolateurs pour le gradient ET l'énergie.
        """
        # 1. Lissage de la grille
        self.fes_smooth = gaussian_filter(fes_grid, sigma=sigma)

        # 2. Calculer les centres
        self.phi_centers = (xedges[:-1] + xedges[1:]) / 2.0
        self.psi_centers = (yedges[:-1] + yedges[1:]) / 2.0

        dphi = self.phi_centers[1] - self.phi_centers[0]
        dpsi = self.psi_centers[1] - self.psi_centers[0]

        # 3. Calcul du gradient sur la grille lissée
        grad_phi, grad_psi = np.gradient(self.fes_smooth, dphi, dpsi)

        # 4. Interpolateur pour l'ÉNERGIE
        self.interp_fes = RegularGridInterpolator(
            (self.phi_centers, self.psi_centers), self.fes_smooth, 
            bounds_error=False, fill_value=np.max(self.fes_smooth) # Si on sort, on considère énergie max (mur)
        )

        # 5. Interpolateurs pour le GRADIENT
        self.interp_grad_phi = RegularGridInterpolator(
            (self.phi_centers, self.psi_centers), grad_phi, bounds_error=False, fill_value=0.0
        )
        self.interp_grad_psi = RegularGridInterpolator(
            (self.phi_centers, self.psi_centers), grad_psi, bounds_error=False, fill_value=0.0
        )

    def get_energy(self, phi_tensor, psi_tensor):
        """
        Retourne l'énergie (FES) aux coordonnées données.
        """
        phi_np = phi_tensor.detach().cpu().numpy()
        psi_np = psi_tensor.detach().cpu().numpy()

        # Wrapping
        phi_w = (phi_np + math.pi) % (2 * math.pi) - math.pi
        psi_w = (psi_np + math.pi) % (2 * math.pi) - math.pi

        pts = np.stack([phi_w.flatten(), psi_w.flatten()], axis=-1)
        energy = self.interp_fes(pts).reshape(phi_np.shape)

        return torch.tensor(energy, dtype=torch.float32, device=phi_tensor.device)

    def get_guidance_force(self, phi_tensor, psi_tensor, alpha=0.01):
        """
        Retourne la direction (vecteur unitaire) du gradient.
        """
        phi_np = phi_tensor.detach().cpu().numpy()
        psi_np = psi_tensor.detach().cpu().numpy()

        phi_w = (phi_np + math.pi) % (2 * math.pi) - math.pi
        psi_w = (psi_np + math.pi) % (2 * math.pi) - math.pi
        pts = np.stack([phi_w.flatten(), psi_w.flatten()], axis=-1)

        # Calcul du gradient
        f_phi_raw = -self.interp_grad_phi(pts).reshape(phi_np.shape)
        f_psi_raw = -self.interp_grad_psi(pts).reshape(psi_np.shape)

        # Normalisation
        norm = np.sqrt(f_phi_raw**2 + f_psi_raw**2 + 1e-8)

        return torch.tensor(f_phi_raw / norm * alpha, dtype=torch.float32, device=phi_tensor.device), \
               torch.tensor(f_psi_raw / norm * alpha, dtype=torch.float32, device=phi_tensor.device)
