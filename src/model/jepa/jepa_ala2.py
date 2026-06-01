"""
JEPA pour l'alanine dipeptide (mdshare backbone-dihedrals)
-----------------------------------------------------------
Choix d'architecture :
  - Encodage sin/cos des angles pour gérer la périodicité (S¹ → ℝ²)
  - Un seul encodeur partagé pour contexte et cible
  - Stop-gradient via .detach() sur la branche cible
  - Espace latent continu via reparamétrisation VAE (μ, σ)
  - Prédicteur léger contexte → latent cible
  - Décodeur latent → angles futurs (φ, ψ)
  - Perte angulaire (1 - cos) pour respecter la périodicité en sortie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Preprocessing — sin/cos encoding
# ---------------------------------------------------------------------------


def encode_angles(angles: torch.Tensor) -> torch.Tensor:
    """
    Plonge les angles dièdres dans ℝ⁴ pour éviter la discontinuité à ±π.

    Args:
        angles : (B, T, 2)  — φ et ψ en radians, dans [-π, π]

    Returns:
        (B, T, 4)  — [cos φ, sin φ, cos ψ, sin ψ]
    """
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


def decode_angles(sincos: torch.Tensor) -> torch.Tensor:
    """
    Reconvertit une sortie (cos φ, sin φ, cos ψ, sin ψ) en angles radians.

    Args:
        sincos : (B, H, 4)

    Returns:
        (B, H, 2)  — φ et ψ en radians
    """
    phi = torch.atan2(sincos[..., 1], sincos[..., 0])  # atan2(sin φ, cos φ)
    psi = torch.atan2(sincos[..., 3], sincos[..., 2])  # atan2(sin ψ, cos ψ)
    return torch.stack([phi, psi], dim=-1)


# ---------------------------------------------------------------------------
# 2. Encodeur partagé (contexte ET cible)
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """
    Encode une fenêtre temporelle de conformations → espace latent continu.

    Entrée  : (B, T, 4)  après encode_angles
    Sortie  : μ (B, latent_dim), log σ² (B, latent_dim)

    Le même encodeur est utilisé pour le contexte et la cible.
    Le stop-gradient s'applique via .detach() à l'extérieur, pas ici.
    """

    def __init__(
        self, input_dim: int = 4, T: int = 10, latent_dim: int = 16, hidden: int = 128
    ):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(input_dim * T, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, T, 4)
        Returns:
            mu     : (B, latent_dim)
            logvar : (B, latent_dim)
        """
        h = self.net(x.flatten(1))  # (B, hidden)
        mu = self.mu_head(h)  # (B, latent_dim)
        logvar = self.logvar_head(h)  # (B, latent_dim)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparamétrisation VAE : z = μ + ε·σ  avec ε ~ N(0, I).
    Permet de backpropager à travers l'échantillonnage.
    """
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std  # (B, latent_dim)


# ---------------------------------------------------------------------------
# 3. Prédicteur latent
# ---------------------------------------------------------------------------


class LatentPredictor(nn.Module):
    """
    Prédit la représentation latente de la cible depuis celle du contexte.

    Entrée  : z_ctx  (B, latent_dim)
    Sortie  : ẑ_target  (B, latent_dim)

    Intentionnellement plus léger que l'encodeur (comme dans I-JEPA),
    pour forcer l'encodeur à porter la sémantique.
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (B, latent_dim)


# ---------------------------------------------------------------------------
# 4. Décodeur
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """
    Décode un vecteur latent en séquence de conformations futures.

    Entrée  : z  (B, latent_dim)
    Sortie  : (B, H, 4)  — [cos φ, sin φ, cos ψ, sin ψ] pour H pas futurs

    On prédit directement les coordonnées sin/cos (pas les angles bruts)
    pour rester dans le même espace que l'encodage d'entrée.
    Le décodage en angles se fait via decode_angles() si nécessaire.

    Note : ce décodeur peut être remplacé par ton décodeur flow matching
    existant en conditionnant le champ de vecteurs sur z.
    """

    def __init__(self, latent_dim: int = 16, H: int = 5, hidden: int = 128):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, H * 4),  # H × (cos φ, sin φ, cos ψ, sin ψ)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            (B, H, 4)  — coordonnées sin/cos des conformations futures
        """
        return self.net(z).view(z.size(0), self.H, 4)


# ---------------------------------------------------------------------------
# 5. Pertes
# ---------------------------------------------------------------------------


def loss_jepa(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """
    Perte JEPA : MSE dans l'espace latent.
    z_target doit déjà être détaché (.detach()) avant l'appel.
    """
    return F.mse_loss(z_pred, z_target)


def loss_recon(pred_sincos: torch.Tensor, target_sincos: torch.Tensor) -> torch.Tensor:
    """
    Perte de reconstruction sur les coordonnées sin/cos.
    Utilise MSE directement dans ℝ⁴ (espace continu, pas de discontinuité).
    """
    return F.mse_loss(pred_sincos, target_sincos)


def loss_angular(
    pred_angles: torch.Tensor, target_angles: torch.Tensor
) -> torch.Tensor:
    """
    Alternative : perte angulaire (1 - cos(Δθ)) respectant la périodicité.
    À utiliser si on travaille directement sur les angles en radians.

    Args:
        pred_angles   : (B, H, 2) en radians
        target_angles : (B, H, 2) en radians
    """
    return (1 - torch.cos(pred_angles - target_angles)).mean()


def loss_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Divergence KL entre q(z|x) = N(μ, σ²) et p(z) = N(0, I).
    Régularise l'espace latent pour qu'il reste continu et gaussien.
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()


# ---------------------------------------------------------------------------
# 6. Boucle d'entraînement
# ---------------------------------------------------------------------------


def train_step(
    batch: torch.Tensor,
    encoder: Encoder,
    predictor: LatentPredictor,
    decoder: Decoder,
    optimizer: torch.optim.Optimizer,
    lambda_recon: float = 0.1,
    lambda_kl: float = 0.001,
):
    """
    Un pas d'entraînement JEPA complet.

    Args:
        batch : (B, T+H, 2)  — angles bruts en radians [φ, ψ]
                T frames de contexte + H frames cibles

    Returns:
        dict avec les valeurs de perte pour le logger
    """
    T = encoder.T
    H = decoder.H

    # --- Découpage contexte / cible ---
    ctx_raw = batch[:, :T]  # (B, T, 2)
    target_raw = batch[:, T : T + H]  # (B, H, 2)

    # --- Encodage sin/cos ---
    ctx_in = encode_angles(ctx_raw)  # (B, T, 4)
    target_in = encode_angles(target_raw)  # (B, H, 4)

    # --- Branche contexte (gradient actif) ---
    mu_ctx, logvar_ctx = encoder(ctx_in)
    z_ctx = reparameterize(mu_ctx, logvar_ctx)  # (B, latent_dim)

    # --- Branche cible (stop-gradient via .detach()) ---
    # On appelle le MÊME encodeur, mais on bloque le gradient retour.
    # Sans .detach(), le réseau minimiserait la perte en faisant converger
    with torch.no_grad():
        mu_target, _ = encoder(target_in)
        # Pas de reparamétrisation ici : représentation déterministe pour la cible
        z_target = mu_target  # (B, latent_dim)
    # z_target est déjà détaché (no_grad), pas besoin de .detach() supplémentaire

    # --- Prédiction dans le latent ---
    z_pred = predictor(z_ctx)  # (B, latent_dim)

    # --- Décodage des conformations futures ---
    pred_sincos = decoder(z_pred)  # (B, H, 4)
    target_sincos = encode_angles(target_raw)  # (B, H, 4)

    # --- Calcul des pertes ---
    l_jepa = loss_jepa(z_pred, z_target)
    l_recon = loss_recon(pred_sincos, target_sincos)
    l_kl = loss_kl(mu_ctx, logvar_ctx)

    loss = l_jepa + lambda_recon * l_recon + lambda_kl * l_kl

    # --- Mise à jour ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss_total": loss.item(),
        "loss_jepa": l_jepa.item(),
        "loss_recon": l_recon.item(),
        "loss_kl": l_kl.item(),
    }


# ---------------------------------------------------------------------------
# 7. Dataset — fenêtrage glissant sur les 3 trajectoires mdshare
# ---------------------------------------------------------------------------

import numpy as np
from torch.utils.data import Dataset


class AlanineDihedralDataset(Dataset):
    """
    Dataset sur les angles dièdres backbone de l'alanine dipeptide (mdshare).

    Le fichier .npz contient arr_0, arr_1, arr_2 :
      shape (250000, 2), dtype float32, en radians
      colonne 0 = φ (phi), colonne 1 = ψ (psi)

    Fenêtrage glissant : chaque exemple est une fenêtre de T+H frames
    consécutives. Les T premières sont le contexte, les H suivantes la cible.
    """

    def __init__(self, npz_path: str, T: int = 10, H: int = 5):
        """
        Args:
            npz_path : chemin vers alanine-dipeptide-3x250ns-backbone-dihedrals.npz
            T        : nombre de frames de contexte
            H        : nombre de frames cibles (à prédire)
        """
        self.T = T
        self.H = H
        self.window = T + H

        with np.load(npz_path) as f:
            # Concatène les 3 trajectoires indépendantes
            trajs = [f[f"arr_{i}"].astype(np.float32) for i in range(3)]

        # On construit les fenêtres par trajectoire (pas de fenêtre à cheval)
        windows = []
        for traj in trajs:
            n = len(traj)
            for start in range(0, n - self.window + 1):
                windows.append(traj[start : start + self.window])

        self.data = np.stack(windows, axis=0)  # (N_windows, T+H, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])  # (T+H, 2)


# ---------------------------------------------------------------------------
# 8. Exemple d'utilisation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Hyperparamètres
    T = 10  # frames de contexte
    H = 5  # frames à prédire
    LATENT_DIM = 16
    BATCH_SIZE = 256
    LR = 3e-4
    EPOCHS = 50

    # Modèles
    encoder = Encoder(input_dim=4, T=T, latent_dim=LATENT_DIM, hidden=128)
    predictor = LatentPredictor(latent_dim=LATENT_DIM, hidden=64)
    decoder = Decoder(latent_dim=LATENT_DIM, H=H, hidden=128)

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(predictor.parameters())
        + list(decoder.parameters()),
        lr=LR,
    )

    # Dataset
    # dataset = AlanineDihedralDataset("alanine-dipeptide-3x250ns-backbone-dihedrals.npz", T=T, H=H)
    # loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Exemple avec données synthétiques pour tester le pipeline
    dummy_batch = torch.randn(BATCH_SIZE, T + H, 2)  # angles bruts

    logs = train_step(dummy_batch, encoder, predictor, decoder, optimizer)
    print("Pertes après 1 step :", logs)

    # Pour visualiser le latent après entraînement :
    # encoder.eval()
    # with torch.no_grad():
    #     all_mu = []
    #     for batch in loader:
    #         mu, _ = encoder(encode_angles(batch[:, :T]))
    #         all_mu.append(mu.numpy())
    # latent = np.concatenate(all_mu)  # (N, latent_dim)
    # → t-SNE ou UMAP pour vérifier que C7eq / αR / C7ax sont bien séparés
