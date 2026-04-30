import torch
import torch.nn as nn
from mamba_ssm import Mamba2



class MambaContextEncoder(nn.Module):
    """
    Encodeur Mamba-2 adapté pour l'historique moléculaire.
    Extrait un vecteur de contexte continu à partir des trajectoires passées.
    """
    def __init__(
        self, 
        in_channels=2,
        mamba_dim=256,
        num_mamba_layers=4,
        out_channels=256
    ):
        super().__init__()

        # Projection initiale
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, mamba_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim // 2),
  
            nn.Conv1d(mamba_dim // 2, mamba_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(mamba_dim)
        )

        # Empilement Mamba-2 (Lecture causale de l'historique)
        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'mixer': Mamba2(
                    d_model=mamba_dim,
                    d_state=64,
                    d_conv=4,
                    expand=2,
                    headdim=64
                ),
                'norm': nn.LayerNorm(mamba_dim)
            }) for _ in range(num_mamba_layers)
        ])
        
        # Tête de projection du Contexte
        self.context_head = nn.Sequential(
            nn.Linear(mamba_dim, mamba_dim),
            nn.SiLU(),
            nn.Linear(mamba_dim, out_channels)
        )

    def forward(self, x_past, batch_mask=None):
        """
        x_past: [Batch, 2, Longueur_Passé]
        """
        # Projection des angles
        x_features = self.input_proj(x_past)

        # Transposition pour Mamba : [Batch, Longueur, Dimension]
        hidden_states = x_features.transpose(1, 2)

        # 3. Traitement récurrent
        for layer in self.mamba_layers:
            residual = hidden_states
            mixed = layer['mixer'](layer['norm'](hidden_states))
            hidden_states = mixed + residual

        # Global Average Pooling
        if batch_mask is not None:
            # Gestion des historiques de tailles variables
            mask_downsampled = torch.nn.functional.interpolate(
                batch_mask.unsqueeze(1).float(), 
                size=hidden_states.shape[1], mode='nearest'
            ).squeeze(1).unsqueeze(-1)

            sum_hidden = (hidden_states * mask_downsampled).sum(dim=1)
            valid_lengths = mask_downsampled.sum(dim=1).clamp(min=1e-9)
            final_rep = sum_hidden / valid_lengths
        else:
            final_rep = hidden_states.mean(dim=1)

        # Création du vecteur de contexte C
        return self.context_head(final_rep)
