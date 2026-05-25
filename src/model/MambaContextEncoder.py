import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba2



def get_1d_sincos_pos_embed(embed_dim, length):
    """Génère l'encodage positionnel sinusoïdal"""
    pos = torch.arange(length, dtype=torch.float32)
    omega = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    pos_emb = torch.cat([emb_sin, emb_cos], dim=1)
    return pos_emb # Shape: [length, embed_dim]


class MambaContextEncoder(nn.Module):
    def __init__(self, in_channels=4, mamba_dim=256, num_mamba_layers=4, dropout=0.1):
        super().__init__()

        # GroupNorm remplace BatchNorm pour la stabilité en microbatching
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, mamba_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, mamba_dim // 2), 
            nn.Conv1d(mamba_dim // 2, mamba_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, mamba_dim)
        )

        max_len = 128
        pos_emb = get_1d_sincos_pos_embed(mamba_dim, max_len).unsqueeze(0) # [1, max_len, D]
        self.register_buffer("pos_embedding", pos_emb)

        self.mamba_layers = nn.ModuleList([
            nn.ModuleDict({
                'mixer': Mamba2(
                    d_model=mamba_dim,
                    d_state=64,
                    headdim=32,
                ),
                'norm': nn.LayerNorm(mamba_dim),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_mamba_layers)
        ])

        self.context_head = nn.Sequential(
            nn.Linear(mamba_dim, mamba_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mamba_dim, mamba_dim)
        )

    def forward(self, x_past):
        h = self.input_proj(x_past)
        h = h.transpose(1, 2).contiguous() # [B, L, D]

        seq_len = h.shape[1]
        h = h + self.pos_embedding[:, :seq_len, :]

        for layer in self.mamba_layers:
            residual = h
            h = layer['norm'](h)
            h = layer['mixer'](h)
            h = layer['dropout'](h)
            h = h + residual

        h_norm = torch.nn.functional.layer_norm(h, [h.shape[-1]])

        # Le réseau dense traite chaque élément de la séquence indépendamment
        full_context = self.context_head(h_norm) 

        return full_context # Shape de retour : [Batch, Length, mamba_dim]
