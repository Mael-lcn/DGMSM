import torch.nn as nn

from MambaContextEncoder import MambaContextEncoder
from .FlowNetwork import FlowNetwork1D



class ConditionalFlowMolecule(nn.Module):
    """
    Modèle End-to-End unifiant la mémoire non-Markovienne (Mamba) 
    et la génération de trajectoire continue (Flow Matching).
    """
    def __init__(
        self,
        mamba_in_channels=2,
        flow_in_channels=4,
        context_dim=256,
        mamba_layers=4,
        flow_channels=128,
        flow_blocks=4
    ):
        super().__init__()

        # Lit le passé et génère le contexte C
        self.encoder = MambaContextEncoder(
            in_channels=mamba_in_channels,
            mamba_dim=context_dim,
            num_mamba_layers=mamba_layers,
            out_channels=context_dim
        )

        # Le Moteur : Prend le bruit, le temps, et le contexte C pour prédire la vélocité
        self.vector_field = FlowNetwork1D(
            in_channels=flow_in_channels,
            model_channels=flow_channels,
            num_blocks=flow_blocks,
            mamba_dim=context_dim
        )

    def forward(self, x_t, t, x_past, batch_mask=None):
        """
        x_t: [Batch, 2, 16] (La trajectoire bruitée à l'instant t de l'ODE)
        t: [Batch] (Le temps d'intégration Flow Matching, de 0 à 1)
        x_past: [Batch, 2, Longueur_Passé] (L'historique de la molécule)
        """
        # Extraire la mémoire du passé
        context_c = self.encoder(x_past, batch_mask)

        # Prédire la dérivée temporelle pour le solveur        
        return self.vector_field(x_t, t, context_c)
