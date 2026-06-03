import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



class AlanineDipeptideChunkDataset(Dataset):
    def __init__(self, file_path, context_length, training=True, aug_config=None):
        self.data = np.load(file_path)[:]
        self.context_length = context_length
        self.training = training

        # Configuration par défaut si rien n'est passé
        self.cfg = aug_config or {
            "time_warp_prob": 0.3,
            "time_warp_mag": 0.2,
            "jitter_prob": 0.3,
            "jitter_concentration": 50.0,
            "mask_prob": 0.3,
            "mask_ratio": 0.15,
            "reflect_prob": 0.3
        }

    def __len__(self):
        return len(self.data)
    
    def apply_time_warp(self, chunk):
        length = chunk.shape[0]
        factor = 1.0 + torch.randn(1) * self.cfg["time_warp_mag"]
        chunk_t = chunk.t().unsqueeze(0) 
        new_length = max(1, int(length * factor))
        warped = F.interpolate(chunk_t, size=new_length, mode='linear', align_corners=True)
        final = F.interpolate(warped, size=length, mode='linear', align_corners=True)
        return final.squeeze(0).t()

    def apply_periodic_jitter(self, chunk):
        """
        Ajoute un bruit gaussien et projette sur le cercle [-pi, pi].
        Cela simule un jittering périodique sans les problèmes de shape de VonMises.
        """
        # On définit l'intensité du bruit (plus il est haut, plus le bruit est faible)
        std = 1.0 / self.cfg.get("jitter_concentration", 50.0)

        noise = torch.randn_like(chunk) * std

        # Addition et projection circulaire (sin/cos -> atan2)
        # Cela garantit que le résultat reste dans [-pi, pi]
        return torch.atan2(torch.sin(chunk + noise), torch.cos(chunk + noise))

    def apply_temporal_mask(self, chunk):
        # Masque aléatoire basé sur le ratio configuré
        mask = torch.rand(chunk.shape[0]) > self.cfg["mask_ratio"]
        return chunk * mask.unsqueeze(-1)

    def apply_reflection(self, chunk):
        return -chunk

    def __getitem__(self, index):
        chunk = torch.from_numpy(self.data[index].copy()).float()

        if self.training:
            if torch.rand(1) < self.cfg["time_warp_prob"]:
                chunk = self.apply_time_warp(chunk)

            if torch.rand(1) < self.cfg["jitter_prob"]:
                chunk = self.apply_periodic_jitter(chunk)

            if torch.rand(1) < self.cfg["mask_prob"]:
                # On ne masque que l'input (contexte) pour le Flow Matching
                input_part = self.apply_temporal_mask(chunk[:self.context_length])
                chunk = torch.cat([input_part, chunk[self.context_length:]], dim=0)
 
            if torch.rand(1) < self.cfg["reflect_prob"]:
                chunk = self.apply_reflection(chunk)

        item_input = chunk[:self.context_length]
        item_ground_truth = chunk[self.context_length:]

        return {
            "input": item_input.transpose(0, 1).contiguous(),
            "GT": item_ground_truth.transpose(0, 1).contiguous()
        }



if __name__ == "__main__":
    train_file = "../../../output/dataset/train.npy"
    context_size = 5

    try:
        train_dataset = AlanineDipeptideChunkDataset(
            file_path=train_file,
            context_length=context_size
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        for batch in train_loader:
            print("\nAnalyse du premier lot de données :")
            print(f"Dimensions de l'entrée : {batch['input'].shape}")
            print(f"Dimensions de la vérité terrain : {batch['ground_truth'].shape}")
            break

    except FileNotFoundError:
        print(f"Erreur : le fichier {train_file} est introuvable.")
    except ValueError as error_message:
        print(error_message)
