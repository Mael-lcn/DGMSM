import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader



class AlanineDipeptideChunkDataset(Dataset):
    """
    Dataset PyTorch pour le chargement de fenêtres temporelles pré-calculées.

    Sépare dynamiquement chaque séquence en une partie contexte pour l'encodeur et une partie vérité terrain
    pour la prédiction.

    Args:
        file_path (str): Chemin vers le fichier array numpy contenant les données.
        context_length (int): Nombre de pas de temps alloués au contexte.
    """

    def __init__(self, file_path, context_length):
        self.data = np.load(file_path)[:]
        self.context_length = context_length

        if len(self.data) > 0:
            self.sequence_length = self.data[0].shape[0]

            if self.context_length >= self.sequence_length:
                raise ValueError(
                    f"Erreur : la taille du contexte ({self.context_length}) "
                    f"doit être strictement inférieure à la taille de la séquence ({self.sequence_length})."
                )
        else:
            self.sequence_length = 0
            print("Avertissement : le dataset est vide.")

        print(f"Dataset chargé depuis {file_path}")
        print(f"Nombre total d'échantillons : {len(self.data)}")

    def __len__(self):
        """
        Retourne le nombre total d'échantillons disponibles.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Récupère un échantillon, le charge en mémoire et effectue la séparation.

        Args:
            index (int): Index de la séquence temporelle à charger.

        Returns:
            dict: Dictionnaire contenant les tenseurs d'entrée et de vérité terrain.
        """
        chunk = np.array(self.data[index], dtype=np.float32)
        item_input = chunk[:self.context_length]
        item_ground_truth = chunk[self.context_length:]

        return {
            "input": torch.from_numpy(item_input).transpose(0, 1).contiguous(),
            "GT": torch.from_numpy(item_ground_truth).transpose(0, 1).contiguous()
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
