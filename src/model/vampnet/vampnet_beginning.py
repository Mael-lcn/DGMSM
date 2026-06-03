import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.data import TrajectoriesDataset
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")


# ------------------------------- data collection -------------------------------
# récupéraction
ala_coords_file = "/Vrac/deepL/output/dataset/train"
data = np.load(ala_coords_file)
dataset = TrajectoriesDataset.from_numpy(1, data)

with np.load(ala_coords_file) as fh:
    dihedral = [fh[f"arr_{i}"] for i in range(3)]

# split val/train
n_val = int(len(dataset) * 0.3)
train_data, val_data = torch.utils.data.random_split(
    dataset, [len(dataset) - n_val, n_val]
)
# Génération DataLoader
loader_train = DataLoader(train_data, batch_size=10000, shuffle=True)
loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)


# ---------------------- training on different classes numbers ----------------------
nb_classe = np.arange(5, 9)

for n_classe in nb_classe:
    print(f"Lancement entrainement pour {n_classe} classes finales...")
    lobe = nn.Sequential(
        nn.BatchNorm1d(data[0].shape[1]),
        nn.Linear(data[0].shape[1], 20),
        nn.ELU(),
        nn.Linear(20, 20),
        nn.ELU(),
        nn.Linear(20, 20),
        nn.ELU(),
        nn.Linear(20, 20),
        nn.ELU(),
        nn.Linear(20, 20),
        nn.ELU(),
        nn.Linear(20, n_classe),
        nn.Softmax(dim=1),
    )
    lobe = lobe.to(device=device)
    vampnet = VAMPNet(lobe=lobe, learning_rate=5e-3, device=device)
    model = vampnet.fit(
        loader_train, n_epochs=100, validation_loader=loader_val, progress=tqdm
    ).fetch_model()

    plt.loglog(*vampnet.train_scores.T, label="training")
    plt.loglog(*vampnet.validation_scores.T, label="validation")
    plt.xlabel("step")
    plt.ylabel("score")
    plt.set_title(f"VAMPNet loss : {n_classe} classes")
    plt.legend()

    state_probabilities = model.transform(data[0])

    assignments = state_probabilities.argmax(1)

    plt.scatter(*dihedral[0].T, c=assignments, s=5, alpha=0.1)
    plt.title(f"Transformed state assignments : {n_classe} classes")
