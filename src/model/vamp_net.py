"""
VAMPNet training on Alanine Dipeptide backbone dihedrals.
Trains models for n_classes in [5, 6, 7, 8] and saves all plots.

Usage:
    python train_vampnet.py
    python train_vampnet.py --n_epochs 50 --output_dir results/
"""

import argparse
import os

import matplotlib.pyplot as plt
import mdshare
import numpy as np
import torch
import torch.nn as nn
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.data import TrajectoriesDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAMPNet on alanine dipeptide")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--val_split", type=float, default=0.3)
    parser.add_argument("--n_classes_min", type=int, default=5)
    parser.add_argument("--n_classes_max", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="vampnet_results")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def build_lobe(n_features: int, n_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm1d(n_features),
        nn.Linear(n_features, 20),
        nn.ELU(),
        nn.Linear(20, 20),
        nn.ELU(),
        nn.Linear(20, 20),
        nn.ELU(),
        nn.Linear(20, 20),
        nn.ELU(),
        nn.Linear(20, 20),
        nn.ELU(),
        nn.Linear(20, n_classes),
        nn.Softmax(dim=1),
    )


def save_loss_plot(vampnet, n_classes: int, output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(*vampnet.train_scores.T, label="training")
    ax.loglog(*vampnet.validation_scores.T, label="validation")
    ax.set_xlabel("step")
    ax.set_ylabel("VAMP-2 score")
    ax.set_title(f"VAMPNet loss — {n_classes} classes")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, f"loss_{n_classes}_classes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Saved {n_classes} classes : {path}")


def save_ramachandran_plot(dihedral, assignments, n_classes: int, output_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        dihedral[:, 0],
        dihedral[:, 1],
        c=assignments,
        cmap="tab10",
        s=5,
        alpha=0.3,
        vmin=0,
        vmax=n_classes - 1,
    )
    # Numéro au centre de chaque cluster
    for state in range(n_classes):
        mask = assignments == state
        if mask.sum() == 0:
            continue
        cx = dihedral[mask, 0].mean()
        cy = dihedral[mask, 1].mean()
        ax.text(cx, cy, str(state + 1), fontsize=13, fontweight="bold", ha="center")
    plt.colorbar(scatter, ax=ax, label="état")
    ax.set_xlabel("φ [rad]")
    ax.set_ylabel("ψ [rad]")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_title(f"Ramachandran — {n_classes} classes")
    fig.tight_layout()
    path = os.path.join(output_dir, f"ramachandran_{n_classes}_classes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → Saved: {path}")


def save_assignments_npy(assignments, n_classes: int, output_dir: str):
    path = os.path.join(output_dir, f"assignments_{n_classes}_classes.npy")
    np.save(path, assignments)
    print(f"  → Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    torch.set_num_threads(12)
    print(f"Using device: {device}")

    # Data
    print("Fetching data...")
    ala_coords_file = mdshare.fetch(
        "alanine-dipeptide-3x250ns-backbone-dihedrals.npz", working_directory="data"
    )
    with np.load(ala_coords_file) as fh:
        dihedral = [fh[f"arr_{i}"] for i in range(3)]

    # TrajectoriesDataset attend des float32
    dihedral_f32 = [d.astype(np.float32) for d in dihedral]
    dataset = TrajectoriesDataset.from_numpy(1, dihedral_f32)

    n_val = int(len(dataset) * args.val_split)
    train_data, val_data = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

    n_features = dihedral_f32[0].shape[1]
    # Concaténation des 3 trajectoires pour la visualisation
    dihedral_all = np.concatenate(dihedral, axis=0)

    # Boucle sur le nombre de classes
    nb_classes = np.arange(args.n_classes_min, args.n_classes_max + 1)

    for n_classes in nb_classes:
        print(f"\n{'=' * 50}")
        print(f"Training VAMPNet — {n_classes} classes")
        print(f"{'=' * 50}")

        lobe = build_lobe(n_features, n_classes).to(device)
        vampnet = VAMPNet(lobe=lobe, learning_rate=args.lr, device=device)

        model = vampnet.fit(
            loader_train,
            n_epochs=args.n_epochs,
            validation_loader=loader_val,
            progress=tqdm,
        ).fetch_model()

        # Plots loss
        save_loss_plot(vampnet, n_classes, args.output_dir)

        # Assignation sur la première trajectoire
        state_probs = model.transform(dihedral_f32[0])  # (T, n_classes)
        assignments = state_probs.argmax(axis=1)  # (T,)

        # Ramachandran
        save_ramachandran_plot(dihedral[0], assignments, n_classes, args.output_dir)

        # Sauvegarde des assignments bruts
        save_assignments_npy(assignments, n_classes, args.output_dir)

    print(f"\nDone. All outputs saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
