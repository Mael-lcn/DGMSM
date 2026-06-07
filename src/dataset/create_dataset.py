import os
import glob
import math
import argparse
import numpy as np
from tqdm import tqdm


def autocorr_fft(x):
    """
    Calcule la fonction d'autocorrélation normalisée C(tau) via FFT.
    x : numpy array 1D (la trajectoire d'une seule caractéristique)
    """
    # 1. Centrer la donnée (x - mu)
    x_centered = x - np.mean(x)
    n = len(x)

    # 2. Zero-padding
    f = np.fft.fft(x_centered, n=2*n)

    # 3. Théorème de Wiener-Khinchin : l'autocorrélation est la transformée 
    # de Fourier inverse du spectre de puissance
    power_spectrum = f * np.conjugate(f)
    acf = np.fft.ifft(power_spectrum)[:n].real

    # 4. Normalisation pour que C(0) = 1
    return acf / acf[0]

def compute_tau_int(acf):
    """
    Intègre C(tau) pour obtenir le temps de décorrélation.
    """
    tau_int = 0.5  # Convention en physique

    # On somme C(tau)
    for c in acf[1:]:
        if c <= 0:
            break
        tau_int += c

    return tau_int


def create_chunks(seq, L, S):
    """
    Divise une séquence en fenêtres glissantes de taille fixe.

    Args:
        seq (numpy.ndarray): Séquence d'entrée de dimension (N, D).
        L (int): Longueur de chaque fenêtre.
        S (int): stride entre deux fenêtres.

    Returns:
        numpy.ndarray: Tableau des fenêtres de dimension (M, L, D).
    """
    n_steps = len(seq)
    if n_steps < L:
        return np.array([])

    return np.array([seq[i : i + L] for i in range(0, n_steps - L + 1, S)])


def run(args):
    """
    Exécute le pipeline de transformation SOTA (Macroscopique + Microscopique).
    """
    if args.train_size + args.val_size >= 1.0:
        raise ValueError("La somme de train_size et val_size doit être inférieure à 1.0")

    files = glob.glob(args.data)
    if not files:
        print(f"Aucune donnée trouvée au chemin : {args.data}")
        return

    all_train, all_val, all_test = [], [], []

    for file_path in tqdm(files, desc="Traitement"):
        with np.load(file_path) as npz_file:
            for key in npz_file.files:
                data = npz_file[key]

                # 1. Calcul du temps de décorrélation physique (Macro)
                max_tau_int = 0
                for dim in range(data.shape[1]):
                    obs_sin = np.sin(data[:, dim])
                    obs_cos = np.cos(data[:, dim])

                    # On calcule les deux autocorrélations
                    acf_sin = autocorr_fft(obs_sin)
                    acf_cos = autocorr_fft(obs_cos)

                    # L'autocorrélation circulaire totale
                    acf_totale = (acf_sin + acf_cos) / 2.0
                    tau_int_dim = compute_tau_int(acf_totale)

                    if tau_int_dim > max_tau_int:
                        max_tau_int = tau_int_dim

                print(f"[{key}] Temps de décorrélation max : {max_tau_int:.2f} pas")

                optimal_gap = int(2 * max_tau_int)

                dense_stride = 8

                print(f"-> Utilise un gap de sécurité (G) de {optimal_gap} pas")
                print(f"-> Extraction DENSE pour Mamba avec un stride (S) de {dense_stride}")

                # 3. Séparation MACROSCOPIQUE de la trajectoire AVANT le chunking
                N = len(data)
                train_end = int(N * args.train_size)
                val_end = train_end + optimal_gap + int(N * args.val_size)

                # Extraction des gros blocs sécurisés
                train_raw = data[:train_end]
                val_raw = data[train_end + optimal_gap : val_end]
                test_raw = data[val_end + optimal_gap :]

                # 4. Découpage très fin à l'intérieur des blocs
                if len(train_raw) > args.L: 
                    all_train.append(create_chunks(train_raw, args.L, dense_stride))
                if len(val_raw) > args.L: 
                    all_val.append(create_chunks(val_raw, args.L, dense_stride))
                if len(test_raw) > args.L: 
                    all_test.append(create_chunks(test_raw, args.L, dense_stride))

    if not all_train:
        print("Échec de la génération : aucun échantillon valide produit.")
        return

    # Concaténation de tous les chunks
    x_train = np.vstack(all_train)
    x_val = np.vstack(all_val) if all_val else np.array([])
    x_test = np.vstack(all_test) if all_test else np.array([])

    # Mélange pour briser l'ordre séquentiel dans les batchs d'entraînement
    np.random.shuffle(x_train)
    np.random.shuffle(x_val)
    np.random.shuffle(x_test)

    print("\n=========================================")
    print(f"Taille de x_train : {len(x_train)} trajectoires")
    print(f"Taille de x_val   : {len(x_val)} trajectoires")
    print(f"Taille de x_test  : {len(x_test)} trajectoires")
    print("=========================================\n")

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "train.npy"), x_train)
    np.save(os.path.join(output_dir, "val.npy"), x_val)
    np.save(os.path.join(output_dir, "test.npy"), x_test)

    print(f"Exportation terminée dans {output_dir} : train.npy, val.npy, test.npy")


def main():
    """
    Définit les arguments de la ligne de commande et initialise le script.
    """
    parser = argparse.ArgumentParser(description="Pipeline de préparation SOTA pour Dynamique Moléculaire")
    parser.add_argument("--data", type=str, default="../../../data/*backbone-dihedrals.npz", help="Chemin des fichiers sources.")
    parser.add_argument("--output", type=str, default="../../../output/dataset", help="Chemin du fichier de sortie.")
    parser.add_argument('-L', type=int, default=32, help="Taille de la fenêtre temporelle.")
    # Le paramètre "blocks" n'est plus nécessaire dans la nouvelle architecture
    parser.add_argument('--train_size', type=float, default=0.7, help="Proportion pour l'entraînement.")
    parser.add_argument('--val_size', type=float, default=0.15, help="Proportion pour la validation.")

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
