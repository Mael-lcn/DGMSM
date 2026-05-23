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
        print(f"Erreur: {n_steps} < {L}")
        return np.array([])

    return np.array([seq[i : i + L] for i in range(0, n_steps - L + 1, S)])

def split_interleaved_blocks(chunks, stride, G, num_blocks=10, train_ratio=0.8, val_ratio=0.1):
    """
    Répartit les fenêtres en ensembles d'entraînement, validation et test.

    Le découpage par blocs entrelacés assure une couverture temporelle uniforme
    de la trajectoire. L'indépendance statistique est garantie par un intervalle 
    de sécurité fixe G exprimé en pas de temps.

    Args:
        chunks (numpy.ndarray): Tableau de fenêtres (N, L, D).
        stride (int): Pas utilisé lors de la création des fenêtres.
        G (int): Intervalle de sécurité en unités de temps.
        num_blocks (int): Nombre de blocs temporels pour l'échantillonnage.
        train_ratio (float): Proportion de données pour l'entraînement.
        val_ratio (float): Proportion de données pour la validation.

    Returns:
        tuple: (train_chunks, val_chunks, test_chunks) sous forme de tableaux numpy.
    """
    N = len(chunks)
    gap_chunks = math.ceil(G / stride)
    block_size = N // num_blocks

    train_idx, val_idx, test_idx = [], [], []

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size if i < num_blocks - 1 else N

        usable_size = (end - start) - (2 * gap_chunks)
        if usable_size <= 0:
            continue

        t_size = int(usable_size * train_ratio)
        v_size = int(usable_size * val_ratio)

        t_end = start + t_size
        v_start = t_end + gap_chunks
        v_end = v_start + v_size
        test_start = v_end + gap_chunks
        
        train_idx.extend(range(start, t_end))
        val_idx.extend(range(v_start, v_end))
        test_idx.extend(range(test_start, end))

    return chunks[train_idx], chunks[val_idx], chunks[test_idx]


def run(args):
    """
    Exécute le pipeline de transformation et de séparation des données.

    Cette fonction gère le chargement des fichiers, la création des fenêtres, 
    la séparation hermétique et l'exportation des ensembles de données. 
    L'exportation utilise un format non compressé pour optimiser la vitesse de lecture.

    Args:
        args (argparse.Namespace): Objet contenant les paramètres de configuration.

    Raises:
        ValueError: Si la somme des proportions d'entraînement et de validation 
            est supérieure ou égale à 1.0.
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

                max_tau_int = 0

                for dim in range(data.shape[1]):
                    obs_sin = np.sin(data[:, dim])
                    obs_cos = np.cos(data[:, dim])

                    # On calcule les deux autocorrélations
                    acf_sin = autocorr_fft(obs_sin)
                    acf_cos = autocorr_fft(obs_cos)

                    # L'autocorrélation circulaire totale est la moyenne des deux
                    acf_totale = (acf_sin + acf_cos) / 2.0

                    tau_int_dim = compute_tau_int(acf_totale)

                    # On garde le temps le plus long pour être sûr que tout est décorrelé
                    if tau_int_dim > max_tau_int:
                        max_tau_int = tau_int_dim

                print(f"[{key}] Temps de décorrélation max : {max_tau_int:.2f} pas")

                optimal_stride = max(1, int(max_tau_int / 10))
                optimal_gap = int(2 * max_tau_int) 

                print(f"-> Utilise un stride (S) de {optimal_stride}")
                print(f"-> Utilise un gap (G) de {optimal_gap}")

                current_chunks = create_chunks(data, args.L, optimal_stride)
                if len(current_chunks) == 0:
                    continue

                train_c, val_c, test_c = split_interleaved_blocks(
                    current_chunks, 
                    optimal_stride, 
                    optimal_gap, 
                    num_blocks=args.blocks, 
                    train_ratio=args.train_size, 
                    val_ratio=args.val_size
                )

                if len(train_c) > 0: all_train.append(train_c)
                if len(val_c) > 0: all_val.append(val_c)
                if len(test_c) > 0: all_test.append(test_c)

    if not all_train:
        print("Échec de la génération : aucun échantillon valide produit.")
        return

    x_train = np.vstack(all_train)
    x_val = np.vstack(all_val) if all_val else np.array([])
    x_test = np.vstack(all_test) if all_test else np.array([])

    np.random.shuffle(x_train)
    np.random.shuffle(x_val)
    np.random.shuffle(x_test)

    print(f"Taille de x_train : {len(x_train)}")
    print(f"Taille de x_val   : {len(x_val)}")
    print(f"Taille de x_test  : {len(x_test)}")

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
    parser = argparse.ArgumentParser(description="Pipeline de préparation de données de dynamique moléculaire")
    parser.add_argument("--data", type=str, default="../../../data/*backbone-dihedrals.npz", help="Chemin des fichiers sources.")
    parser.add_argument("--output", type=str, default="../../../output/dataset", help="Chemin du fichier de sortie.")
    parser.add_argument('-L', type=int, default=32, help="Taille de la fenêtre temporelle.")
    parser.add_argument("--blocks", type=int, default=10, help="Nombre de blocs temporels.")
    parser.add_argument('--train_size', type=float, default=0.7, help="Proportion pour l'entraînement.")
    parser.add_argument('--val_size', type=float, default=0.15, help="Proportion pour la validation.")

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
