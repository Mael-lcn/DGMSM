import torch as th
import math
from .nn import mean_flat



class FlowMatchingEngine:
    """
    Moteur de dynamique continue pour trajectoires angulaires (Torus Flow Matching).
    Gère la topologie circulaire de l'Alanine dipeptide.
    """
    def __init__(self, euler_steps):
        self.euler_steps = euler_steps

    def _map_to_sincos(self, x):
        """
        [Astuce Topologique]
        Transforme les angles [Batch, 2, Longueur] en [Batch, 4, Longueur]
        (sin(phi), cos(phi), sin(psi), cos(psi)) pour aider le réseau.
        """
        return th.cat([th.sin(x), th.cos(x)], dim=1)

    def _get_target_velocity(self, x_start, noise):
        """
        Calcule la "vitesse" cible (le plus court chemin sur le cercle).
        Résultat toujours compris entre -pi et +pi.
        """
        return (x_start - noise + math.pi) % (2 * math.pi) - math.pi

    def q_sample(self, x_start, t, noise):
        """
        Génère l'état intermédiaire x_t sur la géodésique (la ligne droite du cercle).
        Retourne l'état x_t ET la vélocité cible pour optimiser les calculs.
        """
        v_target = self._get_target_velocity(x_start, noise)
        t_expand = t.view(-1, 1, 1) # Alignement des dimensions: [Batch, 1, 1]

        # On avance sur le cercle en partant du bruit
        x_t = (noise + t_expand * v_target + math.pi) % (2 * math.pi) - math.pi
        return x_t, v_target

    def p_sample(self, model, x, t, mamba_context, dt):
        """
        Avance la trajectoire d'un pas dt lors de l'inférence.
        """
        x_mapped = self._map_to_sincos(x)

        # Le modèle prédit la vitesse angulaire (2 canaux)
        v_pred = model(x_mapped, t, mamba_context)
        # Intégration d'Euler sur le cercle
        sample = (x + v_pred * dt + math.pi) % (2 * math.pi) - math.pi

        return {"sample": sample}

    def p_sample_loop(
        self,
        model,
        shape,
        mamba_context,
        noise=None,
        device=None,
        progress=False,
    ):
        """Wrapper final pour générer la trajectoire."""
        final = None
        for sample in self.p_sample_loop_progressive(
            model, shape, mamba_context, noise, device, progress
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        mamba_context,
        noise=None,
        device=None,
        progress=False,
    ):
        """Boucle d'intégration d'Euler de t=0 à t=1."""
        if device is None:
            device = next(model.parameters()).device

        assert len(shape) == 3, "La shape doit être 3D : (Batch, 2, Longueur)"

        # Le bruit initial sur un cercle est Uniforme [-pi, pi], pas Gaussien !
        if noise is not None:
            img = noise
        else:
            img = th.rand(*shape, device=device) * 2 * math.pi - math.pi

        indices = list(range(self.euler_steps))

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)


        power = 2.0 

        for i in indices:
            # Progression linéaire de 0.0 à 1.0
            progression = i / (self.euler_steps - 1)

            # Vrai t_val (commence à 0, finit à 1, ralentit à la fin)
            t_val = 1.0 - (1.0 - progression) ** power

            # Calcul du t suivant pour avoir le vrai dt
            if i < self.euler_steps - 1:
                next_progression = (i + 1) / (self.euler_steps - 1)
                next_t_val = 1.0 - (1.0 - next_progression) ** power
            else:
                next_t_val = 1.0

            dt = next_t_val - t_val
            t = th.tensor([t_val] * shape[0], device=device, dtype=th.float32)

            with th.no_grad():
                out = self.p_sample(model, img, t, mamba_context, dt)
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, mamba_context, noise=None):
        """
        Calcule la loss (MSE) pour l'entraînement.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        # Bruit uniforme angulaire [-pi, pi]
        if noise is None:
            noise = th.rand_like(x_start) * 2 * math.pi - math.pi

        # Tirage Gaussien puis écrasement entre 0 et 1 via la Sigmoïde
        z = th.randn((batch_size,), device=device)
        t = th.sigmoid(z)

        # Récupération de l'état intermédiaire ET de la vraie vitesse
        x_t, v_target = self.q_sample(x_start, t, noise)

        # Projection Sin/Cos pour le réseau
        x_t_mapped = self._map_to_sincos(x_t)

        # Prédiction de la vitesse par le réseau
        v_pred = model(x_t_mapped, t, mamba_context)

        assert v_pred.shape == v_target.shape == x_start.shape

        # Loss = MSE pure dans l'espace des vitesses (espace tangent)
        terms = {}
        terms["mse"] = mean_flat((v_target - v_pred) ** 2)
        terms["loss"] = terms["mse"]

        return terms
