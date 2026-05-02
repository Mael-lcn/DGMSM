import os
import torch as th
from tqdm import tqdm
from torch.optim import AdamW

from .Evaluation import plot_trajectory, TrajectoryMetrics



class TrainLoop:
    def __init__(
        self,
        *,
        model,
        device,
        diffusion,
        data,
        val_loader,
        batch_size,
        microbatch,
        lr,
        log_interval,
        save_interval,
        resume_checkpoint=None,
        weight_decay=0.0,
        euler_steps=20
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_loader = val_loader
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.euler_steps = euler_steps

        self.step = 0
        self.device = device
        self.model.to(self.device)

        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)

        if resume_checkpoint:
            self._load_and_sync_parameters(resume_checkpoint)

        self.val_best_mae = float('inf') # On cherche à minimiser l'erreur
        self.log_dir = os.environ.get("OPENAI_LOGDIR", "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def _load_and_sync_parameters(self, checkpoint_path):
        print(f"Chargement du modèle depuis {checkpoint_path}...")
        self.model.load_state_dict(th.load(checkpoint_path, map_location=self.device))

    def run_loop(self, max_iter=100000):
        pbar = tqdm(total=max_iter, initial=self.step, desc="Entraînement Flow")

        while self.step < max_iter:
            self.model.train()

            # Récupération des données
            data_ = next(self.data)
            batch = data_["GT"].to(self.device)
            cond = data_["input"].to(self.device)

            self.run_step(batch, cond)

            pbar.update(1)

            if self.step % self.log_interval == 0 and self.step > 0:
                self.evaluate_and_plot()

            if self.step % self.save_interval == 0 and self.step > 0:
                self.save_checkpoint("latest")
                
            self.step += 1

    def run_step(self, batch, cond):
        """
        Gère l'entraînement avec le MICROBATCHING (Accumulation de gradients)
        """
        self.opt.zero_grad()

        # On découpe le batch en micro-batchs
        for i in range(0, batch.shape[0], self.microbatch):
            micro_batch = batch[i : i + self.microbatch]
            micro_cond = cond[i : i + self.microbatch]

            # Calcul de la Loss via le Flow Matching Engine
            losses = self.diffusion.training_losses(
                model=self.model, 
                x_start=micro_batch, 
                mamba_context=micro_cond
            )

            loss = losses["loss"].mean()
            loss = loss * (micro_batch.shape[0] / self.batch_size)
            loss.backward()

        # Mise à jour des poids une fois tout le batch accumulé
        self.opt.step()


    def evaluate_and_plot(self):
        """
        Phase de validation : Génération, Métriques et Graphiques !
        """
        print(f"\n--- Évaluation au step {self.step} ---")
        self.model.eval()
        metrics = TrajectoryMetrics()

        # On met à jour le nombre de steps d'Euler du moteur dynamiquement
        self.diffusion.euler_steps = self.euler_steps

        # On prend un batch de validation
        val_data = next(iter(self.val_loader))
        gt_batch = val_data["GT"].to(self.device)
        cond_batch = val_data["input"].to(self.device)

        shape = (gt_batch.shape[0], 2, gt_batch.shape[2]) # [Batch, 2, 16]

        with th.no_grad():
            # Génération de la trajectoire via le Flow
            pred_batch = self.diffusion.p_sample_loop(
                model=self.model,
                shape=shape,
                mamba_context=cond_batch,
                device=self.device
            )

            # Calcul des métriques sur tout le batch
            metrics.update(gt_batch, pred_batch)
            res = metrics.compute()
            print(metrics)  # Affiche la Circular MAE et MSE

            # Sauvegarde du meilleur modèle
            if res["MAE_circ_deg"] < self.val_best_mae:
                print(f"Nouveau Record ! MAE descendue à {res['MAE_circ_deg']:.2f}°")
                self.val_best_mae = res["MAE_circ_deg"]
                self.save_checkpoint("best_model")

            plot_path = os.path.join(self.log_dir, f"traj_step_{self.step}.png")
            plot_trajectory(gt_batch[0], pred_batch[0], res["MAE_circ_deg"], plot_path)


    def save_checkpoint(self, name_suffix):
        filename = f"model_{name_suffix}.pt"
        save_path = os.path.join(self.log_dir, filename)
        th.save(self.model.state_dict(), save_path)
