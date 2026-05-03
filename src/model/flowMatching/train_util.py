import os
import torch as th
from tqdm import tqdm
from torch.optim import AdamW

from .Evaluation import plot_trajectory, TrajectoryMetrics, compute_ramachandran_jsd, plot_ramachandran



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

        self.train_loss_history = []
        self.val_loss_history = []
        self.val_mae_history = []
        self.val_steps = []
        self.val_jsd_history = []

        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = th.amp.GradScaler('cuda')

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

        self.val_best_jsd = float('inf')

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

            step_loss = self.run_step(batch, cond)
            self.train_loss_history.append(step_loss)

            pbar.update(1)
            pbar.set_postfix({"Loss": f"{step_loss:.4f}"})

            if self.step % self.log_interval == 0 and self.step > 0:
                self.valuate_and_plot()
                self.plot_training_loss()

            if self.step % self.save_interval == 0 and self.step > 0:
                self.save_checkpoint("latest")

            self.step += 1

    def run_step(self, batch, cond):
        """
        Gère l'entraînement avec le MICROBATCHING (Accumulation de gradients)
        """
        self.opt.zero_grad()
        step_loss_total = 0.0

        # On découpe le batch en micro-batchs
        for i in range(0, batch.shape[0], self.microbatch):
            micro_batch = batch[i : i + self.microbatch]
            micro_cond = cond[i : i + self.microbatch]

            with th.autocast(device_type=self.device.type, dtype=th.float16, enabled=self.use_amp):
                losses = self.diffusion.training_losses(
                    model=self.model, 
                    x_start=micro_batch, 
                    mamba_context=micro_cond
                )

            loss = losses["loss"].mean()
            loss = loss * (micro_batch.shape[0] / self.batch_size)

            step_loss_total += loss.item()

            # Rétropropagation avec ou sans le Scaler FP16
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Mise à jour des poids
        if self.use_amp:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
            
        return step_loss_total



    def valuate_and_plot(self):
        print(f"\n--- Validation complète sur tout le dataset au step {self.step} ---")
        self.model.eval()
        metrics = TrajectoryMetrics()
        self.diffusion.euler_steps = self.euler_steps

        all_gt = []
        all_pred = []
        total_val_loss = 0.0

        with th.no_grad():
            for i, batch_data in enumerate(tqdm(self.val_loader, desc="Validation Loop", leave=False)):
                gt_batch = batch_data["GT"].to(self.device)
                cond_batch = batch_data["input"].to(self.device)

                # Validation Flow Loss
                losses = self.diffusion.training_losses(self.model, gt_batch, cond_batch)
                total_val_loss += losses["loss"].mean().item()

                # Inférence (Génération)
                shape = (gt_batch.shape[0], 2, gt_batch.shape[2])
                pred_batch = self.diffusion.p_sample_loop(
                    model=self.model, shape=shape, 
                    mamba_context=cond_batch, device=self.device
                )

                # Accumulation pour les métriques globales
                metrics.update(gt_batch, pred_batch)
                all_gt.append(gt_batch.cpu())
                all_pred.append(pred_batch.cpu())

            # Agrégation des résultats
            res = metrics.compute()
            avg_val_loss = total_val_loss / len(self.val_loader)

            # Concaténation pour le calcul JSD global
            full_gt = th.cat(all_gt, dim=0)
            full_pred = th.cat(all_pred, dim=0)
            jsd_score = compute_ramachandran_jsd(full_gt, full_pred)

            # Logs historiques
            self.val_steps.append(self.step)
            self.val_loss_history.append(avg_val_loss)
            self.val_mae_history.append(res["MAE_circ_deg"])
            self.val_jsd_history.append(jsd_score)

            print(f"Val Flow Loss: {avg_val_loss:.4f} | MAE: {res['MAE_circ_deg']:.2f}°")
            print(f"Thermodynamic JSD: {jsd_score:.4f}")

            if jsd_score < self.val_best_jsd:
                print(f"Nouveau Record Thermodynamique ! JSD descendue à {jsd_score:.4f}")
                self.val_best_jsd = jsd_score
                self.save_checkpoint("best_thermo_model")

            # Sauvegarde des plots (sur le dernier batch pour la trajectoire)
            plot_path_traj = os.path.join(self.log_dir, f"traj_step_{self.step}.png")
            plot_trajectory(gt_batch[0], pred_batch[0], res["MAE_circ_deg"], plot_path_traj)

            plot_path_rama = os.path.join(self.log_dir, f"rama_step_{self.step}.png")
            plot_ramachandran(full_gt, full_pred, jsd_score, plot_path_rama)

    def save_checkpoint(self, name_suffix):
        filename = f"model_{name_suffix}.pt"
        save_path = os.path.join(self.log_dir, filename)
        th.save(self.model.state_dict(), save_path)

    def plot_training_loss(self):
        import numpy as np
        import matplotlib.pyplot as plt

        if len(self.train_loss_history) < 10:
            return
    
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Axe 1 (Gauche) : Flow Loss (Entraînement et Validation)
        ax1.plot(self.train_loss_history, color="#1f77b4", alpha=0.3, label="Train Flow Loss (brute)")

        window = min(100, max(1, len(self.train_loss_history) // 5))
        if window > 1:
            smoothed = np.convolve(self.train_loss_history, np.ones(window)/window, mode='valid')
            x_smoothed = np.arange(window-1, len(self.train_loss_history))
            ax1.plot(x_smoothed, smoothed, color="#005b96", linewidth=2, label="Train Flow Loss (lissée)")

        if len(self.val_steps) > 0:
            ax1.plot(self.val_steps, self.val_loss_history, color="#d62728", marker="o", linestyle="dashed", linewidth=2, label="Val Flow Loss")

        ax1.set_xlabel("Itérations")
        ax1.set_ylabel("Flow Matching Loss (MSE)")
        ax1.set_yscale("log")
        ax1.legend(loc="upper left")

        # Axe 2 (Droite) : MAE Topologique en Degrés
        if len(self.val_mae_history) > 0:
            ax2 = ax1.twinx()
            ax2.plot(self.val_steps, self.val_mae_history, color="#2ca02c", marker="s", linestyle="solid", linewidth=2, label="Val MAE (°)")
            ax2.set_ylabel("Erreur Circulaire MAE (Degrés)")
            ax2.legend(loc="upper right")

        plt.title("Convergence : Mamba + Flow Matching")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_loss_curve.png"), dpi=150)
        plt.close()