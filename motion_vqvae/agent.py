import copy
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from pathlib import Path
from typing import Optional, Dict, Tuple
import wandb
import logging

from .models.models import MotionVQVAE
from .data.vq_dataset import MotionVQDataset, cycle
from .data.motion_data_adapter import MotionDataAdapter
from .utils.utils import cal_loss, cal_vel_loss
from .config_loader import ConfigLoader

log = logging.getLogger(__name__)


class StatsLogger:
    def __init__(self, num_epochs, use_wandb: bool = True):
        self.start = time.time()
        self.num_epochs = num_epochs
        self.progress_format = None
        self.use_wandb = use_wandb

    def time_since(self, ep):
        now = time.time()
        elapsed = now - self.start
        estimated = elapsed * self.num_epochs / ep
        remaining = estimated - elapsed

        em, es = divmod(elapsed, 60)
        rm, rs = divmod(remaining, 60)

        if self.progress_format is None:
            time_format = "%{:d}dm %02ds".format(int(np.log10(rm) + 1))
            perc_format = "%{:d}d %5.1f%%".format(int(np.log10(self.num_epochs) + 1))
            self.progress_format = f"{time_format} (- {time_format}) ({perc_format})"

        return self.progress_format % (em, es, rm, rs, ep, ep / self.num_epochs * 100)

    def log_stats(self, data, print_iter):
        ep = data["epoch"]
        
        # Handle both training and validation data formats
        if "ep_recon_loss" in data:
            # Training data format
            ep_recon_loss = data["ep_recon_loss"]
            ep_vel_loss = data["ep_vel_loss"]
            ep_commit_loss = data["ep_commit_loss"]
            ep_perplexity = data["ep_perplexity"]
            
            if ep % print_iter == 0:
                log.info(
                    "{} | Recon: {:.3e} | Vel: {:.3e} | Commit: {:.3e} | PPL: {:.2f}".format(
                        self.time_since(ep), ep_recon_loss, ep_vel_loss, ep_commit_loss, ep_perplexity
                    )
                )
            
            # Log to wandb if enabled
            if self.use_wandb and wandb.run is not None:
                metrics = {
                    "epoch": ep,
                    "train/reconstruction_loss": ep_recon_loss,
                    "train/velocity_loss": ep_vel_loss,
                    "train/commitment_loss": ep_commit_loss,
                    "train/perplexity": ep_perplexity,
                    "train/total_loss": ep_recon_loss + ep_vel_loss + ep_commit_loss,
                }
                wandb.log(metrics, step=ep)
        
        elif "val_loss" in data:
            # Validation data format - just log to wandb, don't print
            if self.use_wandb and wandb.run is not None:
                metrics = {
                    "epoch": ep,
                    "val/loss": data["val_loss"],
                    "val/reconstruction_loss": data["val_recon_loss"],
                    "val/velocity_loss": data["val_vel_loss"],
                    "val/commitment_loss": data["val_commit_loss"],
                    "val/perplexity": data["val_perplexity"],
                }
                wandb.log(metrics, step=ep)


class MVQVAEAgent:
    """Motion-VQ-VAE agent for training and inference."""

    def __init__(self, config=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        # Load config attributes for compatibility with loss functions
        if config is not None:
            self.recons_loss = config.get('recons_loss', 'l1_smooth')
        else:
            self.recons_loss = 'l1_smooth'
        
        self._should_stop = False
        self.current_epoch = 0
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_model_path = None
        
        # Motion data adapter
        self.motion_adapter = None

    def setup_from_file(self, motion_file: str, motion_ids: Optional[list] = None):
        """Setup model and optimizers from motion file"""
        
        # Don't initialize wandb here - we'll do it after warmup to avoid step conflicts

        # Create motion data adapter
        self.motion_adapter = MotionDataAdapter(self.config)
        
        # Load motion data in MVQ format
        mocap_data, end_indices, frame_size = self.motion_adapter.load_motion_data(motion_file, motion_ids)
        
        # Get motion data properties
        self.frame_size = frame_size
        
        # Update config with actual frame_size
        self.config['frame_size'] = frame_size
        
        # Keep a copy for evaluation later
        self.mocap_data = mocap_data  # [F, frame_size]
        self.end_indices = end_indices

        mean = mocap_data.mean(dim=0)
        std = mocap_data.std(dim=0)
        std[std == 0] = 1.0

        # cache for reconstruction
        self.mean = mean
        self.std = std

        dataset = MotionVQDataset(
            mocap_data=mocap_data,
            end_indices=self.end_indices,
            window_size=self.config['window_size'],
            mean=mean,
            std=std
        )

        # Dataset size is too small, so we need to reduce the batch size
        batch_size = min(self.config['batch_size'], len(dataset))

        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
        )
        self.train_loader_iter = cycle(self.train_loader)
        
        # Initialize VQVAE model
        self.model = MotionVQVAE(
            self, # self as args
            self.config['nb_code'],
            self.config['code_dim'],
            self.config['output_emb_width'],
            self.config['down_t'],
            self.config['stride_t'],
            self.config['width'],
            self.config['depth'],
            self.config['dilation_growth_rate'],
            self.config['vq_act'],
            self.config['vq_norm']
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], betas=(0.9, 0.99), weight_decay=self.config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['lr_scheduler'], gamma=self.config['gamma'])

        log.info(f"Setup complete. Model type: MotionVQVAE, Frame size: {self.frame_size}")

    def setup(self, mocap_data, end_indices):
        """Setup model and optimizers with provided data (legacy method)"""
        
        # Don't initialize wandb here - we'll do it after warmup to avoid step conflicts

        # Get motion data properties
        self.frame_size = mocap_data.shape[1]  # number of dims
        
        # Keep a copy for evaluation later
        self.mocap_data = mocap_data  # [F, frame_size]
        self.end_indices = end_indices

        mean = mocap_data.mean(dim=0)
        std = mocap_data.std(dim=0)
        std[std == 0] = 1.0

        # cache for reconstruction
        self.mean = mean
        self.std = std

        dataset = MotionVQDataset(
            mocap_data=mocap_data,
            end_indices=self.end_indices,
            window_size=self.config['window_size'],
            mean=mean,
            std=std
        )

        # Dataset size is too small, so we need to reduce the batch size
        batch_size = min(self.config['batch_size'], len(dataset))

        self.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
        )
        self.train_loader_iter = cycle(self.train_loader)
        
        # Initialize VQVAE model
        self.model = MotionVQVAE(
            self, # self as args
            self.config['nb_code'],
            self.config['code_dim'],
            self.config['output_emb_width'],
            self.config['down_t'],
            self.config['stride_t'],
            self.config['width'],
            self.config['depth'],
            self.config['dilation_growth_rate'],
            self.config['vq_act'],
            self.config['vq_norm']
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'], betas=(0.9, 0.99), weight_decay=self.config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['lr_scheduler'], gamma=self.config['gamma'])

        log.info(f"Setup complete. Model type: MotionVQVAE, Frame size: {self.frame_size}")

    def load(self, checkpoint: Optional[Path]):
        """Load checkpoint."""
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            log.info(f"Loading model from checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.load_parameters(state_dict)

    def load_parameters(self, state_dict):
        """Load model parameters."""
        self.current_epoch = state_dict.get("epoch", 0)
        if "model" in state_dict:
            self.model.load_state_dict(state_dict["model"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

    def get_state_dict(self):
        """Get state dict for saving."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
        }

    def save(self, path=None, name="last.ckpt"):
        """Save checkpoint."""
        if path is None:
            path = Path("./checkpoints")
        save_dir = Path(path)
        save_dir.mkdir(exist_ok=True)
        state_dict = self.get_state_dict()
        torch.save(state_dict, save_dir / name)
        log.info(f"Saved checkpoint to {save_dir / name}")
    
    def save_best_model(self, current_loss, path=None, name="best_model.ckpt"):
        """Save best model based on validation loss."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            if path is None:
                path = Path("./checkpoints")
            save_dir = Path(path)
            save_dir.mkdir(exist_ok=True)
            state_dict = self.get_state_dict()
            torch.save(state_dict, save_dir / name)
            self.best_model_path = save_dir / name
            log.info(f"New best model saved! Loss: {current_loss:.6f} -> {save_dir / name}")
            return True
        return False

    def update_lr_warm_up(self, optimizer, nb_iter, warm_up_iter, lr):
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        return optimizer, current_lr
    
    @torch.no_grad()
    def validate(self, num_batches=10):
        """Validate the model on a few batches."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_vel_loss = 0.0
        total_commit_loss = 0.0
        total_perplexity = 0.0
        
        for i in range(num_batches):
            try:
                motion = next(self.train_loader_iter).to(self.device)
                pred_motion, loss_commit, perplexity = self.model(motion)
                loss_mot = cal_loss(motion, pred_motion, self)
                loss_vel = cal_vel_loss(motion, pred_motion, self)
                
                loss = loss_mot + self.config['commit'] * loss_commit + self.config['loss_vel'] * loss_vel
                
                total_loss += loss.item()
                total_recon_loss += loss_mot.item()
                total_vel_loss += loss_vel.item()
                total_commit_loss += loss_commit.item()
                total_perplexity += perplexity.item()
            except StopIteration:
                break
        
        self.model.train()
        
        return {
            'val_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_vel_loss': total_vel_loss / num_batches,
            'val_commit_loss': total_commit_loss / num_batches,
            'val_perplexity': total_perplexity / num_batches
        }

    def fit(self):
        log.info("Starting Motion VQ-VAE training")

        logger = StatsLogger(self.config['total_iter'], self.config['use_wandb'])

        ##### Warm-up #####
        for nb_iter in range(1, self.config['warmup_iter'] + 1):
            self.optimizer, current_lr = self.update_lr_warm_up(self.optimizer, nb_iter, self.config['warmup_iter'], self.config['lr'])

            motion = next(self.train_loader_iter).to(self.device)

            pred_motion, loss_commit, perplexity = self.model(motion)
            loss_mot = cal_loss(motion, pred_motion, self)
            loss_vel = cal_vel_loss(motion, pred_motion, self)

            loss = loss_mot + self.config['commit'] * loss_commit + self.config['loss_vel'] * loss_vel

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # For warmup, don't log to wandb to avoid step conflicts
            if nb_iter % self.config['print_iter'] == 0:
                log.info(
                    "Warmup {} | Recon: {:.3e} | Vel: {:.3e} | Commit: {:.3e} | PPL: {:.2f} | LR: {:.5f}".format(
                        logger.time_since(nb_iter), loss_mot.item(), loss_vel.item(), loss_commit.item(), perplexity.item(), current_lr
                    )
                )

            log.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {loss_commit.item():.5f} \t PPL. {perplexity.item():.2f} \t Recons.  {loss_mot.item():.5f}")

        log.info("****Warmup completed****")

        # Initialize wandb after warmup to avoid step conflicts
        if self.config['use_wandb']:
            if wandb.run is None:
                wandb.init(
                    project=self.config['wandb_project'],
                    name=self.config['wandb_run_name'],
                    config=self.config,
                    tags=self.config['wandb_tags'],
                )

        ##### Training #####    
        for nb_iter in range(1, self.config['total_iter'] + 1):
            self.current_epoch = nb_iter
            # --- Data Preparation ---
            motion = next(self.train_loader_iter).to(self.device)

            # --- Forward Pass ---
            pred_motion, loss_commit, perplexity = self.model(motion)

            # --- Loss ---
            loss_mot = cal_loss(motion, pred_motion, self)
            loss_vel = cal_vel_loss(motion, pred_motion, self)
            
            loss = loss_mot + self.config['commit'] * loss_commit + self.config['loss_vel'] * loss_vel

            # --- Backward Pass & Optimization ---
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            logger.log_stats({
                "epoch": nb_iter,
                "ep_recon_loss": loss_mot.item(),
                "ep_vel_loss": loss_vel.item(),
                "ep_commit_loss": loss_commit.item(),
                "ep_perplexity": perplexity.item(),
            }, self.config['print_iter'])

            # --- Validation and Best Model Saving ---
            if nb_iter % self.config['save_every'] == 0:
                # Run validation
                val_stats = self.validate(num_batches=5)
                
                # Log validation stats
                logger.log_stats({
                    "epoch": nb_iter,
                    "val_loss": val_stats['val_loss'],
                    "val_recon_loss": val_stats['val_recon_loss'],
                    "val_vel_loss": val_stats['val_vel_loss'],
                    "val_commit_loss": val_stats['val_commit_loss'],
                    "val_perplexity": val_stats['val_perplexity'],
                }, self.config['print_iter'])
                
                # Save best model
                is_best = self.save_best_model(val_stats['val_loss'])
                
                # Save regular checkpoint
                self.save()
                
                log.info(f"Validation - Loss: {val_stats['val_loss']:.5f}, Recon: {val_stats['val_recon_loss']:.5f}, Vel: {val_stats['val_vel_loss']:.5f}, Commit: {val_stats['val_commit_loss']:.5f}, PPL: {val_stats['val_perplexity']:.2f}")
        
        # --- Save Final Model ---
        log.info("Training completed")
        
        # Final validation and best model summary
        final_val_stats = self.validate(num_batches=10)
        log.info(f"Final Validation - Loss: {final_val_stats['val_loss']:.5f}, Recon: {final_val_stats['val_recon_loss']:.5f}, Vel: {final_val_stats['val_vel_loss']:.5f}, Commit: {final_val_stats['val_commit_loss']:.5f}, PPL: {final_val_stats['val_perplexity']:.2f}")
        
        if self.best_model_path:
            log.info(f"Best model saved at: {self.best_model_path} (Loss: {self.best_loss:.5f})")
        else:
            log.info("No best model was saved during training")
        
        if self.config['use_wandb'] and wandb.run is not None:
            wandb.finish()

    # -----------------------------
    # Evaluation
    # -----------------------------

    def get_codebook_seq(self):
        """Get codebook sequences from the model."""
        return self.model.get_codebook_seq()

    @torch.no_grad()
    def evaluate_policy_rec(self, seq_idx: torch.Tensor) -> torch.Tensor:
        """Reconstruct *ground-truth* motion sequence via the trained VQ-VAE."""

        motion_id = int(seq_idx)

        # ------------------------------------------------------------------
        # 1) Locate frames belonging to the requested motion sequence
        # ------------------------------------------------------------------
        start_idx = 0 if motion_id == 0 else self.end_indices[motion_id - 1] + 1
        end_idx = self.end_indices[motion_id]

        seq = self.mocap_data[start_idx : end_idx + 1]               # [T, C]
        seq_len = seq.shape[0]

        # ------------------------------------------------------------------
        # 2) Prepare windows (normalised)
        # ------------------------------------------------------------------
        seq_norm = (seq - self.mean) / self.std                      # [T, C]
        windows = []
        indices = []  # keep track of (start, length) for fusion

        i = 0
        while i < seq_len:
            win = seq_norm[i : i + self.config['window_size']]                 # [<=W, C]
            orig_len = win.shape[0]
            if orig_len < self.config['window_size']:
                pad = win[-1:].repeat(self.config['window_size'] - orig_len, 1)
                win = torch.cat([win, pad], dim=0)
            windows.append(win)
            indices.append((i, orig_len))
            i += self.config['window_size']

        batch = torch.stack(windows, dim=0).to(self.device)          # [B, W, C]

        # ------------------------------------------------------------------
        # 3) Run through the model
        # ------------------------------------------------------------------
        self.model.eval()
        with torch.no_grad():
            recon_batch, _, _ = self.model(batch)                    # [B, W, C]
            codebook_batch = self.model.encode(batch)
            
        recon_batch = recon_batch * self.std + self.mean       # denorm

        # ------------------------------------------------------------------
        # 4) Stitch windows back together
        # ------------------------------------------------------------------
        recon = torch.zeros_like(seq)
        for win_recon, (start, length) in zip(recon_batch, indices):
            recon[start : start + length] = win_recon[:length]

        codebook_seq = codebook_batch.reshape(-1)

        return recon, seq, codebook_seq

    def evalulate_from_codebook_seq(self, codebook_seq: torch.Tensor) -> torch.Tensor:
        """Evaluate the model from codebook sequence."""
        motion_pred = self.model.forward_decoder(codebook_seq.unsqueeze(0)).squeeze()
        motion_pred_denorm = motion_pred * self.std + self.mean
        return motion_pred_denorm

    # -----------------------------
    # Environment Interaction Helpers
    # -----------------------------

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def train(self):
        """Set model to training mode."""
        self.model.train()
