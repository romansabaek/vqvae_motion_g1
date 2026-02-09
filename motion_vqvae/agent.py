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
from .models.policy_classifier import PolicyIDClassifier
from .models.sequence_policy_classifier import SequencePolicyIDClassifier
from .models.latent_predictor import LatentPredictor
from .models.motion_predictor import MotionPredictor
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
                log_str = "{} | Recon: {:.3e} | Vel: {:.3e} | Commit: {:.3e} | PPL: {:.2f}".format(
                    self.time_since(ep), ep_recon_loss, ep_vel_loss, ep_commit_loss, ep_perplexity
                )
                if "ep_policy_loss" in data:
                    log_str += " | Policy: {:.3e}".format(data["ep_policy_loss"])
                log.info(log_str)
            
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
                if "ep_policy_loss" in data:
                    metrics["train/policy_loss"] = data["ep_policy_loss"]
                    metrics["train/total_loss"] += data["ep_policy_loss"]
                if "ep_latent_pred_loss" in data:
                    metrics["train/latent_predictor_loss"] = data["ep_latent_pred_loss"]
                    metrics["train/total_loss"] += data["ep_latent_pred_loss"]
                if "ep_motion_pred_loss" in data:
                    metrics["train/motion_predictor_loss"] = data["ep_motion_pred_loss"]
                    metrics["train/total_loss"] += data["ep_motion_pred_loss"]
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
                if "val_policy_loss" in data:
                    metrics["val/policy_loss"] = data["val_policy_loss"]
                if "val_latent_pred_loss" in data:
                    metrics["val/latent_predictor_loss"] = data["val_latent_pred_loss"]
                if "val_motion_pred_loss" in data:
                    metrics["val/motion_predictor_loss"] = data["val_motion_pred_loss"]
                wandb.log(metrics, step=ep)


class MVQVAEAgent:
    
    def _compute_policy_loss(self, codebook_seqs, policy_classes):
        """
        Compute policy loss for both chunk-wise and sequence-wise prediction.
        
        Args:
            codebook_seqs: Codebook sequences [batch_size, seq_len]
            policy_classes: Policy class indices [batch_size, window_size]
        
        Returns:
            loss_policy: Policy loss scalar
        """
        batch_size = policy_classes.shape[0]
        window_size = policy_classes.shape[1]
        
        if self.use_sequence_policy:
            # Sequence-wise: predict per timestep
            policy_logits = self.policy_classifier(codebook_seqs)  # [batch_size, seq_len, num_policies]
            seq_len = policy_logits.shape[1]
            
            # Map frame-level policy IDs to codebook-level
            policy_classes_expanded = policy_classes.unsqueeze(2)  # [batch_size, window_size, 1]
            policy_classes_expanded = F.interpolate(
                policy_classes_expanded.float().permute(0, 2, 1),  # [batch_size, 1, window_size]
                size=seq_len,
                mode='nearest'
            ).permute(0, 2, 1).long().squeeze(2)  # [batch_size, seq_len]
            
            # Per-timestep cross-entropy loss
            policy_logits_flat = policy_logits.view(-1, policy_logits.shape[-1])  # [batch_size * seq_len, num_policies]
            policy_classes_flat = policy_classes_expanded.view(-1)  # [batch_size * seq_len]
            
            return F.cross_entropy(policy_logits_flat, policy_classes_flat)
        else:
            # Chunk-wise: predict one policy ID per window
            policy_logits = self.policy_classifier(codebook_seqs)  # [batch_size, num_policies]
            num_policies = len(self.policy_id_to_class)
            
            use_soft_labels = self.config.get('policy_use_soft_labels', True)
            transition_weight = self.config.get('policy_transition_weight', 2.0)
            
            if use_soft_labels:
                policy_targets = torch.zeros(batch_size, num_policies, device=policy_classes.device)
                window_weights = torch.ones(batch_size, device=policy_classes.device)
                
                for i in range(batch_size):
                    values, counts = torch.unique(policy_classes[i], return_counts=True)
                    counts_float = counts.float()
                    num_unique = len(values)
                    
                    if num_unique > 1:
                        # Transition window: soft label
                        probs = counts_float / counts_float.sum()
                        for val, prob in zip(values, probs):
                            policy_targets[i, val] = prob
                        window_weights[i] = transition_weight
                    else:
                        # Stable window: hard label
                        policy_targets[i, values[0]] = 1.0
                
                policy_probs = F.log_softmax(policy_logits, dim=1)
                loss_policy = -torch.sum(policy_targets * policy_probs, dim=1)
                return (loss_policy * window_weights).mean()
            else:
                # Hard label (mode)
                policy_mode = torch.zeros(batch_size, dtype=torch.long, device=policy_classes.device)
                for i in range(batch_size):
                    values, counts = torch.unique(policy_classes[i], return_counts=True)
                    policy_mode[i] = values[torch.argmax(counts)]
                
                return F.cross_entropy(policy_logits, policy_mode)
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
        
        # Auxiliary models (initialized to None, set up in setup_from_file or load)
        self.policy_classifier = None
        self.latent_predictor = None
        self.motion_predictor = None
        self.use_policy_loss = False
        self.use_latent_predictor = False
        self.use_motion_predictor = False
        self.policy_id_to_class = {}

    def setup_from_file(self, motion_file: str, motion_ids: Optional[list] = None):
        """Setup model and optimizers from motion file"""
        
        # Don't initialize wandb here - we'll do it after warmup to avoid step conflicts

        # Create motion data adapter
        self.motion_adapter = MotionDataAdapter(self.config)
        
        # Load motion data in MVQ format (fills adapter cache)
        self.motion_adapter.load_motion_data(motion_file, motion_ids)
        
        # Reference-style accessors for parity
        mocap_data = self.motion_adapter.get_mvq_data()
        self.frame_size = self.motion_adapter.FRAME_SIZE
        self.end_indices = self.motion_adapter.get_mvq_end_indices()
        policy_ids = self.motion_adapter.get_policy_ids()  # Get policy IDs if available
        
        # Update config with actual frame_size
        self.config['frame_size'] = self.frame_size
        
        # Keep a copy for evaluation later
        self.mocap_data = mocap_data  # [F, frame_size]

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
            std=std,
            policy_ids=policy_ids
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
        
        # Initialize policy classifier if policy IDs are available
        self.policy_classifier = None
        self.use_policy_loss = policy_ids is not None and self.config.get('use_policy_loss', False)
        self.use_sequence_policy = self.config.get('policy_use_sequence_wise', False)  # Sequence-wise prediction
        
        if self.use_policy_loss:
            # Determine number of unique policy IDs
            unique_policies = np.unique(policy_ids)
            num_policies = len(unique_policies)
            log.info(f"Initializing policy classifier with {num_policies} unique policies")
            log.info(f"Using {'sequence-wise' if self.use_sequence_policy else 'chunk-wise'} policy prediction")
            
            if self.use_sequence_policy:
                # Sequence-wise: predict policy ID per timestep
                self.policy_classifier = SequencePolicyIDClassifier(
                    num_codebooks=self.config['nb_code'],
                    num_policies=num_policies,
                    code_dim=self.config['code_dim'],
                    hidden_dim=self.config.get('policy_classifier_hidden_dim', 256),
                    num_layers=self.config.get('policy_classifier_layers', 2),
                    dropout=self.config.get('policy_classifier_dropout', 0.1),
                    architecture=self.config.get('policy_classifier_architecture', 'lstm'),  # 'lstm', 'transformer', or 'cnn1d'
                    num_heads=self.config.get('policy_classifier_num_heads', 8),  # For transformer
                    kernel_size=self.config.get('policy_classifier_kernel_size', 3),  # For CNN1D
                ).to(self.device)
            else:
                # Chunk-wise: predict one policy ID per window (original)
                self.policy_classifier = PolicyIDClassifier(
                    num_codebooks=self.config['nb_code'],
                    num_policies=num_policies,
                    code_dim=self.config['code_dim'],
                    hidden_dim=self.config.get('policy_classifier_hidden_dim', 256),
                    num_layers=self.config.get('policy_classifier_layers', 2),
                    dropout=self.config.get('policy_classifier_dropout', 0.1),
                    architecture=self.config.get('policy_classifier_architecture', 'lstm'),  # 'mlp', 'cnn1d', 'lstm', or 'transformer'
                    num_heads=self.config.get('policy_classifier_num_heads', 8),  # For transformer
                    kernel_size=self.config.get('policy_classifier_kernel_size', 3),  # For CNN1D
                ).to(self.device)
            
            # Create mapping from policy IDs to class indices (0 to num_policies-1)
            self.policy_id_to_class = {pid: idx for idx, pid in enumerate(unique_policies)}
            log.info(f"Policy ID mapping: {self.policy_id_to_class}")
        
        # Initialize latent predictor for dynamics learning (legacy, for predicting next token)
        self.latent_predictor = None
        self.use_latent_predictor = self.config.get('use_latent_predictor', False)
        
        if self.use_latent_predictor:
            architecture = self.config.get('latent_predictor_architecture', 'lstm')  # 'mlp', 'lstm', 'transformer', or 'cnn1d'
            log.info(f"Initializing latent predictor for dynamics learning (architecture: {architecture})")
            self.latent_predictor = LatentPredictor(
                num_codebooks=self.config['nb_code'],
                code_dim=self.config['code_dim'],
                hidden_dim=self.config.get('latent_predictor_hidden_dim', 256),
                num_layers=self.config.get('latent_predictor_layers', 2),
                dropout=self.config.get('latent_predictor_dropout', 0.1),
                pred_len=self.config.get('latent_predictor_pred_len', 1),  # Number of future steps to predict
                architecture=architecture,
                num_heads=self.config.get('latent_predictor_num_heads', 8),  # For transformer
                kernel_size=self.config.get('latent_predictor_kernel_size', 3),  # For CNN1D
            ).to(self.device)
        
        # Initialize motion predictor for future raw motion prediction (e.g., 0.5 seconds ahead)
        self.motion_predictor = None
        self.use_motion_predictor = self.config.get('use_motion_predictor', False)
        
        if self.use_motion_predictor:
            architecture = self.config.get('motion_predictor_architecture', 'lstm')  # 'mlp', 'lstm', 'transformer', or 'cnn1d'
            # Calculate pred_frames: 0.5 seconds ahead at 30 FPS = 15 frames
            fps = self.config.get('fps', 30.0)
            future_time = self.config.get('motion_predictor_future_time', 0.5)  # seconds ahead
            pred_frames = int(fps * future_time)
            log.info(f"Initializing motion predictor for future raw motion prediction (architecture: {architecture}, {future_time}s ahead = {pred_frames} frames at {fps} FPS)")
            self.motion_predictor = MotionPredictor(
                num_codebooks=self.config['nb_code'],
                code_dim=self.config['code_dim'],
                frame_size=self.frame_size,
                hidden_dim=self.config.get('motion_predictor_hidden_dim', 256),
                num_layers=self.config.get('motion_predictor_layers', 2),
                dropout=self.config.get('motion_predictor_dropout', 0.1),
                pred_frames=pred_frames,
                architecture=architecture,
                num_heads=self.config.get('motion_predictor_num_heads', 8),  # For transformer
                kernel_size=self.config.get('motion_predictor_kernel_size', 3),  # For CNN1D
            ).to(self.device)
        
        # Setup optimizer - include policy classifier, latent predictor, and motion predictor if available
        model_params = list(self.model.parameters())
        if self.policy_classifier is not None:
            model_params += list(self.policy_classifier.parameters())
        if self.latent_predictor is not None:
            model_params += list(self.latent_predictor.parameters())
        if self.motion_predictor is not None:
            model_params += list(self.motion_predictor.parameters())
        
        self.optimizer = optim.AdamW(model_params, lr=self.config['lr'], betas=(0.9, 0.99), weight_decay=self.config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['lr_scheduler'], gamma=self.config['gamma'])

        log.info(f"Setup complete. Model type: MotionVQVAE, Frame size: {self.frame_size}, Policy loss: {self.use_policy_loss}, Latent predictor: {self.use_latent_predictor}, Motion predictor: {self.use_motion_predictor}")

    
    def load(self, checkpoint: Optional[Path]):
        """Load checkpoint."""
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            log.info(f"Loading model from checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.load_parameters(state_dict)

    def load_parameters(self, state_dict):
        """Load model parameters."""
        self.current_epoch = state_dict.get("epoch", 0)
        if "model" in state_dict:
            if self.model is None:
                raise AttributeError("Model not initialized. Call setup_from_file() or initialize model before loading checkpoint.")
            self.model.load_state_dict(state_dict["model"])
        if "policy_classifier" in state_dict and self.policy_classifier is not None:
            self.policy_classifier.load_state_dict(state_dict["policy_classifier"])
        if "latent_predictor" in state_dict and self.latent_predictor is not None:
            self.latent_predictor.load_state_dict(state_dict["latent_predictor"])
        if "motion_predictor" in state_dict and self.motion_predictor is not None:
            self.motion_predictor.load_state_dict(state_dict["motion_predictor"])
        if "optimizer" in state_dict and hasattr(self, 'optimizer') and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer"])
            except (ValueError, KeyError) as e:
                log.warning(f"Could not load optimizer state: {e}. Continuing without optimizer state (this is fine for evaluation).")
        if "scheduler" in state_dict and hasattr(self, 'scheduler') and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state_dict["scheduler"])
            except (ValueError, KeyError) as e:
                log.warning(f"Could not load scheduler state: {e}. Continuing without scheduler state.")
        # Load best model tracking
        self.best_loss = state_dict.get("best_loss", float('inf'))
        best_path_str = state_dict.get("best_model_path")
        self.best_model_path = Path(best_path_str) if best_path_str else None
        if self.best_model_path is not None:
            log.info(f"Loaded best model tracking: loss={self.best_loss:.6f}, path={self.best_model_path}")
        # Load normalization statistics if available
        if "mean" in state_dict and state_dict["mean"] is not None:
            self.mean = state_dict["mean"].to(self.device) if isinstance(state_dict["mean"], torch.Tensor) else torch.tensor(state_dict["mean"], device=self.device)
        if "std" in state_dict and state_dict["std"] is not None:
            self.std = state_dict["std"].to(self.device) if isinstance(state_dict["std"], torch.Tensor) else torch.tensor(state_dict["std"], device=self.device)

    def get_state_dict(self):
        """Get state dict for saving."""
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
        }
        if self.policy_classifier is not None:
            state["policy_classifier"] = self.policy_classifier.state_dict()
            # Save metadata needed for loading
            state["num_policies"] = self.policy_classifier.num_policies
            state["policy_id_to_class"] = self.policy_id_to_class
        if self.latent_predictor is not None:
            state["latent_predictor"] = self.latent_predictor.state_dict()
        if self.motion_predictor is not None:
            state["motion_predictor"] = self.motion_predictor.state_dict()
        # Save normalization statistics for inference
        if hasattr(self, 'mean') and self.mean is not None:
            state["mean"] = self.mean.cpu()
        if hasattr(self, 'std') and self.std is not None:
            state["std"] = self.std.cpu()
        return state

    def save(self, path=None, name="last.ckpt"):
        """Save checkpoint."""
        if path is None:
            # Use checkpoint_dir from config if available, otherwise default to ./checkpoints
            if self.config is not None and 'checkpoint_dir' in self.config:
                path = Path(self.config['checkpoint_dir'])
            else:
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
                # Use checkpoint_dir from config if available, otherwise default to ./checkpoints
                if self.config is not None and 'checkpoint_dir' in self.config:
                    path = Path(self.config['checkpoint_dir'])
                else:
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
        if self.policy_classifier is not None:
            self.policy_classifier.eval()
        if self.latent_predictor is not None:
            self.latent_predictor.eval()
        if self.motion_predictor is not None:
            self.motion_predictor.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_vel_loss = 0.0
        total_commit_loss = 0.0
        total_perplexity = 0.0
        total_policy_loss = 0.0
        total_latent_pred_loss = 0.0
        total_motion_pred_loss = 0.0
        
        for i in range(num_batches):
            try:
                data = next(self.train_loader_iter)
                if isinstance(data, (tuple, list)) and len(data) == 2:
                    motion, policy_ids = data
                    motion = motion.to(self.device)
                    if policy_ids is not None:
                        policy_ids = policy_ids.to(self.device)
                else:
                    # Fallback: data is just motion tensor
                    motion = data.to(self.device) if isinstance(data, torch.Tensor) else torch.stack(data).to(self.device)
                    policy_ids = None
                
                pred_motion, loss_commit, perplexity = self.model(motion)
                loss_mot = cal_loss(motion, pred_motion, self)
                loss_vel = cal_vel_loss(motion, pred_motion, self)
                
                loss = loss_mot + self.config['commit'] * loss_commit + self.config['loss_vel'] * loss_vel
                
                # Policy loss (same logic as training)
                if self.use_policy_loss and policy_ids is not None:
                    codebook_seqs = self.model.encode(motion)
                    policy_classes = torch.zeros_like(policy_ids)
                    for pid, class_idx in self.policy_id_to_class.items():
                        mask = (policy_ids == pid)
                        policy_classes[mask] = class_idx
                    
                    loss_policy = self._compute_policy_loss(codebook_seqs, policy_classes)
                    
                    policy_loss_weight = self.config.get('policy_loss_weight', 0.1)
                    loss = loss + policy_loss_weight * loss_policy
                    total_policy_loss += loss_policy.item()
                
                # Latent predictor loss (same logic as training)
                if self.use_latent_predictor:
                    codebook_seqs = self.model.encode(motion)
                    pred_len = self.config.get('latent_predictor_pred_len', 1)
                    
                    if codebook_seqs.shape[1] > pred_len:
                        input_seq = codebook_seqs[:, :-pred_len]
                        target_seq = codebook_seqs[:, -pred_len:]
                        
                        pred_logits = self.latent_predictor(input_seq)
                        pred_logits_flat = pred_logits.view(-1, pred_logits.shape[-1])
                        target_seq_flat = target_seq.view(-1)
                        
                        loss_latent_pred = F.cross_entropy(pred_logits_flat, target_seq_flat)
                        latent_pred_weight = self.config.get('latent_predictor_weight', 0.1)
                        loss = loss + latent_pred_weight * loss_latent_pred
                        total_latent_pred_loss += loss_latent_pred.item()
                
                # Motion predictor loss (same logic as training)
                if self.use_motion_predictor:
                    codebook_seqs = self.model.encode(motion)
                    fps = self.config.get('fps', 30.0)
                    future_time = self.config.get('motion_predictor_future_time', 0.5)
                    pred_frames = int(fps * future_time)
                    window_size = motion.shape[1]
                    
                    if window_size > pred_frames:
                        input_seq = codebook_seqs
                        # Target: future raw motion frames starting from position pred_frames ahead
                        target_start_idx = pred_frames
                        target_end_idx = min(target_start_idx + pred_frames, window_size)
                        target_motion = motion[:, target_start_idx:target_end_idx, :]
                        
                        # If we don't have enough frames, pad with the last frame
                        if target_motion.shape[1] < pred_frames:
                            last_frame = motion[:, -1:, :].repeat(1, pred_frames - target_motion.shape[1], 1)
                            target_motion = torch.cat([target_motion, last_frame], dim=1)
                        
                        pred_motion = self.motion_predictor(input_seq)
                        
                        # Ensure target and prediction have the same shape
                        if target_motion.shape[1] != pred_frames:
                            target_motion = target_motion[:, :pred_frames, :]
                        
                        loss_type = self.config.get('motion_predictor_loss_type', 'mse')
                        if loss_type == 'mse':
                            loss_motion_pred = F.mse_loss(pred_motion, target_motion)
                        elif loss_type == 'l1':
                            loss_motion_pred = F.l1_loss(pred_motion, target_motion)
                        elif loss_type == 'smooth_l1':
                            loss_motion_pred = F.smooth_l1_loss(pred_motion, target_motion)
                        else:
                            loss_motion_pred = F.mse_loss(pred_motion, target_motion)
                        
                        motion_pred_weight = self.config.get('motion_predictor_weight', 0.1)
                        loss = loss + motion_pred_weight * loss_motion_pred
                        total_motion_pred_loss += loss_motion_pred.item()
                
                total_loss += loss.item()
                total_recon_loss += loss_mot.item()
                total_vel_loss += loss_vel.item()
                total_commit_loss += loss_commit.item()
                total_perplexity += perplexity.item()
            except StopIteration:
                break
        
        self.model.train()
        if self.policy_classifier is not None:
            self.policy_classifier.train()
        if self.latent_predictor is not None:
            self.latent_predictor.train()
        if self.motion_predictor is not None:
            self.motion_predictor.train()
        
        result = {
            'val_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_vel_loss': total_vel_loss / num_batches,
            'val_commit_loss': total_commit_loss / num_batches,
            'val_perplexity': total_perplexity / num_batches
        }
        if self.use_policy_loss:
            result['val_policy_loss'] = total_policy_loss / num_batches
        if self.use_latent_predictor:
            result['val_latent_pred_loss'] = total_latent_pred_loss / num_batches
        if self.use_motion_predictor:
            result['val_motion_pred_loss'] = total_motion_pred_loss / num_batches
        
        return result

    def fit(self):
        log.info("Starting Motion VQ-VAE training")

        logger = StatsLogger(self.config['total_iter'], self.config['use_wandb'])

        ##### Warm-up #####
        for nb_iter in range(1, self.config['warmup_iter'] + 1):
            self.optimizer, current_lr = self.update_lr_warm_up(self.optimizer, nb_iter, self.config['warmup_iter'], self.config['lr'])

            data = next(self.train_loader_iter)
            if isinstance(data, (tuple, list)) and len(data) == 2:
                motion, policy_ids = data
                motion = motion.to(self.device)
                policy_ids = policy_ids.to(self.device)
                # Check if policy_ids are dummy values (-1) indicating no policy IDs available
                # Check if all values in the entire batch are -1
                if (policy_ids == -1).all().item():
                    policy_ids = None
            else:
                # Fallback: data is just motion tensor
                motion = data.to(self.device) if isinstance(data, torch.Tensor) else torch.stack(data).to(self.device)
                policy_ids = None

            pred_motion, loss_commit, perplexity = self.model(motion)
            loss_mot = cal_loss(motion, pred_motion, self)
            loss_vel = cal_vel_loss(motion, pred_motion, self)

            loss = loss_mot + self.config['commit'] * loss_commit + self.config['loss_vel'] * loss_vel
            
            # Policy loss during warmup (optional, can be disabled)
            if self.use_policy_loss and policy_ids is not None and self.config.get('use_policy_loss_warmup', False):
                # Allow gradients to flow for policy loss
                codebook_seqs = self.model.encode(motion)
                policy_classes = torch.zeros_like(policy_ids)
                for pid, class_idx in self.policy_id_to_class.items():
                    mask = (policy_ids == pid)
                    policy_classes[mask] = class_idx
                
                loss_policy = self._compute_policy_loss(codebook_seqs, policy_classes)
                
                policy_loss_weight = self.config.get('policy_loss_weight', 0.1)
                loss = loss + policy_loss_weight * loss_policy

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
            data = next(self.train_loader_iter)
            if isinstance(data, (tuple, list)) and len(data) == 2:
                motion, policy_ids = data
                motion = motion.to(self.device)
                policy_ids = policy_ids.to(self.device)
                # Check if policy_ids are dummy values (-1) indicating no policy IDs available
                # Check if all values in the entire batch are -1
                if (policy_ids == -1).all().item():
                    policy_ids = None
            else:
                # Fallback: data is just motion tensor
                motion = data.to(self.device) if isinstance(data, torch.Tensor) else torch.stack(data).to(self.device)
                policy_ids = None

            # --- Forward Pass ---
            pred_motion, loss_commit, perplexity = self.model(motion)

            # --- Loss ---
            loss_mot = cal_loss(motion, pred_motion, self)
            loss_vel = cal_vel_loss(motion, pred_motion, self)
            
            loss = loss_mot + self.config['commit'] * loss_commit + self.config['loss_vel'] * loss_vel
            
            # --- Auxiliary Policy ID Loss ---
            loss_policy = torch.tensor(0.0, device=self.device)
            if self.use_policy_loss and policy_ids is not None:
                # Get codebook sequences from the model
                # Note: We want gradients to flow through encoder for policy loss
                codebook_seqs = self.model.encode(motion)  # [batch_size, seq_len]
                
                # Convert policy IDs to class indices
                policy_classes = torch.zeros_like(policy_ids)
                for pid, class_idx in self.policy_id_to_class.items():
                    mask = (policy_ids == pid)
                    policy_classes[mask] = class_idx
                
                # Compute policy loss using helper function
                loss_policy = self._compute_policy_loss(codebook_seqs, policy_classes)
                
                # Add to total loss with weight
                policy_loss_weight = self.config.get('policy_loss_weight', 0.1)
                loss = loss + policy_loss_weight * loss_policy
            
            # --- Latent Predictor Loss (Dynamics Learning) ---
            loss_latent_pred = torch.tensor(0.0, device=self.device)
            if self.use_latent_predictor:
                # Get codebook sequences from the model
                codebook_seqs = self.model.encode(motion)  # [batch_size, seq_len]
                
                # Predict future codebook sequences
                # Input: current sequence, Target: next step(s) in sequence
                pred_len = self.config.get('latent_predictor_pred_len', 1)
                
                if codebook_seqs.shape[1] > pred_len:
                    # Use first (seq_len - pred_len) steps as input
                    input_seq = codebook_seqs[:, :-pred_len]  # [batch_size, seq_len - pred_len]
                    # Use last pred_len steps as target
                    target_seq = codebook_seqs[:, -pred_len:]  # [batch_size, pred_len]
                    
                    # Predict future sequences
                    pred_logits = self.latent_predictor(input_seq)  # [batch_size, pred_len, num_codebooks]
                    
                    # Reshape for cross-entropy: [batch_size * pred_len, num_codebooks]
                    pred_logits_flat = pred_logits.view(-1, pred_logits.shape[-1])
                    target_seq_flat = target_seq.view(-1)  # [batch_size * pred_len]
                    
                    # Compute cross-entropy loss
                    loss_latent_pred = F.cross_entropy(pred_logits_flat, target_seq_flat)
                    
                    # Add to total loss with weight
                    latent_pred_weight = self.config.get('latent_predictor_weight', 0.1)
                    loss = loss + latent_pred_weight * loss_latent_pred
            
            # --- Motion Predictor Loss (Future Raw Motion Prediction) ---
            loss_motion_pred = torch.tensor(0.0, device=self.device)
            if self.use_motion_predictor:
                # Get codebook sequences from the model
                codebook_seqs = self.model.encode(motion)  # [batch_size, seq_len]
                
                # Calculate how many frames to predict ahead (e.g., 0.5 seconds)
                fps = self.config.get('fps', 30.0)
                future_time = self.config.get('motion_predictor_future_time', 0.5)  # seconds ahead
                pred_frames = int(fps * future_time)
                
                # motion shape: [batch_size, window_size, frame_size]
                # We need to extract future motion frames from the motion tensor
                window_size = motion.shape[1]
                
                if window_size > pred_frames:
                    # Use codebook sequence from the current window to predict motion frames that are pred_frames ahead
                    # Input: codebook sequence from the entire current window
                    input_seq = codebook_seqs  # [batch_size, seq_len]
                    
                    # Target: future raw motion frames starting from position pred_frames ahead in the window
                    # This simulates predicting 0.5 seconds into the future
                    # Target is the motion frames starting from index pred_frames
                    target_start_idx = pred_frames
                    target_end_idx = min(target_start_idx + pred_frames, window_size)
                    target_motion = motion[:, target_start_idx:target_end_idx, :]  # [batch_size, actual_frames, frame_size]
                    
                    # If we don't have enough frames, pad with the last frame
                    if target_motion.shape[1] < pred_frames:
                        last_frame = motion[:, -1:, :].repeat(1, pred_frames - target_motion.shape[1], 1)
                        target_motion = torch.cat([target_motion, last_frame], dim=1)
                    
                    # Predict future motion
                    pred_motion = self.motion_predictor(input_seq)  # [batch_size, pred_frames, frame_size]
                    
                    # Ensure target and prediction have the same shape
                    if target_motion.shape[1] != pred_frames:
                        target_motion = target_motion[:, :pred_frames, :]
                    
                    # Compute MSE loss on raw motion features
                    # Optionally use L1 loss or smooth L1 loss
                    loss_type = self.config.get('motion_predictor_loss_type', 'mse')  # 'mse', 'l1', 'smooth_l1'
                    if loss_type == 'mse':
                        loss_motion_pred = F.mse_loss(pred_motion, target_motion)
                    elif loss_type == 'l1':
                        loss_motion_pred = F.l1_loss(pred_motion, target_motion)
                    elif loss_type == 'smooth_l1':
                        loss_motion_pred = F.smooth_l1_loss(pred_motion, target_motion)
                    else:
                        raise ValueError(f"Unknown motion predictor loss type: {loss_type}")
                    
                    # Add to total loss with weight
                    motion_pred_weight = self.config.get('motion_predictor_weight', 0.1)
                    loss = loss + motion_pred_weight * loss_motion_pred

            # --- Backward Pass & Optimization ---
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            log_data = {
                "epoch": nb_iter,
                "ep_recon_loss": loss_mot.item(),
                "ep_vel_loss": loss_vel.item(),
                "ep_commit_loss": loss_commit.item(),
                "ep_perplexity": perplexity.item(),
            }
            if self.use_policy_loss:
                log_data["ep_policy_loss"] = loss_policy.item()
            if self.use_latent_predictor:
                log_data["ep_latent_pred_loss"] = loss_latent_pred.item()
            if self.use_motion_predictor:
                log_data["ep_motion_pred_loss"] = loss_motion_pred.item()
            
            logger.log_stats(log_data, self.config['print_iter'])

            # --- Validation and Best Model Saving ---
            if nb_iter % self.config['save_every'] == 0:
                # Run validation
                val_stats = self.validate(num_batches=5)
                
                # Log validation stats
                val_log_data = {
                    "epoch": nb_iter,
                    "val_loss": val_stats['val_loss'],
                    "val_recon_loss": val_stats['val_recon_loss'],
                    "val_vel_loss": val_stats['val_vel_loss'],
                    "val_commit_loss": val_stats['val_commit_loss'],
                    "val_perplexity": val_stats['val_perplexity'],
                }
                if 'val_policy_loss' in val_stats:
                    val_log_data['val_policy_loss'] = val_stats['val_policy_loss']
                if 'val_latent_pred_loss' in val_stats:
                    val_log_data['val_latent_pred_loss'] = val_stats['val_latent_pred_loss']
                if 'val_motion_pred_loss' in val_stats:
                    val_log_data['val_motion_pred_loss'] = val_stats['val_motion_pred_loss']
                logger.log_stats(val_log_data, self.config['print_iter'])
                
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
        """Reconstruct *ground-truth* motion sequence via the trained VQ-VAE.

        Parameters
        ----------
        ground_truth : int
            Index of the motion sequence to reconstruct (0-based). This is the
            same indexing used in ``self.end_indices``.

        Returns
        -------
        recon : torch.Tensor
            (T, frame_size) tensor containing the reconstructed motion (CPU).
        """

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
