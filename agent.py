import copy
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# from protomotions.agents.common.common import update_linear_schedule
# from protomotions.envs.base_env.env import BaseEnv

from pathlib import Path
from typing import Optional, Dict, Tuple

# from lightning.fabric import Fabric
# from hydra.utils import instantiate, get_class
import wandb

# from protomotions.utils.time_report import TimeReport
# from protomotions.utils.average_meter import AverageMeter, TensorAverageMeterDict
# from protomotions.utils.motion_lib_mvq import MotionLibMVQ
# from protomotions.utils.motion_lib_mvq_h1g1 import MotionLibMVQ_h1g1
# from protomotions.utils.direct_motion_lib_mvq import DirectNpyMotionLib
# from protomotions.envs.base_env.env import BaseEnv

from protomotions.agents.mvq.models import MotionVQVAE
from protomotions.agents.mvq.vq_dataset import MotionVQDataset, cycle
from protomotions.agents.mvq.utils import cal_loss, cal_vel_loss

# from omegaconf import OmegaConf


# TODO
'''
To Dr. Donghoon:
    위의 동훈님이 사용하지 않으실 import들을 주석처리하였습니다 (기존에는 모두 이용)
    그리고 필수로 사용하셔야할 library들을 import 유지하였습니다.
    마지막 models, vq_dataset, utils를 제가 제공한 모델파일들의 path를 지정하셔서 동훈님 코드베이스에 맞게 import하면 좋을 것 같습니다.
'''


import logging
log = logging.getLogger(__name__)


class StatsLogger:
    def __init__(self, num_epochs, fabric: Fabric, use_wandb: bool = True):
        self.start = time.time()
        self.num_epochs = num_epochs
        self.progress_format = None
        self.fabric = fabric
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
        if self.fabric.global_rank == 0:
            ep = data["epoch"]
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
            
            # Log to fabric loggers (wandb, etc.)
            metrics = {
                "epoch": ep,
                "train/reconstruction_loss": ep_recon_loss,
                "train/velocity_loss": ep_vel_loss,
                "train/commitment_loss": ep_commit_loss,
                "train/perplexity": ep_perplexity,
                "train/total_loss": ep_recon_loss + ep_vel_loss + ep_commit_loss,
            }
            self.fabric.log_dict(metrics, step=ep)
            
            # Additional wandb logging if enabled
            if self.use_wandb and wandb.run is not None:
                wandb.log(metrics, step=ep)




class MVQVAEAgent:
    """Motion-VQ-VAE agent usable with (inference) or without (training) an Env."""

    def __init__(
        self,
        fabric: Fabric,
        env: Optional[BaseEnv] = None,          # ← Optional now
        config=None,
    ):
        self.fabric = fabric
        self.device: torch.device = fabric.device if fabric is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.config = config

        # ------------------------------------------------------------------
        # Motion-library                                         (24-joint SMPL)
        # ------------------------------------------------------------------
        if env is not None:                       # online use-case
            self.motion_lib = env.motion_lib

        else:                                     # offline training – build it here

            # Build RobotConfig from the YAML specified in agent config
            from protomotions.simulator.base_simulator.config import RobotConfig

            robot_cfg_omega = OmegaConf.load(config.robot_config_file)
            robot_dict = OmegaConf.to_container(robot_cfg_omega.robot, resolve=True)

            # ensure required field
            robot_dict.setdefault("asset", {}).setdefault("collapse_fixed_joints", False)
            robot_cfg = RobotConfig.from_dict(robot_dict)

            if config.use_motion_lib:
                MotionLibClass = get_class(config.motion_lib._target_)
                ml_args = {k: v for k, v in config.motion_lib.items() if k != "_target_"}

                key_body_names = robot_cfg.key_bodies
                key_body_ids = [robot_cfg.body_names.index(n) for n in key_body_names]

                self.motion_lib = MotionLibClass(
                    robot_config=robot_cfg,
                    key_body_ids=key_body_ids,
                    skeleton_tree=None,
                    **ml_args,
                )
            else: # for using direct motion file (npy)
                print(f"Using direct motion file: {config.direct_motion_file}")
                npy_path = config.direct_motion_file
                self.motion_lib = DirectNpyMotionLib(npy_path, robot_cfg, self.device)

        '''
        # TODO
        To Dr. Donghoon:
            말씀드렸던 바꿔야할 init(), 동훈님의 코드베이스에 맞춰서 위의 코드 수정 필요합니다.

            기존:
                motion_lib를 서로다른 로봇에 맞춰서 생성하기위한 코드입니다. 동훈님은 필요없을것으로 예상됩니다.
            
            Recommend:
                다 지우셔도 될정도로 코드 흐름과 무관. 
                동훈님의 코드베이스에 맞춰서 init이 필요합니다.
                추가로, omegaconfig, hydra, Febric 모듈들 또한 사용하지 않으시면 다 제거하시면 좋을 것 같아요.
        '''

        self.batch_size = config.batch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.total_iter = config.total_iter
        self.warmup_iter = config.warmup_iter
        self.lr_scheduler = config.lr_scheduler
        self.gamma = config.gamma

        # VQ-VAE model parameters
        self.num_joints = self.motion_lib.num_joints
        self.code_dim = config.code_dim
        self.nb_code = config.nb_code
        self.mu = config.mu
        self.down_t = config.down_t
        self.stride_t = config.stride_t
        self.width = config.width
        self.depth = config.depth
        self.dilation_growth_rate = config.dilation_growth_rate
        self.output_emb_width = config.output_emb_width
        self.vq_act = config.vq_act
        self.vq_norm = config.vq_norm

        # Loss parameters
        self.recons_loss = config.recons_loss
        self.loss_vel = config.loss_vel
        self.commit = config.commit
        self.quantizer = config.quantizer
        self.beta = config.beta

        # Data related parameters
        self.window_size = config.window_size
        self.num_workers = config.num_workers

        # Logging and saving parameters
        self.print_iter = config.print_iter
        self.save_every = config.save_every
        self.seed = config.seed

        # Wandb parameters
        self.use_wandb = config.use_wandb if env is None else False
        
        self._should_stop = False
        self.current_epoch = 0
        
        # Initialize logging
        self.time_report = TimeReport()
        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        # TODO:
        '''
        To Dr. Donghoon:
            logging 하실때, TimeReport,AverageMeter 가 필요하시면 추가로 코드드리겠습니다. 안이용하셔도 무방합니다.
        '''

    @property
    def should_stop(self):
        return self.fabric.broadcast(self._should_stop)

    def setup(self):
        """Setup model and optimizers following"""
        
        # Initialize wandb if enabled
        if self.use_wandb and self.fabric.global_rank == 0:
            if wandb.run is None:
                wandb.init(
                    project=getattr(self.config, 'wandb_project', 'motion-vq-vae'),
                    name=getattr(self.config, 'wandb_run_name', f'mvq-vae'),
                    config=dict(self.config),
                    tags=getattr(self.config, 'wandb_tags', ['motion-vq-vae']),
                )
        '''
        # TODO
        To Dr. Donghoon:
            기존: 모션 라이브러리를 이용해서 mocap data(amass kinematic information (dx,dy,dx,dyaw,p^j,v^j,r^j))추출
                mocap_data = amass kinematic data 
                    torch.cat(dim=0)함. 모든 motion data를 한 tensor에.. shape=[total_frames_of_motions, self.frame_size]
                self.frame_size = data column 개수
                self.end_indices = 각 모션들이 mocap_data에서 끝나는 index의 축적형; 
                    e.g, [500, 1000, ...] 첫번째 모션은 mocap_data에서 500번에서 끝남
            recommend: 
                동훈님의 모션라이브러리 motion_lib 혹은 모션class 생성후 알맞는 input삽입가능.
                end_indicies convention을 추천..!
        '''
        # Get motion data properties
        mocap_data = self.motion_lib.get_mvq_data()
        self.frame_size = self.motion_lib.FRAME_SIZE  # Use the frame size from MotionLibMVQ #number of dims
        
        # Get end indices for bad index computation
        self.end_indices = self.motion_lib.get_mvq_end_indices()
        
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
            window_size=self.window_size,
            mean=mean,
            std=std
        )

        # Dataset size is too small, so we need to reduce the batch size
        ## for now batching is less than batch_size
        ## so we need to pad the data
        self.batch_size = min(self.batch_size, len(dataset))

        self.train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=self.num_workers,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
        )
        self.train_loader_iter = cycle(self.train_loader)
        
        # Initialize VQVAE model
        self.model = MotionVQVAE(
            self, # self as args
            self.nb_code,
            self.code_dim,
            self.output_emb_width,
            self.down_t,
            self.stride_t,
            self.width,
            self.depth,
            self.dilation_growth_rate,
            self.vq_act,
            self.vq_norm
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_scheduler, gamma=self.gamma)

        # Setup with fabric
        if self.fabric is not None:
            self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        self.model.mark_forward_method('encode')
        self.model.mark_forward_method('forward_decoder')

        log.info(f"Setup complete. Model type: MotionVQVAE, Frame size: {self.frame_size}")


    def load(self, checkpoint: Optional[Path]):
        """Load checkpoint following ProtoMotions pattern."""
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
        """Save checkpoint following ProtoMotions pattern."""
        if path is None:
            path = self.fabric.loggers[0].log_dir
        save_dir = Path(path)
        state_dict = self.get_state_dict()
        self.fabric.save(save_dir / name, state_dict)
        
        if self.fabric.global_rank == 0:
            log.info(f"Saved checkpoint to {save_dir / name}")

    def update_lr_warm_up(self, optimizer, nb_iter, warm_up_iter, lr):
            current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            return optimizer, current_lr

    def fit(self):
        log.info("Starting Motion VQ-VAE training")

        logger = StatsLogger(self.total_iter, self.fabric, self.use_wandb)

        ##### Warm-up #####
        for nb_iter in range(1, self.warmup_iter + 1):

            self.optimizer, current_lr = self.update_lr_warm_up(self.optimizer, nb_iter, self.warmup_iter, self.lr)

            motion = next(self.train_loader_iter).to(self.device)

            pred_motion, loss_commit, perplexity = self.model(motion)
            loss_mot = cal_loss(motion, pred_motion, self)
            loss_vel = cal_vel_loss(motion, pred_motion, self)

            loss = loss_mot + self.commit * loss_commit + self.loss_vel * loss_vel

            self.optimizer.zero_grad()
            self.fabric.backward(loss)
            self.optimizer.step()

            logger.log_stats({
                "epoch": nb_iter,
                "ep_recon_loss": loss_mot.item(),
                "ep_vel_loss": loss_vel.item(),
                "ep_commit_loss": loss_commit.item(),
                "ep_perplexity": perplexity.item(),
            }, self.print_iter)

            log.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {loss_commit.item():.5f} \t PPL. {perplexity.item():.2f} \t Recons.  {loss_mot.item():.5f}")

        log.info("****Warmup completed****")

        ##### Training #####    
        for nb_iter in range(1, self.total_iter + 1):
            self.current_epoch = nb_iter
            # --- Data Preparation ---
            motion = next(self.train_loader_iter).to(self.device)

            # --- Forward Pass ---
            pred_motion, loss_commit, perplexity = self.model(motion)

            # --- Loss ---
            loss_mot = cal_loss(motion, pred_motion, self)
            loss_vel = cal_vel_loss(motion, pred_motion, self)
            
            loss = loss_mot + self.commit * loss_commit + self.loss_vel * loss_vel

            # --- Backward Pass & Optimization ---
            self.optimizer.zero_grad()
            self.fabric.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

            logger.log_stats({
                "epoch": nb_iter,
                "ep_recon_loss": loss_mot.item(),
                "ep_vel_loss": loss_vel.item(),
                "ep_commit_loss": loss_commit.item(),
                "ep_perplexity": perplexity.item(),
            }, self.print_iter)

            # --- Save Model ---
            if nb_iter % self.save_every == 0:
                self.save()
        
        # --- Save Final Model ---
        log.info("Training completed")
        if self.use_wandb and self.fabric.global_rank == 0 and wandb.run is not None:
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
            win = seq_norm[i : i + self.window_size]                 # [<=W, C]
            orig_len = win.shape[0]
            if orig_len < self.window_size:
                pad = win[-1:].repeat(self.window_size - orig_len, 1)
                win = torch.cat([win, pad], dim=0)
            windows.append(win)
            indices.append((i, orig_len))
            i += self.window_size

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
