
"""
specgan_trainer.py
A modular, class-based PyTorch trainer for SpecGAN-style spectrogram GANs.
Generated from a user's notebook to provide a cleaner, reusable interface.

Usage (example):
    from specgan_trainer import SpecGANTrainer
    trainer = SpecGANTrainer(data_root="/data/specs", moments_path="/data/moments.npy", out_dir="./out")
    trainer.build_models()        # builds or imports generator/discriminator
    trainer.prepare_data(batch_size=64)
    trainer.train(epochs=100, save_every=5)

Notes:
- This script expects a `specgan.py` file with `SpecGANGenerator` and `SpecGANDiscriminator`
  classes in the same directory or on PYTHONPATH. If not available, the trainer will raise
  a helpful ImportError when build_models() is called.
- The dataset loader here is generic: by default it looks for .npy files under `data_root`
  or a single moments .npy file (moments_path). Adjust SpectrogramDataset to match your data.
"""

from typing import Optional, Tuple, List, Dict, Any
import os
import time
import json
import random
import math
from dataclasses import dataclass, field
from dotenv import load_dotenv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from spectrogram_dataset import SpectrogramDataset

GREEN = '\033[32m'
RED = '\033[31m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'

# Attempt to import our models. If not present, exit
try:
    from specgan_models import SpecGANGenerator, SpecGANDiscriminator
except Exception:
    print(f"{RED}ERROR: User GAN models not found{RESET}")


@dataclass
class TrainerConfig:
    seed: int = 42
    data_root: Optional[str] = None
    moments_path: Optional[str] = None
    out_dir: str = "./output"
    nz: int = 100  # latent dim
    ngf: int = 64
    ndf: int = 64
    nc: int = 1  # number of channels in spectrograms
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.9
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    use_amp: bool = False  # mixed precision
    checkpoint_prefix: str = "specgan"
    sample_size: int = 16  # number of samples to save for visualization
    log_interval: int = 100


class SpecGANTrainer:
    """
    High-level trainer class for SpecGAN-style models.
    Exposes methods to build models, prepare data, and run training.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = config.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._set_seed(config.seed)
        os.makedirs(self.config.out_dir, exist_ok=True)

        # Placeholders to be set by build_models()
        self.netG: Optional[nn.Module] = None
        self.netD: Optional[nn.Module] = None
        self.optimG: Optional[optim.Optimizer] = None
        self.optimD: Optional[optim.Optimizer] = None
        self.criterion = nn.BCEWithLogitsLoss()

        # Dataset related
        self.dataset: Optional[Dataset] = None
        self.dataloader: Optional[DataLoader] = None

        # Statistic trackers
        self.G_losses: List[float] = []
        self.D_losses: List[float] = []
        self.fixed_noise = torch.randn(self.config.sample_size, self.config.nz, 1, 1, device=self.device)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def prepare_data(self, batch_size: int = 64, num_workers: int = 4, shuffle: bool = True):
        """Create dataset and dataloader based on config"""
        self.dataset = SpectrogramDataset(root_dir=self.config.data_root, moments_path=self.config.moments_path)
        # collate to ensure tensors are same shape: pad/crop to the first sample size if needed
        first = self.dataset[0]
        c, h, w = first.shape
        def collate_fn(batch):
            tensors = []
            for x in batch:
                # If shape mismatches, center-crop or pad to target HxW
                if x.shape[1] != h or x.shape[2] != w:
                    x_np = x.numpy()
                    # simple crop or pad implementation
                    pad_h = max(0, h - x_np.shape[1])
                    pad_w = max(0, w - x_np.shape[2])
                    # pad equally both sides
                    if pad_h > 0 or pad_w > 0:
                        x_np = np.pad(x_np, ((0,0),(pad_h//2, pad_h-pad_h//2),(pad_w//2, pad_w-pad_w//2)), mode='constant', constant_values=0)
                    else:
                        # crop center
                        start_h = (x_np.shape[1] - h)//2
                        start_w = (x_np.shape[2] - w)//2
                        x_np = x_np[:, start_h:start_h+h, start_w:start_w+w]
                    tensors.append(torch.from_numpy(x_np.copy()))
                else:
                    tensors.append(x)
            return torch.stack(tensors, dim=0)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fn
        )
        print(f"\n*\tData prepared: {len(self.dataset)} samples, batch_size={batch_size}")

    def build_models(self, overwrite: bool = False):
        """
        Build or import the generator and discriminator modules.
        Tries to import SpecGANGenerator/SpecGANDiscriminator from specgan.py if present.
        Otherwise, user should provide compatible nn.Module classes.
        """
        if self.netG is not None and not overwrite:
            print("Models already built. Set overwrite=True to rebuild.")
            return

        # Instantiate using config. The constructor signatures may vary depending on your specgan.py.
        # We'll attempt a few common patterns, else raise a helpful error.
        try:
            self.netG = SpecGANGenerator(self.config.nz, self.config.ngf, self.config.nc).to(self.device)
            self.netD = SpecGANDiscriminator(self.config.nc, self.config.ndf).to(self.device)
        except TypeError:
            # try alternate ordering
            try:
                self.netG = SpecGANGenerator(self.config.nz, self.config.nc, self.config.ngf).to(self.device)
                self.netD = SpecGANDiscriminator(self.config.nc, self.config.ndf).to(self.device)
            except Exception as e:
                raise RuntimeError("Failed to instantiate SpecGANGenerator/SpecGANDiscriminator. Check constructors in specgan.py") from e

        print("\n*\tModels instantiated and moved to device:", self.device)

        # optimizers
        self.optimG = optim.Adam(self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))
        self.optimD = optim.Adam(self.netD.parameters(), lr=self.config.lr, betas=(self.config.beta1, self.config.beta2))

    def save_checkpoint(self, epoch: int, kind: str = "latest"):
        """Save model + optimizer state to out_dir with a JSON metadata file"""
        meta = {
            "epoch": epoch,
            "seed": self.config.seed,
            "nz": self.config.nz,
            "ngf": self.config.ngf,
            "ndf": self.config.ndf,
            "nc": self.config.nc,
            "timestamp": time.time()
        }
        model_state = {
            "netG": self.netG.state_dict() if self.netG is not None else None,
            "netD": self.netD.state_dict() if self.netD is not None else None,
            "optimG": self.optimG.state_dict() if self.optimG is not None else None,
            "optimD": self.optimD.state_dict() if self.optimD is not None else None,
            "meta": meta
        }
        prefix = f"{self.config.checkpoint_prefix}_{kind}_epoch{epoch}"
        save_path = os.path.join(self.config.out_dir, prefix + ".pth")
        torch.save(model_state, save_path)
        with open(os.path.join(self.config.out_dir, prefix + ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        if self.netG is None or self.netD is None:
            raise RuntimeError("Models must be built before loading checkpoint.")
        if state.get("netG") is not None:
            self.netG.load_state_dict(state["netG"])
        if state.get("netD") is not None:
            self.netD.load_state_dict(state["netD"])
        if state.get("optimG") is not None and self.optimG is not None:
            self.optimG.load_state_dict(state["optimG"])
        if state.get("optimD") is not None and self.optimD is not None:
            self.optimD.load_state_dict(state["optimD"])
        print(f"Loaded checkpoint from {path}")


    def train(self, epochs: int = 50, batch_size: int = 16, d_updates: int = 1, g_updates: int = 1, save_every: int = 5):
        """
        Basic adversarial training loop. Keeps the structure general so it can be adapted
        to the loss variant you were using in the notebook.

        - d_updates: number of discriminator updates per batch (commonly 1)
        - g_updates: number of generator updates per loop (commonly 1)
        """
        if self.dataloader is None:
            self.prepare_data(batch_size=batch_size)

        if self.netG is None or self.netD is None:
            self.build_models()

        device = self.device
        criterion = self.criterion

        real_label = 1.0
        fake_label = 0.0

        scalerG = torch.cuda.amp.GradScaler() if self.config.use_amp else None
        scalerD = torch.cuda.amp.GradScaler() if self.config.use_amp else None

        start_epoch = 1
        for epoch in range(start_epoch, epochs + 1):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                real = data.to(device)
                b_size = real.size(0)

                # Update Discriminator
                for _ in range(d_updates):
                    self.netD.zero_grad()
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                    with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                        output_real = self.netD(real).view(-1)
                        lossD_real = criterion(output_real, label)

                        # Generate fake
                        noise = torch.randn(b_size, self.config.nz, 1, 1, device=device)
                        fake = self.netG(noise)
                        label.fill_(fake_label)
                        output_fake = self.netD(fake.detach()).view(-1)
                        lossD_fake = criterion(output_fake, label)
                        lossD = (lossD_real + lossD_fake) * 0.5

                    if scalerD is not None:
                        scalerD.scale(lossD).backward()
                        scalerD.step(self.optimD)
                        scalerD.update()
                    else:
                        lossD.backward()
                        self.optimD.step()

                # Update Generator
                for _ in range(g_updates):
                    self.netG.zero_grad()
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

                    with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                        # IMPORTANT: regenerate noise for generator step
                        noise_g = torch.randn(b_size, self.config.nz, 1, 1, device=device)

                        fake_g = self.netG(noise_g)
                        output = self.netD(fake_g).view(-1)

                        lossG = criterion(output, label)

                    if scalerG is not None:
                        scalerG.scale(lossG).backward()
                        scalerG.step(self.optimG)
                        scalerG.update()
                    else:
                        lossG.backward()
                        self.optimG.step()

                epoch_d_loss += lossD.item()
                epoch_g_loss += lossG.item()

                if (i + 1) % self.config.log_interval == 0:
                    print(f"[{epoch}/{epochs}] Batch {i+1}/{len(self.dataloader)}\tLoss_D: {lossD.item():.4f}\tLoss_G: {lossG.item():.4f}")

            # epoch averages
            self.D_losses.append(epoch_d_loss / max(1, len(self.dataloader)))
            self.G_losses.append(epoch_g_loss / max(1, len(self.dataloader)))
            print(f"Epoch {epoch} finished. Avg Loss_D: {self.D_losses[-1]:.4f}\tAvg Loss_G: {self.G_losses[-1]:.4f}")

            # save checkpoint and sample images
            if epoch % save_every == 0 or epoch == epochs:
                self.save_checkpoint(epoch, kind="latest")
                self._save_sample_images(epoch)

        # final save
        self.save_checkpoint(epochs, kind="best")
        self._save_metrics()

    def _save_sample_images(self, epoch: int):
        """Generate samples with fixed noise and save a grid image to out_dir"""
        if self.netG is None:
            return
        with torch.no_grad():
            self.netG.eval()
            fake = self.netG(self.fixed_noise).detach().cpu()
            # ensure fake is in [-1,1] then scale to [0,1] for saving
            grid = make_grid(fake, nrow=int(math.sqrt(self.fixed_noise.size(0))), normalize=True, value_range=(-1,1))
            save_path = os.path.join(self.config.out_dir, f"samples_epoch{epoch}.png")
            save_image(grid, save_path)
            print(f"Saved sample grid to {save_path}")
            self.netG.train()

    def _save_metrics(self):
        meta = {
            "G_losses": self.G_losses,
            "D_losses": self.D_losses
        }
        with open(os.path.join(self.config.out_dir, "training_metrics.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("Saved training metrics to training_metrics.json")


if __name__ == "__main__":
    # Example CLI usage to run training quickly (can read environment variables if given)
    load_dotenv()

    cfg = TrainerConfig(
        seed=int(os.getenv("SEED")),
        data_root=os.getenv("DATA_ROOT"),
        moments_path=os.getenv("MOMENTS_PATH"),
        out_dir=os.getenv("TRAINING_OUTPUT_ROOT"),
        nz=int(os.getenv("NZ", "100")),
        ngf=int(os.getenv("NGF", "64")),
        ndf=int(os.getenv("NDF", "64")),
        nc=int(os.getenv("NC", "1")),
        lr=float(os.getenv("LR", "0.0002")),
    )
    trainer = SpecGANTrainer(cfg)
    # If user runs the script directly, attempt to prepare data and run a short training
    trainer.prepare_data(batch_size=int(os.getenv("BATCH_SIZE")))
    trainer.build_models()
    trainer.train(epochs=int(os.getenv("EPOCHS")), batch_size=int(os.getenv("BATCH_SIZE")), save_every=1)
