"""
Training Utilities for SpecGAN-based Solar Radio Burst Generation

Ported from Chris Donahue's SpecGAN (TensorFlow) to PyTorch
Original: https://github.com/chrisdonahue/wavegan

Key components:
- PerFrequencyNormalizer: Per-frequency bin normalization (from moments() and t_to_f())
- GANLoss: Multiple GAN loss functions (DCGAN, LSGAN, WGAN, WGAN-GP)
- compute_gradient_penalty: WGAN-GP gradient penalty computation
- Checkpoint utilities: Model saving/loading (PyTorch adaptation)
"""

import os

import numpy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from glob import glob
from typing import Tuple


# ============================================================================
# 1. Per-Frequency Normalization
#    Ported from train_specgan.py: moments() and t_to_f() functions
# ============================================================================

class PerFrequencyNormalizer:
    """
    Per-frequency bin normalization for spectrograms (SpecGAN approach)
    
    Original SpecGAN code:
    - moments() function (train_specgan.py, Lines 575-614)
    - t_to_f() normalization (train_specgan.py, Lines 38-40)
    
    Key difference from global normalization:
    - Computes separate mean/std for each frequency bin
    - Preserves frequency-specific intensity characteristics
    """
    
    def __init__(self):
        """Initialize normalizer (moments will be computed or loaded later)"""
        self.mean_per_freq = None
        self.std_per_freq = None
        self._clip_nstd = 3.0  # SpecGAN default: _CLIP_NSTD = 3.0 (Line 24)
    
    def compute_moments(self, csv_files:str, verbose:bool=True)->Tuple[numpy.ndarray, numpy.ndarray]:
        """
        Compute mean and standard deviation for each frequency bin
        
        Ported from: train_specgan.py, moments() function (Lines 575-614)
        Original logic:
            _X_lmags = np.concatenate(_X_lmags, axis=0)
            mean, std = np.mean(_X_lmags, axis=0), np.std(_X_lmags, axis=0)
        
        Args:
            csv_files: List of CSV file paths
            verbose: Print progress information
        
        Returns:
            mean_per_freq: [n_freq] array (one mean per frequency bin)
            std_per_freq: [n_freq] array (one std per frequency bin)
        """
        if verbose:
            print(f"*\tComputing per-frequency moments from {len(csv_files)} files...")
            print(f" \tThis implements SpecGAN's moments() function")
        
        all_spectrograms = []
        
        for i, fp in enumerate(csv_files):
            if verbose and (i % 50 == 0) and len(csv_files) > 50:
                print(f" \tProgress: {i}/{len(csv_files)}")
            
            # Load spectrogram (CSV or NPY)
            if fp.endswith('.npy'):
                spec = np.load(fp)  # [128, 128]
            elif fp.endswith('.csv'):
                spec = pd.read_csv(fp, header=None).values  # [128, 128]
            else:
                continue
            
            # Optional: Apply log transform (SpecGAN uses log magnitude)
            # For radio bursts, may or may not be needed - test both
            # spec = np.log(spec + 1e-6)  # Uncomment if needed
            
            all_spectrograms.append(spec)
        
        # Stack all spectrograms: [N_samples, n_freq, n_time]
        all_spectrograms = np.stack(all_spectrograms, axis=0)
        
        if verbose:
            print(f" \tStacked data shape: {all_spectrograms.shape} (n, width, height)\n")
        
        # Compute per-frequency statistics
        # Original SpecGAN: mean, std = np.mean(_X_lmags, axis=0), np.std(_X_lmags, axis=0)
        # axis=0: average over all samples
        # axis=2: average over all time steps
        # Result: one mean/std value per frequency bin
        self.mean_per_freq = np.mean(all_spectrograms, axis=(0, 2))  # [128]
        self.std_per_freq = np.std(all_spectrograms, axis=(0, 2))    # [128]
        
        if verbose:
            print(f"*\tPer-frequency moments computed:")
            print(f" \tMean shape: {self.mean_per_freq.shape[0]}")
            print(f" \tStd shape: {self.std_per_freq.shape[0]}")
            print(f" \tMean range: [{self.mean_per_freq.min():.2f}, {self.mean_per_freq.max():.2f}]")
            print(f" \tStd range: [{self.std_per_freq.min():.2f}, {self.std_per_freq.max():.2f}]\n")
        return self.mean_per_freq, self.std_per_freq
    
    def normalize(self, spectrogram:numpy.ndarray)->numpy.ndarray:
        """
        Apply per-frequency normalization to spectrogram
        
        Ported from: train_specgan.py, t_to_f() function (Lines 38-40)
        Original code:
            X_norm = (X_lmag - X_mean[:-1]) / X_std[:-1]
            X_norm /= _CLIP_NSTD
            X_norm = tf.clip_by_value(X_norm, -1., 1.)
        
        Args:
            spectrogram: [n_freq, n_time] numpy array (e.g., [128, 128])
        
        Returns:
            normalized: [n_freq, n_time] numpy array in range [-1, 1]
        """
        if self.mean_per_freq is None or self.std_per_freq is None:
            raise ValueError("Moments not computed! Call compute_moments() or load_moments() first.")
        
        # Expand dimensions for broadcasting: [128] → [128, 1]
        mean = self.mean_per_freq[:, np.newaxis]
        std = self.std_per_freq[:, np.newaxis]
        
        # Per-frequency standardization (SpecGAN approach)
        # Original: X_norm = (X_lmag - X_mean[:-1]) / X_std[:-1]
        normalized = (spectrogram - mean) / (std + 1e-8)
        
        # Clip to N standard deviations (SpecGAN uses 3)
        # Original: X_norm /= _CLIP_NSTD
        normalized /= self._clip_nstd
        
        # Final clipping to [-1, 1]
        # Original: X_norm = tf.clip_by_value(X_norm, -1., 1.)
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized.astype(np.float32)
    
    def denormalize(self, normalized:numpy.ndarray)->numpy.ndarray:
        """
        Reverse normalization to get original scale
        
        Inverse of normalize() - not in original SpecGAN but useful for visualization
        
        Args:
            normalized: [n_freq, n_time] array in [-1, 1]
        
        Returns:
            spectrogram: [n_freq, n_time] array in original scale
        """
        if self.mean_per_freq is None or self.std_per_freq is None:
            raise ValueError("Moments not computed!")
        
        mean = self.mean_per_freq[:, np.newaxis]
        std = self.std_per_freq[:, np.newaxis]
        
        # Reverse the operations
        spectrogram = normalized * self._clip_nstd  # Undo /= 3.0
        spectrogram = spectrogram * std + mean      # Undo standardization
        
        return spectrogram.astype(np.float32)
    
    def save_moments(self, filepath:str, verbose:bool=True)->None:
        """
        Save computed moments to file
        
        Ported from: train_specgan.py, Lines 612-613
        Original: pickle.dump((mean, std), f)
        
        Args:
            filepath: Path to save moments (will use .npz format)
        """
        if self.mean_per_freq is None or self.std_per_freq is None:
            raise ValueError("No moments to save! Call compute_moments() first.")

        np.savez(filepath,
                 mean=self.mean_per_freq, 
                 std=self.std_per_freq,
                 clip_nstd=self._clip_nstd)

        if verbose:
            print(f"*\tMoments saved to {filepath}")
    
    def load_moments(self, filepath:str)->None:
        """
        Load pre-computed moments from file
        
        Ported from: train_specgan.py, Lines 726-729
        Original: _mean, _std = pickle.load(f)
        
        Args:
            filepath: Path to moments file (.npz)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Moments file not found: {filepath}")
        
        data = np.load(filepath)
        self.mean_per_freq = data['mean']
        self.std_per_freq = data['std']
        
        if 'clip_nstd' in data:
            self._clip_nstd = float(data['clip_nstd'])
        
        print(f"*\tMoments loaded from {filepath}")
        print(f" \tMean shape: {self.mean_per_freq.shape}")
        print(f" \tStd shape: {self.std_per_freq.shape}")


# ============================================================================
# 2. GAN Loss Functions
#    Ported from train_specgan.py: Lines 183-271
# ============================================================================

class GANLoss:
    """
    Multiple GAN loss function implementations
    
    Ported from: train_specgan.py, Lines 183-271
    
    Supported losses:
    - dcgan: Standard GAN with sigmoid cross-entropy (Lines 183-201)
    - lsgan: Least Squares GAN (Lines 202-206)
    - wgan: Wasserstein GAN with weight clipping (Lines 207-221)
    - wgan-gp: WGAN with Gradient Penalty (Lines 222-236) - RECOMMENDED
    """
    
    @staticmethod
    def dcgan_loss(D_real, D_fake):
        """
        Standard DCGAN loss with BCE
        
        Ported from: Lines 183-201
        Original TensorFlow code:
            G_loss = sigmoid_cross_entropy_with_logits(D_G_z, real)
            D_loss = (sigmoid_cross_entropy(D_G_z, fake) + 
                      sigmoid_cross_entropy(D_x, real)) / 2
        
        Args:
            D_real: Discriminator output on real samples
            D_fake: Discriminator output on fake samples
        
        Returns:
            (G_loss, D_loss): Tuple of generator and discriminator losses
        """
        criterion = nn.BCEWithLogitsLoss()
        
        real_labels = torch.ones_like(D_real)
        fake_labels = torch.zeros_like(D_fake)
        
        # Generator loss: fool discriminator
        G_loss = criterion(D_fake, real_labels)
        
        # Discriminator loss: classify real as real, fake as fake
        D_loss_real = criterion(D_real, real_labels)
        D_loss_fake = criterion(D_fake, fake_labels)
        D_loss = (D_loss_real + D_loss_fake) / 2.0
        
        return G_loss, D_loss
    
    @staticmethod
    def lsgan_loss(D_real, D_fake):
        """
        Least Squares GAN loss
        
        Ported from: Lines 202-206
        Original TensorFlow code:
            G_loss = mean((D_G_z - 1.) ** 2)
            D_loss = (mean((D_x - 1.) ** 2) + mean(D_G_z ** 2)) / 2
        
        Args:
            D_real: Discriminator output on real samples
            D_fake: Discriminator output on fake samples
        
        Returns:
            (G_loss, D_loss): Tuple of losses
        """
        # Generator loss: make D(fake) close to 1
        G_loss = torch.mean((D_fake - 1.) ** 2)
        
        # Discriminator loss: D(real) close to 1, D(fake) close to 0
        D_loss_real = torch.mean((D_real - 1.) ** 2)
        D_loss_fake = torch.mean(D_fake ** 2)
        D_loss = (D_loss_real + D_loss_fake) / 2.0
        
        return G_loss, D_loss
    
    @staticmethod
    def wgan_loss(D_real, D_fake):
        """
        Wasserstein GAN loss (without gradient penalty)
        
        Ported from: Lines 207-221
        Original TensorFlow code:
            G_loss = -mean(D_G_z)
            D_loss = mean(D_G_z) - mean(D_x)
        
        Note: Requires weight clipping in discriminator (handled in training loop)
        
        Args:
            D_real: Discriminator output on real samples
            D_fake: Discriminator output on fake samples
        
        Returns:
            (G_loss, D_loss): Tuple of losses
        """
        # Generator loss: maximize D(fake)
        G_loss = -torch.mean(D_fake)
        
        # Discriminator loss: maximize D(real) - D(fake)
        D_loss = torch.mean(D_fake) - torch.mean(D_real)
        
        return G_loss, D_loss
    
    @staticmethod
    def wgan_gp_loss(D_real, D_fake, netD, real_data, fake_data, device, lambda_gp=10):
        """
        Wasserstein GAN with Gradient Penalty (RECOMMENDED for stable training)
        
        Ported from: Lines 222-236
        Original TensorFlow code:
            G_loss = -tf.reduce_mean(D_G_z)
            D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
            D_loss += LAMBDA * gradient_penalty
        
        SpecGAN default: lambda_gp = 10 (Line 232)
        
        Args:
            D_real: Discriminator output on real samples
            D_fake: Discriminator output on fake samples  
            netD: Discriminator network
            real_data: Real data tensor [N, C, H, W]
            fake_data: Fake data tensor [N, C, H, W]
            device: torch device
            lambda_gp: Gradient penalty coefficient (default: 10)
        
        Returns:
            (G_loss, D_loss): Tuple of losses
        """
        # Wasserstein loss
        # Original: G_loss = -tf.reduce_mean(D_G_z)
        G_loss = -torch.mean(D_fake)
        
        # Original: D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)
        D_loss = torch.mean(D_fake) - torch.mean(D_real)
        
        # Compute gradient penalty
        gp = compute_gradient_penalty(netD, real_data, fake_data, device)
        
        # Add gradient penalty to discriminator loss
        # Original: D_loss += LAMBDA * gradient_penalty (Line 236)
        D_loss += lambda_gp * gp
        
        return G_loss, D_loss


def compute_gradient_penalty(netD, real_data, fake_data, device):
    """
    Compute gradient penalty for WGAN-GP
    
    Ported from: train_specgan.py, Lines 226-236
    Original TensorFlow code:
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
        differences = G_z - x
        interpolates = x + (alpha * differences)
        D_interp = SpecGANDiscriminator(interpolates)
        gradients = tf.gradients(D_interp, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    
    Args:
        netD: Discriminator network
        real_data: Real samples [N, C, H, W]
        fake_data: Fake samples [N, C, H, W]
        device: torch device
    
    Returns:
        gradient_penalty: Scalar tensor
    """
    batch_size = real_data.size(0)
    
    # Random interpolation coefficient
    # Original: alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolated samples
    # Original: interpolates = x + (alpha * differences), where differences = G_z - x
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    
    # Discriminator output on interpolates
    # Original: D_interp = SpecGANDiscriminator(interpolates)
    D_interp = netD(interpolates)
    
    # Compute gradients of D w.r.t. interpolates
    # Original: gradients = tf.gradients(D_interp, [interpolates])[0]
    gradients = torch.autograd.grad(
        outputs=D_interp,
        inputs=interpolates,
        grad_outputs=torch.ones_like(D_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient norm
    # Original: slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    # Note: SpecGAN reduces over [1, 2] because gradients shape is [N, H, W, C]
    # In PyTorch [N, C, H, W], we flatten all except batch dimension
    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    
    # Gradient penalty: penalize deviation from norm=1
    # Original: gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    gradient_penalty = torch.mean((gradient_norm - 1.) ** 2)
    
    return gradient_penalty


# ============================================================================
# 3. Checkpoint Management (PyTorch Adaptation)
#    TensorFlow uses MonitoredTrainingSession (auto-saves)
#    PyTorch requires manual checkpoint management
# ============================================================================

def save_gan_checkpoint(netG, netD, optimizerG, optimizerD, epoch, quality_metric, checkpoint_dir, hyperparams=None):
    """
    Save GAN checkpoint with both Generator and Discriminator
    
    PyTorch adaptation (TensorFlow has this built-in via MonitoredTrainingSession)
    Reference: train_specgan.py uses tf.train.MonitoredTrainingSession (Lines 279-282)
    
    Args:
        netG: Generator network
        netD: Discriminator network
        optimizerG: Generator optimizer
        optimizerD: Discriminator optimizer
        epoch: Current epoch number
        quality_metric: Quality metric value (e.g., D(G(z)) or FID)
        checkpoint_dir: Directory to save checkpoints
        hyperparams: Optional dict of hyperparameters to save
    
    Returns:
        checkpoint_path: Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Filename with epoch and metric
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"checkpoint_epoch_{epoch}_quality_{quality_metric:.4f}.pth"
    )
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'quality_metric': quality_metric,
    }
    
    # Add hyperparameters if provided
    if hyperparams:
        checkpoint.update(hyperparams)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    print(f"*\tCheckpoint saved: epoch {epoch}, quality {quality_metric:.4f}")
    print(f" \tPath: {checkpoint_path}")
    
    return checkpoint_path


def load_gan_checkpoint(checkpoint_path, netG, netD, optimizerG=None, optimizerD=None, device='cpu'):
    """
    Load GAN checkpoint
    
    PyTorch adaptation (TensorFlow uses tf.train.Saver)
    
    Args:
        checkpoint_path: Path to checkpoint file
        netG: Generator network (will load state_dict into this)
        netD: Discriminator network (will load state_dict into this)
        optimizerG: Generator optimizer (optional, for resuming training)
        optimizerD: Discriminator optimizer (optional, for resuming training)
        device: Device to load to
    
    Returns:
        checkpoint: Full checkpoint dictionary (includes epoch, metric, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    # PyTorch 2.6+ requires weights_only=False for full checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model states
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Load optimizer states if provided (for resuming training)
    if optimizerG is not None and 'optimizerG_state_dict' in checkpoint:
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    
    if optimizerD is not None and 'optimizerD_state_dict' in checkpoint:
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metric = checkpoint.get('quality_metric', 0.0)
    
    print(f"📂 Checkpoint loaded: epoch {epoch}, quality {metric:.4f}")
    print(f"   From: {checkpoint_path}")
    
    return checkpoint


def load_generator_only(checkpoint_path, netG, device='cpu'):
    """
    Load only the generator (for inference/generation)
    
    Useful when you only need to generate samples, not continue training
    
    Args:
        checkpoint_path: Path to checkpoint file
        netG: Generator network
        device: Device to load to
    
    Returns:
        epoch: Epoch number of loaded checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    netG.load_state_dict(checkpoint['generator_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"*\tGenerator loaded from epoch {epoch}")
    
    return epoch


def find_best_checkpoint(checkpoint_dir, metric='quality'):
    """
    Find the best checkpoint in directory based on metric in filename
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        metric: Metric to search for (default: 'quality')
    
    Returns:
        best_checkpoint_path: Path to the best checkpoint (highest metric value)
    """
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    
    # Find all checkpoint files
    checkpoint_files = glob(os.path.join(checkpoint_dir, f'checkpoint_*_{metric}_*.pth'))
    
    if len(checkpoint_files) == 0:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    # Extract metric values from filenames
    best_metric = -float('inf')
    best_path = None
    
    for path in checkpoint_files:
        try:
            # Extract metric value from filename: checkpoint_epoch_X_quality_Y.pth
            filename = os.path.basename(path)
            metric_str = filename.split(f'{metric}_')[1].split('.pth')[0]
            metric_val = float(metric_str)
            
            if metric_val > best_metric:
                best_metric = metric_val
                best_path = path
        except:
            continue
    
    if best_path:
        print(f"*\tBest checkpoint found: {os.path.basename(best_path)}")
        print(f"   Metric value: {best_metric:.4f}")
        return best_path
    else:
        raise ValueError("Could not parse metric values from checkpoint filenames")


# ============================================================================
# 4. Weight Clipping for WGAN (if needed)
#    Ported from train_specgan.py, Lines 211-221
# ============================================================================

def clip_discriminator_weights(netD, clip_value=0.01):
    """
    Clip discriminator weights for WGAN (not WGAN-GP)
    
    Ported from: train_specgan.py, Lines 211-221
    Original TensorFlow code:
        clip_ops = []
        for var in D_vars:
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)))
    
    Note: Only needed for vanilla WGAN, NOT for WGAN-GP
    
    Args:
        netD: Discriminator network
        clip_value: Clipping bound (default: 0.01, same as SpecGAN)
    """
    for param in netD.parameters():
        param.data.clamp_(-clip_value, clip_value)


# ============================================================================
# 5. SpecGAN Default Hyperparameters
#    From train_specgan.py, Lines 687-712
# ============================================================================

SPECGAN_DEFAULTS = {
    # Model architecture
    'latent_dim': 100,           # Line 697: specgan_latent_dim=100
    'kernel_len': 5,             # Line 698: specgan_kernel_len=5
    'dim': 64,                   # Line 699: specgan_dim=64
    'use_batchnorm': False,      # Line 700: specgan_batchnorm=False
    
    # Training
    'disc_nupdates': 5,          # Line 701: specgan_disc_nupdates=5
    'loss_type': 'wgan-gp',      # Line 702: specgan_loss='wgan-gp'
    'batch_size': 64,            # Line 705: train_batch_size=64
    
    # Optimizer (for WGAN-GP, Lines 261-269)
    'lr': 1e-4,                  # learning_rate=1e-4
    'beta1': 0.5,                # beta1=0.5
    'beta2': 0.9,                # beta2=0.9
    
    # For other losses (DCGAN, Lines 244-250)
    'lr_dcgan': 2e-4,            # learning_rate=2e-4
    'beta1_dcgan': 0.5,          # beta1=0.5
    
    # Normalization
    'clip_nstd': 3.0,            # Line 24: _CLIP_NSTD = 3.0
    'log_eps': 1e-6,             # Line 25: _LOG_EPS = 1e-6
}


def get_specgan_optimizer(netG, netD, loss_type='wgan-gp'):
    """
    Get recommended optimizers for SpecGAN based on loss type
    
    Ported from: train_specgan.py, Lines 243-271
    
    Args:
        netG: Generator network
        netD: Discriminator network
        loss_type: 'dcgan', 'lsgan', 'wgan', or 'wgan-gp'
    
    Returns:
        (optimizerG, optimizerD): Tuple of optimizers
    """
    if loss_type == 'dcgan':
        # Lines 244-250
        optimizerG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    elif loss_type == 'lsgan':
        # Lines 251-255
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=1e-4)
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=1e-4)
    
    elif loss_type == 'wgan':
        # Lines 256-260
        optimizerG = torch.optim.RMSprop(netG.parameters(), lr=5e-5)
        optimizerD = torch.optim.RMSprop(netD.parameters(), lr=5e-5)
    
    elif loss_type == 'wgan-gp':
        # Lines 261-269 (SpecGAN default)
        optimizerG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return optimizerG, optimizerD


# ============================================================================
# 6. Utility Functions
# ============================================================================

def compute_csv_moments(csv_dir, output_path='moments.npz', pattern='window_*.csv', verbose=True):
    """
    Convenience function to compute moments from a directory of spectrogram files
    
    Wrapper around PerFrequencyNormalizer.compute_moments()
    Supports both CSV and NPY files.
    
    Args:
        csv_dir: Directory containing spectrogram files (CSV or NPY)
        output_path: Where to save computed moments
        pattern: File pattern to match (default: 'window_*.csv')
        verbose: Print progress
    
    Returns:
        normalizer: PerFrequencyNormalizer instance with computed moments
    """
    # Find all spectrogram files (CSV or NPY)
    csv_files = []
    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            # Support CSV files
            if file.startswith('window_') and file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
            # Support NPY files (e.g., eCallisto processed data)
            elif file.startswith('burst-') and file.endswith('.npy'):
                csv_files.append(os.path.join(root, file))
    
    if len(csv_files) == 0:
        raise ValueError(f"No spectrogram files found in {csv_dir}. Searched for 'window_*.csv' or 'burst-*.npy'")
    
    if verbose:
        csv_count = sum(1 for f in csv_files if f.endswith('.csv'))
        npy_count = sum(1 for f in csv_files if f.endswith('.npy'))
        print(f"Found {len(csv_files)} files (CSV: {csv_count}, NPY: {npy_count})")
    
    # Compute moments
    normalizer = PerFrequencyNormalizer()
    normalizer.compute_moments(csv_files, verbose=verbose)
    
    # Save moments
    normalizer.save_moments(output_path, verbose=verbose)
    
    return normalizer


if __name__ == '__main__':
    # Example usage for computing moments
    print("=" * 70)
    print("SpecGAN Utils - Compute Moments Example")
    print("=" * 70)
    print("Usage:")
    print("\tfrom specgan_utils import compute_csv_moments")
    print("\tnormalizer = compute_csv_moments('path/to/csv/dir', 'moments.npz')")
    print("\nThis will:")
    print("\t1. Find all window_*.csv files in directory")
    print("\t2. Compute per-frequency mean and std")
    print("\t3. Save to moments.npz")
    print("\t4. Return normalizer instance")
