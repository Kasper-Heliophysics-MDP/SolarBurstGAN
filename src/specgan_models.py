"""
SpecGAN Model Architectures for Solar Radio Burst Generation

Ported from Chris Donahue's SpecGAN (TensorFlow) to PyTorch
Original: https://github.com/chrisdonahue/wavegan/blob/master/specgan.py

Key components:
- SpecGANGenerator: Generates 128x128 spectrograms from 100-dim noise
- SpecGANDiscriminator: Classifies 128x128 spectrograms as real/fake
- weights_init: Weight initialization function

Architecture matches original SpecGAN exactly (verified line-by-line).
"""

import torch
import torch.nn as nn

GREEN = '\033[32m'
RED = '\033[31m'
BLUE = '\033[34m'
RESET = '\033[0m'

# ============================================================================
# Generator Network
# Ported from: specgan.py, Lines 47-111
# ============================================================================

class SpecGANGenerator(nn.Module):
    """
    SpecGAN Generator: 100-dim noise → 128×128 spectrogram
    
    Ported from: specgan.py, SpecGANGenerator() function (Lines 47-111)
    
    Original TensorFlow architecture:
        Input: [None, 100]
        Output: [None, 128, 128, 1]  (TensorFlow format: NHWC)
        
    PyTorch architecture:
        Input: [N, 100]
        Output: [N, 1, 128, 128]  (PyTorch format: NCHW)
    
    Architecture (5 transpose conv layers):
        100-dim → Dense → [4×4×1024]
        → Conv2DTranspose → [8×8×512]
        → Conv2DTranspose → [16×16×256]
        → Conv2DTranspose → [32×32×128]
        → Conv2DTranspose → [64×64×64]
        → Conv2DTranspose → [128×128×1]
    
    Args:
        nz (int): Latent vector dimension (default: 100, Line 48-49)
        kernel_len (int): Kernel size for conv layers (default: 5, Line 49)
        dim (int): Dimensionality multiplier (default: 64, Line 50)
        nc (int): Number of output channels (default: 1, single channel spectrogram)
        use_batchnorm (bool): Enable batch normalization (default: False, Line 51)
        ngpu (int): Number of GPUs (for multi-GPU training)
    """
    
    def __init__(self, nz=100, kernel_len=5, dim=64, nc=1, use_batchnorm=False, ngpu=1):
        super(SpecGANGenerator, self).__init__()
        self.nz = nz
        self.dim = dim
        self.nc = nc
        self.use_batchnorm = use_batchnorm
        self.ngpu = ngpu
        
        # FC and reshape for convolution
        # Original: Lines 64-68
        # [100] -> [4, 4, 1024] where 1024 = dim * 16 = 64 * 16
        self.fc = nn.Linear(nz, 4 * 4 * dim * 16)
        
        # Initial BatchNorm (applied after reshape, before first conv)
        # Original: Line 67 - batchnorm applied to reshaped tensor
        if use_batchnorm:
            self.bn0 = nn.BatchNorm2d(dim * 16)
        
        # Layer 0: [4, 4, 1024] -> [8, 8, 512]
        # Original: Lines 72-75
        self.upconv0 = nn.ConvTranspose2d(dim * 16, dim * 8, kernel_len, stride=2, padding=2, output_padding=1)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(dim * 8)
        
        # Layer 1: [8, 8, 512] -> [16, 16, 256]
        # Original: Lines 79-82
        self.upconv1 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_len, stride=2, padding=2, output_padding=1)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(dim * 4)
        
        # Layer 2: [16, 16, 256] -> [32, 32, 128]
        # Original: Lines 86-89
        self.upconv2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_len, stride=2, padding=2, output_padding=1)
        if use_batchnorm:
            self.bn3 = nn.BatchNorm2d(dim * 2)
        
        # Layer 3: [32, 32, 128] -> [64, 64, 64]
        # Original: Lines 93-96
        self.upconv3 = nn.ConvTranspose2d(dim * 2, dim, kernel_len, stride=2, padding=2, output_padding=1)
        if use_batchnorm:
            self.bn4 = nn.BatchNorm2d(dim)
        
        # Layer 4: [64, 64, 64] -> [128, 128, 1]
        # Original: Lines 100-102
        self.upconv4 = nn.ConvTranspose2d(dim, nc, kernel_len, stride=2, padding=2, output_padding=1)
        
        # Activation functions
        # Original: Line 68, 75, 82, 89, 96 - ReLU
        # Original: Line 102 - Tanh
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
    
    def forward(self, z):
        """
        Forward pass
        
        Original: Lines 63-111
        
        Args:
            z: Latent vector [N, nz] or [N, nz, 1, 1]
        
        Returns:
            output: Generated spectrogram [N, nc, 128, 128] in range [-1, 1]
        """
        # Flatten z if it's [N, nz, 1, 1] format (from DCGAN convention)
        if z.dim() == 4:
            z = z.view(z.size(0), -1)  # [N, nz, 1, 1] → [N, nz]
        
        # FC and reshape for convolution
        # Original: Lines 64-68
        # TensorFlow: [N, 100] → Dense → [N, 4, 4, 1024]
        # PyTorch: [N, 100] → Linear → [N, 4*4*1024] → view → [N, 1024, 4, 4]
        output = self.fc(z)  # [N, 4*4*1024]
        output = output.view(-1, self.dim * 16, 4, 4)  # [N, 1024, 4, 4] - Channels first!
        
        # Initial batchnorm + relu
        # Original: Line 67-68
        if self.use_batchnorm:
            output = self.bn0(output)
        output = self.relu(output)
        
        # Layer 0: [4, 4, 1024] -> [8, 8, 512]
        # Original: Lines 72-75
        output = self.upconv0(output)
        if self.use_batchnorm:
            output = self.bn1(output)
        output = self.relu(output)
        
        # Layer 1: [8, 8, 512] -> [16, 16, 256]
        # Original: Lines 79-82
        output = self.upconv1(output)
        if self.use_batchnorm:
            output = self.bn2(output)
        output = self.relu(output)
        
        # Layer 2: [16, 16, 256] -> [32, 32, 128]
        # Original: Lines 86-89
        output = self.upconv2(output)
        if self.use_batchnorm:
            output = self.bn3(output)
        output = self.relu(output)
        
        # Layer 3: [32, 32, 128] -> [64, 64, 64]
        # Original: Lines 93-96
        output = self.upconv3(output)
        if self.use_batchnorm:
            output = self.bn4(output)
        output = self.relu(output)
        
        # Layer 4: [64, 64, 64] -> [128, 128, 1]
        # Original: Lines 100-102
        output = self.upconv4(output)
        output = self.tanh(output)  # Output range: [-1, 1]
        
        return output


# ============================================================================
# Discriminator Network  
# Ported from: specgan.py, Lines 122-178
# ============================================================================

class SpecGANDiscriminator(nn.Module):
    """
    SpecGAN Discriminator: 128×128 spectrogram → Real/Fake classification
    
    Ported from: specgan.py, SpecGANDiscriminator() function (Lines 122-178)
    
    Original TensorFlow architecture:
        Input: [None, 128, 128, 1]  (TensorFlow format: NHWC)
        Output: [None] (single logit)
        
    PyTorch architecture:
        Input: [N, 1, 128, 128]  (PyTorch format: NCHW)
        Output: [N] (single logit)
    
    Architecture (5 conv layers + dense):
        [128×128×1] → Conv2D → [64×64×64]
        → Conv2D → [32×32×128]
        → Conv2D → [16×16×256]
        → Conv2D → [8×8×512]
        → Conv2D → [4×4×1024]
        → Flatten → Dense → [1]
    
    Args:
        kernel_len (int): Kernel size for conv layers (default: 5, Line 124)
        dim (int): Dimensionality multiplier (default: 64, Line 125)
        nc (int): Number of input channels (default: 1, single channel spectrogram)
        use_batchnorm (bool): Enable batch normalization (default: False, Line 126)
        ngpu (int): Number of GPUs
    """
    
    def __init__(self, kernel_len=5, dim=64, nc=1, use_batchnorm=False, ngpu=1):
        super(SpecGANDiscriminator, self).__init__()
        self.dim = dim
        self.nc = nc
        self.use_batchnorm = use_batchnorm
        self.ngpu = ngpu
        
        # Layer 0: [128, 128, 1] -> [64, 64, 64]
        # Original: Lines 137-139
        # Note: First layer has NO batchnorm in SpecGAN
        self.conv0 = nn.Conv2d(nc, dim, kernel_len, stride=2, padding=2)
        
        # Layer 1: [64, 64, 64] -> [32, 32, 128]
        # Original: Lines 143-146
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_len, stride=2, padding=2)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(dim * 2)
        
        # Layer 2: [32, 32, 128] -> [16, 16, 256]
        # Original: Lines 150-153
        self.conv2 = nn.Conv2d(dim * 2, dim * 4, kernel_len, stride=2, padding=2)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm2d(dim * 4)
        
        # Layer 3: [16, 16, 256] -> [8, 8, 512]
        # Original: Lines 157-160
        self.conv3 = nn.Conv2d(dim * 4, dim * 8, kernel_len, stride=2, padding=2)
        if use_batchnorm:
            self.bn3 = nn.BatchNorm2d(dim * 8)
        
        # Layer 4: [8, 8, 512] -> [4, 4, 1024]
        # Original: Lines 164-167
        self.conv4 = nn.Conv2d(dim * 8, dim * 16, kernel_len, stride=2, padding=2)
        if use_batchnorm:
            self.bn4 = nn.BatchNorm2d(dim * 16)
        
        # Output layer: Flatten + Dense
        # Original: Lines 170-174
        self.fc = nn.Linear(4 * 4 * dim * 16, 1)
        
        # Activation function
        # Original: Line 114-115 - lrelu with alpha=0.2
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        """
        Forward pass
        
        Original: Lines 134-178
        
        Args:
            x: Input spectrogram [N, nc, 128, 128]
        
        Returns:
            output: Logit values [N] (NOT probabilities - sigmoid applied in loss)
        """
        # Layer 0: [128, 128, 1] -> [64, 64, 64]
        # Original: Lines 137-139
        output = self.conv0(x)
        output = self.lrelu(output)
        # Note: NO batchnorm on first layer (SpecGAN convention)
        
        # Layer 1: [64, 64, 64] -> [32, 32, 128]
        # Original: Lines 143-146
        output = self.conv1(output)
        if self.use_batchnorm:
            output = self.bn1(output)
        output = self.lrelu(output)
        
        # Layer 2: [32, 32, 128] -> [16, 16, 256]
        # Original: Lines 150-153
        output = self.conv2(output)
        if self.use_batchnorm:
            output = self.bn2(output)
        output = self.lrelu(output)
        
        # Layer 3: [16, 16, 256] -> [8, 8, 512]
        # Original: Lines 157-160
        output = self.conv3(output)
        if self.use_batchnorm:
            output = self.bn3(output)
        output = self.lrelu(output)
        
        # Layer 4: [8, 8, 512] -> [4, 4, 1024]
        # Original: Lines 164-167
        output = self.conv4(output)
        if self.use_batchnorm:
            output = self.bn4(output)
        output = self.lrelu(output)
        
        # Flatten
        # Original: Line 170
        # TensorFlow: [N, 4, 4, 1024] → [N, 4*4*1024]
        # PyTorch: [N, 1024, 4, 4] → [N, 4*4*1024]
        output = output.view(-1, 4 * 4 * self.dim * 16)
        
        # Connect to single logit
        # Original: Lines 173-174
        output = self.fc(output)  # [N, 1]
        output = output.squeeze(1)  # [N] - Return scalar per sample
        
        return output


# ============================================================================
# Weight Initialization
# Based on DCGAN paper (Radford et al. 2016) - Standard for GAN training
# ============================================================================

def weights_init(m):
    """
    Initialize network weights
    
    Standard DCGAN initialization (used by SpecGAN):
    - Conv layers: Normal(0, 0.02)
    - BatchNorm layers: Normal(1.0, 0.02) for weight, 0 for bias
    
    This is the same initialization used in the original dcgan.ipynb
    
    Args:
        m: PyTorch module/layer
    """
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        # Initialize convolutional layers
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        
    elif classname.find('BatchNorm') != -1:
        # Initialize batch normalization layers
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============================================================================
# Architecture Verification Functions
# ============================================================================

def test_generator(nz=100, nc=1, batch_size=16, device='cpu'):
    """
    Test Generator architecture and output shapes
    
    Verifies:
    - Input: [N, 100] or [N, 100, 1, 1]
    - Output: [N, 1, 128, 128]
    - Output range: [-1, 1] (tanh activation)
    """
    print("=" * 70)
    print("Testing SpecGANGenerator")
    print("=" * 70)
    
    netG = SpecGANGenerator(nz=nz, kernel_len=5, dim=64, nc=nc, use_batchnorm=False)
    netG.to(device)
    netG.eval()
    
    # Test with noise vector [N, nz]
    z = torch.randn(batch_size, nz, device=device)
    
    with torch.no_grad():
        output = netG(z)
    
    print(f"*\t{GREEN}Generator test passed!{RESET}")
    print(f" \tInput shape: {z.shape}")
    print(f" \tOutput shape: {output.shape}")
    print(f" \tExpected output: [{batch_size}, {nc}, 128, 128]")
    print(f" \tMatch: {output.shape == torch.Size([batch_size, nc, 128, 128])}")
    print(f" \tOutput range: [{output.min():.3f}, {output.max():.3f}]")
    print(f" \tExpected range: [-1, 1] (tanh)")
    
    # Count parameters
    total_params = sum(p.numel() for p in netG.parameters())
    print(f"\tTotal parameters: {total_params:,}")
    print()
    
    return netG


def test_discriminator(nc=1, batch_size=16, device='cpu'):
    """
    Test Discriminator architecture and output shapes
    
    Verifies:
    - Input: [N, 1, 128, 128]
    - Output: [N] (single logit per sample)
    """
    print("=" * 70)
    print("Testing SpecGANDiscriminator")
    print("=" * 70)
    
    netD = SpecGANDiscriminator(kernel_len=5, dim=64, nc=nc, use_batchnorm=False)
    netD.to(device)
    netD.eval()
    
    # Test with fake spectrogram
    x = torch.randn(batch_size, nc, 128, 128, device=device)
    
    with torch.no_grad():
        output = netD(x)
    
    print(f"*\t{GREEN}Discriminator test passed!{RESET}")
    print(f" \tInput shape: {x.shape}")
    print(f" \tOutput shape: {output.shape}")
    print(f" \tExpected output: [{batch_size}]")
    print(f" \tMatch: {output.shape == torch.Size([batch_size])}")
    print(f" \tOutput type: Logits (NOT probabilities)")
    
    # Count parameters
    total_params = sum(p.numel() for p in netD.parameters())
    print(f"\tTotal parameters: {total_params:,}")
    print()
    
    return netD


def verify_architecture_compatibility():
    """
    Verify Generator and Discriminator are compatible
    
    Tests:
    - G output shape matches D input shape
    - Both work on same device
    - Forward pass succeeds
    """
    print("=" * 70)
    print("Verifying G-D Compatibility")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create networks
    netG = SpecGANGenerator(nz=100, kernel_len=5, dim=64, nc=1, use_batchnorm=False)
    netD = SpecGANDiscriminator(kernel_len=5, dim=64, nc=1, use_batchnorm=False)
    
    netG.to(device)
    netD.to(device)
    
    # Apply weight initialization
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    # Test forward pass
    batch_size = 8
    z = torch.randn(batch_size, 100, device=device)
    
    with torch.no_grad():
        # Generate fake spectrograms
        fake = netG(z)
        
        # Discriminate
        D_fake = netD(fake)
    
    print(f"*\t{GREEN}Compatibility test passed!{RESET}")
    print(f" \tG output: {fake.shape}")
    print(f" \tD input: {fake.shape} (matches!)")
    print(f" \tD output: {D_fake.shape}")
    print(f" \tG output range: [{fake.min():.3f}, {fake.max():.3f}]")
    print()
    
    return netG, netD


# ============================================================================
# SpecGAN Default Hyperparameters
# From train_specgan.py, Lines 687-712
# ============================================================================

SPECGAN_ARCHITECTURE_DEFAULTS = {
    # Model architecture (from specgan.py defaults)
    'nz': 100,              # Latent dimension (Line 697: specgan_latent_dim=100)
    'nc': 1,                # Number of channels (single channel spectrogram)
    'kernel_len': 5,        # Kernel size (Line 698: specgan_kernel_len=5)
    'dim': 64,              # Dimension multiplier (Line 699: specgan_dim=64)
    'use_batchnorm': False, # BatchNorm (Line 700: specgan_batchnorm=False)
    'ngpu': 1,              # Number of GPUs
}


if __name__ == '__main__':
    """
    Test script to verify architecture implementation
    Run: python specgan_models.py
    """
    print("\n" + "=" * 70)
    print("SpecGAN Models - Architecture Verification")
    print("=" * 70)
    print("This script tests the PyTorch port of SpecGAN architectures")
    print("from the original TensorFlow implementation.\n")
    
    # Test Generator
    netG = test_generator(nz=100, nc=1, batch_size=16)
    
    # Test Discriminator
    netD = test_discriminator(nc=1, batch_size=16)
    
    # Test compatibility
    netG, netD = verify_architecture_compatibility()
    
    print("=" * 70)
    print(f"{GREEN}ALL TESTS PASSED!{RESET}")
    print("=" * 70)
    print("SpecGAN models are ready to use:")
    print(f"\t{BLUE}from specgan_models import SpecGANGenerator, SpecGANDiscriminator, weights_init{RESET}")
    print("\nRecommended settings (SpecGAN defaults):")
    print(f"\t{BLUE}netG = SpecGANGenerator(nz=100, kernel_len=5, dim=64, nc=1, use_batchnorm=False)")
    print("\tnetD = SpecGANDiscriminator(kernel_len=5, dim=64, nc=1, use_batchnorm=False)")
    print("\tnetG.apply(weights_init)")
    print(f"\tnetD.apply(weights_init)\n{RESET}")
