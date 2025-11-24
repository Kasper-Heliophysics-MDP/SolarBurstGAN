"""
Custom PyTorch Dataset for Loading Spectrogram Data for GAN Training

This module provides a custom Dataset class to load 128x128 spectrogram windows
for training GAN on solar radio burst data.

Supports:
- CSV files (window_*.csv format)
- NumPy files (burst-*.npy format, e.g., eCallisto processed data)
- DCGAN (global normalization) and SpecGAN (per-frequency normalization)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dotenv import load_dotenv
import matplotlib.pyplot as plt

GREEN = '\033[32m'
RED = '\033[31m'
BLUE = '\033[34m'
RESET = '\033[0m'

class SpectrogramDataset(Dataset):
    """
    Custom Dataset for loading spectrogram files for GAN training
    
    Supports both CSV and NumPy (.npy) formats:
    - CSV: 128x128 matrix with no headers (window_*.csv)
    - NPY: 128x128 numpy array (burst-*.npy, e.g., eCallisto processed data)
    
    This dataset loads, normalizes, and converts them to PyTorch tensors.
    """
    
    def __init__(self, root_dir, transform=None, normalize_method='minmax', 
                 grayscale=True, subsample_ratio=1.0,
                 normalizer=None, moments_path=None, augment=False):
        """
        Initialize the CSV Spectrogram Dataset
        
        Args:
            root_dir (str): Root directory containing CSV files (can have subdirectories)
            transform (callable, optional): Optional transform to apply to spectrograms
            normalize_method (str): Normalization method:
                - 'minmax': Global min-max normalization (DCGAN)
                - 'standardize': Global z-score normalization
                - 'global': Global log-scale normalization
                - 'per_frequency': Per-frequency bin normalization (SpecGAN) - requires normalizer
            grayscale (bool): If True, output 1 channel; if False, duplicate to 3 channels (RGB)
            subsample_ratio (float): Ratio of data to use (for quick testing, 0-1)
            normalizer (PerFrequencyNormalizer, optional): For 'per_frequency' normalization
            moments_path (str, optional): Path to pre-computed moments file (.npz)
            augment (bool): If True, apply temporal shift augmentation (SpecGAN-inspired)
        """
        self.root_dir = os.path.abspath(root_dir)  # Convert to absolute path
        self.transform = transform
        self.normalize_method = normalize_method
        self.grayscale = grayscale
        self.subsample_ratio = subsample_ratio
        self.normalizer = normalizer
        self.augment = augment
        
        # Check if directory exists
        if not os.path.exists(self.root_dir):
            raise ValueError(f"{RED}Directory does not exist: {self.root_dir}\n"
                           f"Current working directory: {os.getcwd()}\n"
                           f"Please check your path or use absolute path.{RESET}")
        
        # Find all spectrogram files recursively (CSV or NPY)
        self.csv_files = []  # Name kept for backward compatibility
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                # Support CSV files (window_*.csv format)
                if file.endswith('.csv') and file.startswith('window_'):
                    self.csv_files.append(os.path.join(root, file))
                # Support NPY files (burst-*.npy format, e.g., eCallisto)
                elif file.endswith('.npy') and file.startswith('burst-'):
                    self.csv_files.append(os.path.join(root, file))
        
        # Subsample if needed (for quick testing)
        if subsample_ratio < 1.0:
            n_samples = int(len(self.csv_files) * subsample_ratio)
            self.csv_files = self.csv_files[:n_samples]
        
        if len(self.csv_files) == 0:
            raise ValueError(f"No spectrogram files found in {self.root_dir}\n"
                           f"Searched for patterns: 'window_*.csv' or 'burst-*.npy'\n"
                           f"Directory exists: {os.path.exists(self.root_dir)}\n"
                           f"Directory contents: {os.listdir(self.root_dir) if os.path.exists(self.root_dir) else 'N/A'}")
        
        # Auto-load moments if path provided
        if moments_path:
            if self.normalizer is None:
                # Lazy import to avoid circular dependency
                try:
                    from specgan_utils import PerFrequencyNormalizer
                    self.normalizer = PerFrequencyNormalizer()
                except ImportError:
                    raise ImportError(
                        f"{RED}Per-frequency normalization requires specgan_utils.py. "
                        f"Please create specgan_utils.py with PerFrequencyNormalizer class.{RESET}"
                    )
            self.normalizer.load_moments(moments_path)
            print(f"*\t{GREEN}Loaded per-frequency moments from: {moments_path}{RESET}")
        
        # Count file types
        csv_count = sum(1 for f in self.csv_files if f.endswith('.csv'))
        npy_count = sum(1 for f in self.csv_files if f.endswith('.npy'))
        
        print(f"*\tSpectrogramDataset initialized:")
        print(f" \tRoot directory: {root_dir}")
        print(f" \tTotal files found: {len(self.csv_files)} (CSV: {csv_count}, NPY: {npy_count})")
        print(f" \tNormalization method: {normalize_method}")
        print(f" \tOutput channels: {1 if grayscale else 3}")
        print(f" \tData augmentation: {'Enabled' if augment else 'Disabled'}")
        if normalize_method == 'per_frequency':
            status = 'Ready' if self.normalizer else 'Missing (provide normalizer or moments_path)'
            print(f" \tPer-frequency normalizer: {status}")
        
        # Analyze burst type distribution
        self._analyze_dataset()
        
    def _analyze_dataset(self):
        """Analyze and display dataset composition"""
        type_counts = {}
        location_counts = {}
        
        for file_path in self.csv_files:
            basename = os.path.basename(file_path)

            # Extract type from CSV filename: window_type3_...csv
            if 'type' in basename.lower() and basename.endswith('.csv'):
                burst_type = basename.split('type')[1].split('_')[0]
                type_counts[f"Type {burst_type}"] = type_counts.get(f"Type {burst_type}", 0) + 1
            
            # Extract location from NPY filename: burst-LOCATION-...npy
            elif basename.startswith('burst-') and basename.endswith('.npy'):
                parts = basename.split('-')
                if len(parts) >= 2:
                    location = parts[1]
                    location_counts[location] = location_counts.get(location, 0) + 1
        
        # Print type distribution (for CSV files)
        if type_counts:
            print(f" \tBurst type distribution (CSV):")
            for burst_type, count in sorted(type_counts.items()):
                print(f" \t\t{burst_type}: {count} windows ({count/len(self.csv_files)*100:.1f}%)")
        
        # Print location distribution (for NPY files)
        if location_counts:
            print(f" \tLocation distribution (NPY):")
            for location, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True):
                print(f" \t\t{location}: {count} files ({count/len(self.csv_files)*100:.1f}%)")
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        """
        Load and process a single spectrogram (CSV or NPY)
        
        Args:
            idx (int): Index of the sample to load
            
        Returns:
            torch.Tensor: Processed spectrogram tensor [C, H, W]
        """
        file_path = self.csv_files[idx]
        
        try:
            # Load file based on extension
            if file_path.endswith('.npy'):
                # Load NumPy file (already 128×128 array)
                spectrogram = np.load(file_path)
                
            elif file_path.endswith('.csv'):
                # Load CSV file (pure numerical data, no headers)
                spectrogram = pd.read_csv(file_path, header=None).values
                
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Convert to float32
            spectrogram = spectrogram.astype(np.float32)
            
            # Apply augmentation BEFORE normalization (SpecGAN-inspired)
            if self.augment:
                spectrogram = self._augment(spectrogram)
            
            # Normalize
            spectrogram = self._normalize(spectrogram)
            
            # Convert to tensor [H, W] -> [1, H, W]
            spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)
            
            # If not grayscale, duplicate to 3 channels for compatibility with RGB models
            if not self.grayscale:
                spectrogram = spectrogram.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
            
            # Apply additional transforms if provided
            if self.transform:
                spectrogram = self.transform(spectrogram)
            
            return spectrogram
            
        except Exception as e:
            print(f"*\t{RED}Error loading {file_path}: {e}{RESET}")
            # Return a zero tensor as fallback
            if self.grayscale:
                return torch.zeros(1, 128, 128)
            else:
                return torch.zeros(3, 128, 128)
    
    def _augment(self, spectrogram):
        """
        Apply data augmentation to spectrogram (SpecGAN-inspired)
        
        Args:
            spectrogram (np.array): Raw spectrogram data [H, W]
            
        Returns:
            np.array: Augmented spectrogram
        """
        # Temporal shift augmentation (shifts burst along time axis)
        # This addresses the "left-side concentration" issue
        max_shift = 30  # Maximum shift in time bins (~3 seconds at 0.1s sampling)
        shift = np.random.randint(-max_shift, max_shift + 1)
        spectrogram = np.roll(spectrogram, shift, axis=1)  # Shift along time axis (axis=1)
        
        # Optional: Add small noise for robustness (uncomment if needed)
        # noise_std = 0.02
        # noise = np.random.randn(*spectrogram.shape) * noise_std
        # spectrogram = spectrogram + noise
        
        return spectrogram
    
    def _normalize(self, spectrogram):
        """
        Normalize the spectrogram data
        
        Args:
            spectrogram (np.array): Raw spectrogram data [H, W]
            
        Returns:
            np.array: Normalized spectrogram in range [-1, 1] (for tanh activation)
        """
        if self.normalize_method == 'per_frequency':
            # SpecGAN-style per-frequency bin normalization
            # Each frequency bin is normalized independently using pre-computed statistics
            if self.normalizer is None:
                raise ValueError(
                    "Per-frequency normalization requires a normalizer. "
                    "Please provide 'normalizer' or 'moments_path' in __init__. "
                    "Run compute_moments.py first to generate moments file."
                )
            return self.normalizer.normalize(spectrogram)
            
        elif self.normalize_method == 'minmax':
            # Min-max normalization to [-1, 1]
            min_val = spectrogram.min()
            max_val = spectrogram.max()
            if max_val > min_val:
                spectrogram = 2 * (spectrogram - min_val) / (max_val - min_val) - 1
            else:
                spectrogram = np.zeros_like(spectrogram)
                
        elif self.normalize_method == 'standardize':
            # Z-score standardization, then clip to [-3, 3] and scale to [-1, 1]
            mean = spectrogram.mean()
            std = spectrogram.std()
            if std > 0:
                spectrogram = (spectrogram - mean) / std
                spectrogram = np.clip(spectrogram, -3, 3) / 3  # Scale to [-1, 1]
            else:
                spectrogram = np.zeros_like(spectrogram)
                
        elif self.normalize_method == 'global':
            # Global normalization using log scale (common for spectrograms)
            # Avoid log(0) by adding small epsilon
            epsilon = 1e-10
            spectrogram = np.log10(spectrogram + epsilon)
            # Then apply min-max to [-1, 1]
            min_val = spectrogram.min()
            max_val = spectrogram.max()
            if max_val > min_val:
                spectrogram = 2 * (spectrogram - min_val) / (max_val - min_val) - 1
            else:
                spectrogram = np.zeros_like(spectrogram)
        
        return spectrogram
    
    def get_sample_paths(self, n_samples=5):
        """
        Get paths of first n samples for inspection
        
        Args:
            n_samples (int): Number of sample paths to return
            
        Returns:
            list: List of file paths
        """
        return self.csv_files[:n_samples]
    
    def visualize_sample(self, idx, save_path=None):
        """
        Visualize a single sample spectrogram
        
        Args:
            idx (int): Index of the sample to visualize
            save_path (str, optional): Path to save the visualization
        """

        spectrogram = self[idx]
        
        # Convert tensor to numpy for visualization
        if spectrogram.shape[0] == 1:
            # Grayscale
            spec_np = spectrogram.squeeze(0).numpy()
        else:
            # RGB - take first channel
            spec_np = spectrogram[0].numpy()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(spec_np, aspect='auto', cmap='hot', origin='lower')
        plt.colorbar(label='Normalized Intensity')
        plt.title(f'Sample {idx}: {os.path.basename(self.csv_files[idx])}')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"*\t{GREEN}Saved visualization to {save_path}{RESET}")
        else:
            plt.show()
        
        plt.close()


def test_dataset():
    """
    Test function to verify the dataset works correctly
    """
    print("="*70)
    print("Testing CSVSpectrogramDataset...")
    print("="*70)

    # Path to prepared data
    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    img_root = os.getenv("TEST_IMG_ROOT")

    # Create dataset
    dataset = SpectrogramDataset(
        root_dir=data_root,
        normalize_method='minmax',
        grayscale=False  # Use 3 channels for RGB compatibility
    )
    
    print(f"\n*\tDataset loaded successfully!")
    print(f" \tTotal samples: {len(dataset)}")
    
    # Test loading a few samples
    print(f"\n*\tTesting sample loading...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f" \tSample {i}: shape={sample.shape}, dtype={sample.dtype}, "
              f"range=[{sample.min():.3f}, {sample.max():.3f}]")
    
    # Visualize first sample
    print(f"\n*\tCreating visualization of first sample...\n")
    dataset.visualize_sample(0, save_path=f'{img_root}/test_sample_visualization.png')
    
    return dataset

if __name__ == "__main__":
    test_dataset()
