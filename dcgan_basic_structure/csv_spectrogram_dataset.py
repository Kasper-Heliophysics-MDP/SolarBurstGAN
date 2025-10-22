"""
Custom PyTorch Dataset for Loading CSV Spectrogram Data for GAN Training

This module provides a custom Dataset class to load 128x128 CSV spectrogram windows
for training DCGAN on solar radio burst data.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from glob import glob


class CSVSpectrogramDataset(Dataset):
    """
    Custom Dataset for loading CSV spectrogram files for GAN training
    
    Each CSV file contains a 128x128 matrix of spectral intensity values.
    This dataset loads, normalizes, and converts them to PyTorch tensors.
    """
    
    def __init__(self, root_dir, transform=None, normalize_method='minmax', 
                 grayscale=True, subsample_ratio=1.0):
        """
        Initialize the CSV Spectrogram Dataset
        
        Args:
            root_dir (str): Root directory containing CSV files (can have subdirectories)
            transform (callable, optional): Optional transform to apply to spectrograms
            normalize_method (str): Normalization method - 'minmax', 'standardize', or 'global'
            grayscale (bool): If True, output 1 channel; if False, duplicate to 3 channels (RGB)
            subsample_ratio (float): Ratio of data to use (for quick testing, 0-1)
        """
        self.root_dir = os.path.abspath(root_dir)  # Convert to absolute path
        self.transform = transform
        self.normalize_method = normalize_method
        self.grayscale = grayscale
        self.subsample_ratio = subsample_ratio
        
        # Check if directory exists
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Directory does not exist: {self.root_dir}\n"
                           f"Current working directory: {os.getcwd()}\n"
                           f"Please check your path or use absolute path.")
        
        # Find all CSV files recursively
        self.csv_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.csv') and file.startswith('window_'):
                    self.csv_files.append(os.path.join(root, file))
        
        # Subsample if needed (for quick testing)
        if subsample_ratio < 1.0:
            n_samples = int(len(self.csv_files) * subsample_ratio)
            self.csv_files = self.csv_files[:n_samples]
        
        if len(self.csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.root_dir}\n"
                           f"Searched for files matching pattern: 'window_*.csv'\n"
                           f"Directory exists: {os.path.exists(self.root_dir)}\n"
                           f"Directory contents: {os.listdir(self.root_dir) if os.path.exists(self.root_dir) else 'N/A'}")
        
        print(f"📊 CSVSpectrogramDataset initialized:")
        print(f"   Root directory: {root_dir}")
        print(f"   Total CSV files found: {len(self.csv_files)}")
        print(f"   Normalization method: {normalize_method}")
        print(f"   Output channels: {1 if grayscale else 3}")
        
        # Analyze burst type distribution
        self._analyze_dataset()
        
    def _analyze_dataset(self):
        """Analyze and display dataset composition"""
        type_counts = {}
        for csv_file in self.csv_files:
            # Extract type from filename: window_type3_...csv
            basename = os.path.basename(csv_file)
            if 'type' in basename.lower():
                burst_type = basename.split('type')[1].split('_')[0]
                type_counts[f"Type {burst_type}"] = type_counts.get(f"Type {burst_type}", 0) + 1
        
        print(f"   Burst type distribution:")
        for burst_type, count in sorted(type_counts.items()):
            print(f"     {burst_type}: {count} windows ({count/len(self.csv_files)*100:.1f}%)")
    
    def __len__(self):
        """Return the total number of samples"""
        return len(self.csv_files)
    
    def __getitem__(self, idx):
        """
        Load and process a single CSV spectrogram
        
        Args:
            idx (int): Index of the sample to load
            
        Returns:
            torch.Tensor: Processed spectrogram tensor [C, H, W]
        """
        csv_path = self.csv_files[idx]
        
        try:
            # Load CSV file (pure numerical data, no headers)
            spectrogram = pd.read_csv(csv_path, header=None).values
            
            # Convert to float32
            spectrogram = spectrogram.astype(np.float32)
            
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
            print(f"❌ Error loading {csv_path}: {e}")
            # Return a zero tensor as fallback
            if self.grayscale:
                return torch.zeros(1, 128, 128)
            else:
                return torch.zeros(3, 128, 128)
    
    def _normalize(self, spectrogram):
        """
        Normalize the spectrogram data
        
        Args:
            spectrogram (np.array): Raw spectrogram data
            
        Returns:
            np.array: Normalized spectrogram in range [-1, 1] (for tanh activation)
        """
        if self.normalize_method == 'minmax':
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
        import matplotlib.pyplot as plt
        
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
            print(f"✅ Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def test_dataset():
    """
    Test function to verify the dataset works correctly
    """
    print("🧪 Testing CSVSpectrogramDataset...")
    
    # Path to your prepared data
    data_root = "/Users/remiliascarlet/Desktop/MDP/transfer_learning/burst_data/csv/gan_training_windows_128"
    
    # Create dataset
    dataset = CSVSpectrogramDataset(
        root_dir=data_root,
        normalize_method='minmax',
        grayscale=False  # Use 3 channels for RGB compatibility
    )
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(dataset)}")
    
    # Test loading a few samples
    print(f"\n🔍 Testing sample loading...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"   Sample {i}: shape={sample.shape}, dtype={sample.dtype}, "
              f"range=[{sample.min():.3f}, {sample.max():.3f}]")
    
    # Visualize first sample
    print(f"\n📊 Creating visualization of first sample...")
    dataset.visualize_sample(0, save_path='test_sample_visualization.png')
    
    return dataset


if __name__ == "__main__":
    test_dataset()

