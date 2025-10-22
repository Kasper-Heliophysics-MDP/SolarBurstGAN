# Training DCGAN with CSV Spectrogram Data

## Problem Statement

The original `dcgan.ipynb` uses `torchvision.datasets.ImageFolder` which only loads PNG/JPG image files, but our prepared data is in CSV format (128×128 numerical matrices).

## Solution

The following files were created to solve this problem:

### 1. `csv_spectrogram_dataset.py` - Custom Data Loader

This is the core component implementing the `CSVSpectrogramDataset` class, which can:
- Recursively load all CSV files (supports type_2, type_3, type_5 subdirectories)
- Normalize spectrogram data to [-1, 1] range (matching tanh activation)
- Automatically convert to PyTorch tensor format
- Support single-channel or 3-channel (RGB compatible) output
- Provide data visualization functionality

**Key Features:**
```python
dataset = CSVSpectrogramDataset(
    root_dir="path/to/gan_training_windows_128/",
    normalize_method='minmax',  # 'minmax', 'standardize', 'global'
    grayscale=False,  # False=3 channels, True=1 channel
    subsample_ratio=1.0  # Use all data
)
```

### 2. `test_csv_dataset.py` - Test Script

Quickly verify that data loading works correctly.

### 3. `dcgan_csv_training.ipynb` - Complete Training Notebook

Complete DCGAN training workflow, specifically optimized for 128×128 CSV data.

## Quick Start

### Step 1: Test Data Loading

```bash
cd /Users/remiliascarlet/Desktop/MDP/transfer_learning/dcgan
python test_csv_dataset.py
```

This will:
- Load your CSV data
- Verify data format is correct
- Generate visualization images of several samples
- Display data statistics

### Step 2: Train GAN Model

#### Method A: Use the new notebook (Recommended)

Open `dcgan_csv_training.ipynb` and run all cells in sequence.

#### Method B: Modify existing dcgan.ipynb

Only need to modify the data loading section:

**Original code (Cell 7):**
```python
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
```

**New code:**
```python
# Import custom CSV dataset loader
from csv_spectrogram_dataset import CSVSpectrogramDataset

# Create the dataset
dataset = CSVSpectrogramDataset(
    root_dir=dataroot,  # Path to gan_training_windows_128/
    normalize_method='minmax',
    grayscale=False  # Use 3 channels for compatibility
)
```

**Also modify configuration parameters (Cell 7):**
```python
dataroot = "../burst_data/csv/gan_training_windows_128/"
image_size = 128  # Change from 64 to 128
batch_size = 16  # Adjust based on your GPU memory
```

## Key Changes Explained

### 1. Input Size: 64×64 → 128×128

Generator and Discriminator require additional layers to handle larger images:

```python
# Generator adds one layer
nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),  # Added
# ... subsequent layers

# Discriminator adds one layer
nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),  # Added
# ... subsequent layers
```

### 2. Data Normalization

CSV data is automatically normalized to [-1, 1] range, matching Generator's tanh output.

### 3. Number of Channels

Although spectrogram data is inherently single-channel, we replicate to 3 channels to maintain compatibility with standard DCGAN architecture.

## Data Statistics

Based on your prepared data:
- **Type 2**: 36 windows
- **Type 3**: 218 windows
- **Type 5**: 4 windows
- **Total**: 258 128×128 spectrogram windows

## Training Recommendations

1. **Batch Size**: Adjust based on GPU memory (recommend 16-32)
2. **Epochs**: Start with 500, adjust based on loss curves
3. **Learning Rate**: 0.0002 (DCGAN paper recommendation)
4. **Data Augmentation**: Consider adding random flips, rotations, etc.

## Potential Issues

### Issue 1: CSV Files Not Found
```
ValueError: No CSV files found in ...
```
**Solution**: Check that dataroot path correctly points to `gan_training_windows_128/` directory

### Issue 2: Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch_size or use CPU training (set ngpu=0)

### Issue 3: Shape Mismatch
**Solution**: Ensure CSV files are indeed 128×128 (without Date/Time columns)

## Alignment with Paper

According to the paper *"Simulating Solar Radio Bursts Using Generative Adversarial Networks"* (Scully et al., 2023):

- ✅ Uses 128×128 spectrograms
- ✅ Type II and Type III radio bursts
- ✅ DCGAN architecture
- ✅ Normalized to [-1, 1] range

## Next Steps

After training completes, you can:
1. Use the trained Generator to generate new radio burst samples
2. Use generated data to augment your segmentation model training set
3. Use FID (Fréchet Inception Distance) to evaluate generation quality
4. Compare generation effects of different burst types

## References

- Original DCGAN Tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
- Paper: Scully et al. (2023), "Simulating Solar Radio Bursts Using GANs", Solar Physics

