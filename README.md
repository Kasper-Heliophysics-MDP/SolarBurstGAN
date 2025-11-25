# SolarBurstGAN – GAN for Solar Radio Burst Generation

PyTorch implementation of SpecGAN/DCGAN-style models for generating **128×128 solar radio spectrograms**.

Includes end-to-end workflow:
- Dataset loading (CSV / NumPy)
- Per-frequency normalization
- SpecGAN architecture (PyTorch)
- Training & sample generation
- Environment-based configuration

Ported and adapted from: [Chris Donahue’s SpecGAN](https://github.com/chrisdonahue/wavegan)

---

## Quick Start

### 1. Set Up `.env` File
Create a file named **`.env`** in the src file:

```env
DATA_ROOT="../input_data"
TEST_IMG_ROOT="../test_img"
MOMENTS_PATH="../moments/moments.npz"
CHECKPOINT_ROOT="../checkpoints"
TRAINING_OUTPUT_ROOT="../gan_data"

SAVE_INTERVAL=5
NPGU=1
EPOCHS=5
DISC_NUPDATES=5
ALL_BURSTS=True
BATCH_SIZE=16
SEED=999
NEW_SAMPLES=16
```
This can be used to customize the input/output directories and the paramaters for training!

---

### 2. Pre-compute Normalization Statistics (Run Once)

```bash
python src/compute_moments.py
```

This generates:

```
moments/moments.npz
```

---

### 3. Train the Model

> **Recommended**: Use the Jupyter Notebook  
> **`src/specgan_training.ipynb` is the main training workflow.**

It loads the dataset, model, trainer, runs training loops, and saves checkpoints and generated samples.

Open the notebook:

```bash
jupyter notebook src/specgan_training.ipynb
```

Run through all cells.

---

### 4. View Output

Generated test images before training:
```
test_img/
```

GAN sample `.npy` files (the actual result of the GAN):

```
gan_data/
```

Checkpoints of the best and most recent GAN models:

```
checkpoints/
```
#### Our best checkpoint so far can be found here: https://drive.google.com/file/d/1VRXuY8vkVb7zOJidTYQdEpNmc-PQ9dVu/view?usp=sharing
---

## Files

- **src/specgan_models.py** - Generator & Discriminator
- **src/specgan_utils.py** - Training utilities (loss, normalization, checkpoints)
- **src/compute_moments.py** - Preprocessing script
- **src/specgan_training.ipynb** - Primary training workflow
- **src/spectrogram_dataset.py** - Dataset loader

---

## Citation

If you use this work, please cite the original SpecGAN paper:

```bibtex
@inproceedings{{donahue2019wavegan,
  title={{Adversarial Audio Synthesis}},
  author={{Donahue, Chris and McAuley, Julian and Puckette, Miller}},
  booktitle={{ICLR}},
  year={{2019}}
}}
```

