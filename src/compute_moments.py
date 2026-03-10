#!/usr/bin/env python
"""
Compute Per-Frequency Moments for SpecGAN Training

This script computes mean and standard deviation for each frequency bin
across all CSV spectrogram files. Must be run BEFORE training SpecGAN.

Based on SpecGAN's moments computation workflow:
    python train_specgan.py moments ./train --data_dir ... --data_moments_fp moments.pkl

Usage:
    python compute_moments.py

Or with custom paths:
    python compute_moments.py --data_dir path/to/csvs --output moments.npz
"""

import os
import specgan_utils as gan_utils
from dotenv import load_dotenv
import platform

GREEN = '\033[32m'
RED = '\033[31m'
BLUE = '\033[34m'
RESET = '\033[0m'
IS_ALTERNATE_PATH = True
"""
def read_args()->Tuple[str, str, bool]:
    \"""
    Read command line arguments for computing the normalization of input .npy files
    :return: Tuple of the path to the directory holding the data, path to the output directory, and if to check all types
    \"""
    parser = argparse.ArgumentParser(
        description='Compute per-frequency moments for SpecGAN training'
    )

    parser.add_argument(
        '-d', '--data_dir',
        type=str,
        required=True,
        help='Directory containing CSV spectrogram files'
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Output path for moments file'
    )

    parser.add_argument(
        '-a', '--all_types',
        action='store_true',
        help='If set, compute moments from all burst types (type_2, type_3, type_5)'
    )

    args = parser.parse_args()

    # Check if directories exist
    if not os.path.exists(args.data_dir):
        print("Error: Specified data directory does not exist")
        exit(1)

    if not os.path.exists(args.output_dir):
        print("Error: Output directory does not exist")
        exit(1)

    return args.data_dir, args.output_dir, args.all_types
"""

def create_moments()->None:

    load_dotenv()
    # 1. Read in arguments for input/output files

    input_dir = os.getenv('DATA_ROOT')
    output_dir = os.getenv('MOMENTS_PATH')
    all_types = os.getenv('ALL_BURSTS')

    if IS_ALTERNATE_PATH:
        input_dir = os.getenv('WINDOWS_DATA_ROOT')
        output_dir = os.getenv('WINDOWS_MOMENTS_PATH')
        all_types = os.getenv('WINDOWS_ALL_BURSTS')

    # Print configuration
    print(
        ("="* 70), "\n",
       "SpecGAN Per-Frequency Moments Computation \n" +
       ("="* 70), "\n",
       "Config:\n"
       f"\tData Directory: {input_dir}\n"
       f"\tOutput Directory: {output_dir}\n"
       f"\tAll Burst Types: {all_types}"
    )

    # 2. Computer the moments and normalize
    print("\n" + "=" * 70)
    print("Starting moments computation...")
    print("" + "=" * 70)

    try:
        normalizer = gan_utils.compute_csv_moments(
            csv_dir=input_dir,
            output_path=f"{output_dir}",
            verbose=True
        )

    except Exception as e:
        print(f"\n{RED}Error during computation: {e}{RESET}")
        import traceback
        traceback.print_exc()
        exit(1)

    #3. Output the SpecGAN dataset
    print(f"\n" + ("=" * 70) + "\n" +
        f"{GREEN}Successfully computed moments!\n{RESET}" +
        ("=" * 70) + "\n" +
        f"Moments file saved to: {BLUE}{output_dir}\n{RESET}"
        f"\nYou can now train SpecGAN with:\n" +
        ("*" * 45) + "\n"
        f"\t{BLUE}dataset = CSVSpectrogramDataset(\n"
        f"\t\troot_dir='{input_dir}',\n"
        f"\t\tnormalize_method='per_frequency',\n"
        f"\t\tmoments_path='{output_dir}/moments.csv',\n"
        f" \t\tgrayscale=True,\n"
        f"\t\taugment=True\n"
        f"\t)\n{RESET}" +
        ("*" * 45)
    )

if __name__ == '__main__':
    create_moments()

