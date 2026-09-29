#!/usr/bin/env python3
"""
BraTS Dataset Script 1: Automated Downloader & Extractor
Downloads Medical Segmentation Decathlon Task 01 (Brain Tumour / BraTS)
from the official public AWS S3 mirror (~1.4 GB) and unpacks imagesTr and labelsTr.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path


MSD_BRATS_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "raw_brats"


def download_progress_hook(block_num, block_size, total_size):
    """Displays a clean progress bar during download."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_tot = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[BraTS Downloader] Progress: {percent:5.1f}% ({mb_down:6.1f} MB / {mb_tot:6.1f} MB)")
        sys.stdout.flush()


def download_and_extract_brats(dest_dir: Path = DEFAULT_DEST) -> Path:
    """Downloads Task01_BrainTumour.tar and unpacks it."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "Task01_BrainTumour.tar"
    extracted_folder = dest_dir / "Task01_BrainTumour"

    if extracted_folder.exists() and any(extracted_folder.iterdir()):
        print(f"[BraTS Downloader] Dataset already extracted at: {extracted_folder}")
        return extracted_folder

    if not tar_path.exists():
        print(f"[BraTS Downloader] Downloading Task01_BrainTumour from {MSD_BRATS_URL}...")
        print(f"[BraTS Downloader] Target location: {tar_path}")
        urllib.request.urlretrieve(MSD_BRATS_URL, tar_path, reporthook=download_progress_hook)
        print("\n[BraTS Downloader] Download completed successfully!")
    else:
        print(f"[BraTS Downloader] Archive found at: {tar_path}")

    print(f"[BraTS Downloader] Extracting archive to {dest_dir} (this may take 1-2 minutes)...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dest_dir)

    print(f"[BraTS Downloader] Extraction complete! Cases located at: {extracted_folder}")
    return extracted_folder


if __name__ == "__main__":
    download_and_extract_brats()
