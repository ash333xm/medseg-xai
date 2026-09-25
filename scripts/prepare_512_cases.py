"""
Prepares the standardized 512x512x3 BraTS cases for MedSeg-XAI shared Google Drive folder.
Each case conforms strictly to the team contracts:
  - image: (512, 512, 3) uint8 in [0, 255]
  - true_mask: (512, 512) uint8 containing only 0 and 1
  - box: [x_min, y_min, x_max, y_max] in identical 512x512 coordinates
  - metadata: modality, slice_idx, intensity scaling, label merging, orientation, box padding.
"""

from pathlib import Path
import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [
    WORKSPACE_ROOT / "medsegxai" / "data",
    WORKSPACE_ROOT / "medseg_all_datasets" / "medsegxai" / "data"
]

METADATA_SPECS = [
    {
        "case_id": "case_01",
        "patient_id": "BraTS20_Training_001",
        "slice_idx": 78,
        "modality": "T1ce / T1gd (Contrast-Enhanced T1-weighted MRI, MSD Task 01 Channel 2)",
        "intensity_scaling": "0.5th to 99.5th percentile clipping, min-max normalized to [0, 255] uint8",
        "tumor_labels_merged": "Whole Tumor (WT): Labels 1 (NCR/NET), 2 (ED), and 3 (ET) merged into 1 (binary 0/1)",
        "orientation": "Axial plane, radiological orientation, upright (identical affine applied to image and mask)",
        "box_padding": "Padded by 12 pixels around tumor perimeter",
        "pathology": "Glioblastoma (GBM) with central necrotic cavity and active peripheral rim enhancement"
    },
    {
        "case_id": "case_02",
        "patient_id": "BraTS20_Training_002",
        "slice_idx": 84,
        "modality": "T1ce / T1gd (Contrast-Enhanced T1-weighted MRI, MSD Task 01 Channel 2)",
        "intensity_scaling": "0.5th to 99.5th percentile clipping, min-max normalized to [0, 255] uint8",
        "tumor_labels_merged": "Whole Tumor (WT): Labels 1 (NCR/NET), 2 (ED), and 3 (ET) merged into 1 (binary 0/1)",
        "orientation": "Axial plane, radiological orientation, upright (identical affine applied to image and mask)",
        "box_padding": "Padded by 12 pixels around tumor perimeter",
        "pathology": "High-grade glioma with diffuse infiltrative margins extending into subcortical white matter"
    },
    {
        "case_id": "case_03",
        "patient_id": "BraTS20_Training_003",
        "slice_idx": 90,
        "modality": "T1ce / T1gd (Contrast-Enhanced T1-weighted MRI, MSD Task 01 Channel 2)",
        "intensity_scaling": "0.5th to 99.5th percentile clipping, min-max normalized to [0, 255] uint8",
        "tumor_labels_merged": "Whole Tumor (WT): Labels 1 (NCR/NET), 2 (ED), and 3 (ET) merged into 1 (binary 0/1)",
        "orientation": "Axial plane, radiological orientation, upright (identical affine applied to image and mask)",
        "box_padding": "Padded by 12 pixels around tumor perimeter",
        "pathology": "Deep parenchymal astrocytoma abutting lateral ventricles with extensive peritumoral edema"
    },
    {
        "case_id": "case_04",
        "patient_id": "BraTS20_Training_004",
        "slice_idx": 82,
        "modality": "T1ce / T1gd (Contrast-Enhanced T1-weighted MRI, MSD Task 01 Channel 2)",
        "intensity_scaling": "0.5th to 99.5th percentile clipping, min-max normalized to [0, 255] uint8",
        "tumor_labels_merged": "Whole Tumor (WT): Labels 1 (NCR/NET), 2 (ED), and 3 (ET) merged into 1 (binary 0/1)",
        "orientation": "Axial plane, radiological orientation, upright (identical affine applied to image and mask)",
        "box_padding": "Padded by 12 pixels around tumor perimeter",
        "pathology": "Temporal lobe glioblastoma with distinct hyperintense contrast-enhancing nodule"
    },
    {
        "case_id": "case_05",
        "patient_id": "BraTS20_Training_005",
        "slice_idx": 88,
        "modality": "T1ce / T1gd (Contrast-Enhanced T1-weighted MRI, MSD Task 01 Channel 2)",
        "intensity_scaling": "0.5th to 99.5th percentile clipping, min-max normalized to [0, 255] uint8",
        "tumor_labels_merged": "Whole Tumor (WT): Labels 1 (NCR/NET), 2 (ED), and 3 (ET) merged into 1 (binary 0/1)",
        "orientation": "Axial plane, radiological orientation, upright (identical affine applied to image and mask)",
        "box_padding": "Padded by 12 pixels around tumor perimeter",
        "pathology": "Frontal lobe glioma presenting with discrete hyperintense mass effect"
    }
]


def generate_and_save():
    print("Converting cases to 512x512x3 standard format...")

    for spec in METADATA_SPECS:
        case_id = spec["case_id"]
        src_file = DATA_DIRS[0] / f"{case_id}.npy"
        d = np.load(src_file, allow_pickle=True).item()

        img_256 = d["image"]
        mask_256 = d["true_mask"]
        box_256 = d["box"]

        # 1. Upscale image to 512x512 with bicubic interpolation
        pil_img = Image.fromarray(img_256).resize((512, 512), resample=Image.BICUBIC)
        img_512_gray = np.array(pil_img, dtype=np.uint8)
        # Replicate to 3-channel RGB for MedSAM / Vision Transformer
        img_512_3ch = np.repeat(img_512_gray[:, :, None], 3, axis=-1)

        # 2. Upscale ground-truth mask to 512x512 with nearest-neighbor to preserve exact binary {0, 1}
        pil_mask = Image.fromarray((mask_256 * 255).astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
        mask_512 = (np.array(pil_mask) > 127).astype(np.uint8)

        # 3. Rescale bounding box to 512x512 coordinates
        box_512 = [
            int(round(box_256[0] * 2)),
            int(round(box_256[1] * 2)),
            int(round(box_256[2] * 2)),
            int(round(box_256[3] * 2)),
        ]

        # Pack case dictionary
        case_dict = {
            "case_id": case_id,
            "image": img_512_3ch,        # (512, 512, 3) uint8
            "true_mask": mask_512,       # (512, 512) uint8 [0, 1]
            "box": box_512,              # [x_min, y_min, x_max, y_max]
            **spec
        }

        # Save to all data locations
        for d_dir in DATA_DIRS:
            if d_dir.exists():
                out_path = d_dir / f"{case_id}.npy"
                np.save(out_path, case_dict, allow_pickle=True)
                print(f"  [OK] Saved {out_path} -> image: {img_512_3ch.shape}, mask: {mask_512.shape}, box: {box_512}")

    print("\nAll 5 cases successfully upgraded to 512x512x3 format!")


if __name__ == "__main__":
    generate_and_save()
