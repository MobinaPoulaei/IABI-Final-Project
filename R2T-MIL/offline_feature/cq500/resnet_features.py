import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from ICMIL.model.feature_extraction import resnet50_baseline


# ==============================
# USER CONFIG
# ==============================
BATCH_SIZE = 32
SAVE_DIR = "./CQ500_ICH_VS_NORMAL_MIL"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================
# Dataset
# ==============================
class SliceDataset(Dataset):
    """
    Slice-level dataset for feature extraction.
    """

    def __init__(self, slice_paths: List[str], transform: T.Compose):
        self.slice_paths = slice_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.slice_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.slice_paths[idx]

        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)[0].astype(np.float32)

        arr -= arr.min()
        arr /= (arr.max() + 1e-8)
        arr *= 255.0
        arr = arr.astype(np.uint8)

        pil_img = Image.fromarray(arr).convert("RGB")
        return self.transform(pil_img)


# ==============================
# Label Creation
# ==============================
def create_patient_labels(meta_df: pd.DataFrame) -> pd.DataFrame:
    pathology_cols = ["ICH", "Fracture", "MassEffect", "MidlineShift"]

    patient_pathology = meta_df.groupby("PatientID")[pathology_cols].any()

    ich_only = (
        (patient_pathology["ICH"] == True)
        & (~patient_pathology[["Fracture", "MassEffect", "MidlineShift"]].any(axis=1))
    )

    normal_only = ~patient_pathology.any(axis=1)

    patient_labels = pd.DataFrame(index=patient_pathology.index)
    patient_labels["class"] = None
    patient_labels.loc[ich_only, "class"] = "ICH"
    patient_labels.loc[normal_only, "class"] = "Normal"

    return patient_labels.dropna()


def create_data_splits(patient_labels: pd.DataFrame) -> Dict[str, np.ndarray]:
    patients = patient_labels.index.values
    labels = patient_labels["class"].values

    train_patients, temp_patients, train_labels, temp_labels = train_test_split(
        patients,
        labels,
        test_size=(1 - TRAIN_RATIO),
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    val_patients, test_patients, _, _ = train_test_split(
        temp_patients,
        temp_labels,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        stratify=temp_labels,
        random_state=RANDOM_STATE,
    )

    return {
        "train": train_patients,
        "val": val_patients,
        "test": test_patients,
    }


def build_patient_slice_map(meta_df: pd.DataFrame) -> Dict[str, List[str]]:
    patient_slices = {}

    for _, row in meta_df.iterrows():
        pid = row.PatientID
        patient_slices.setdefault(pid, []).append((row.path, row.SliceNumber))

    for pid in patient_slices:
        patient_slices[pid].sort(key=lambda x: x[1])
        patient_slices[pid] = [x[0] for x in patient_slices[pid]]

    return patient_slices


# ==============================
# Main
# ==============================
def main(meta_df: pd.DataFrame) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    model = resnet50_baseline(True).to(DEVICE)
    model.eval()

    patient_labels = create_patient_labels(meta_df)
    splits = create_data_splits(patient_labels)
    patient_slices = build_patient_slice_map(meta_df)

    label_map = {"Normal": 0, "ICH": 1}

    for split_name, split_patients in splits.items():
        print(f"\nProcessing {split_name}")

        split_dir = os.path.join(SAVE_DIR, split_name)
        bags_dir = os.path.join(split_dir, "features")

        os.makedirs(bags_dir, exist_ok=True)

        records = []

        for pid in tqdm(split_patients):
            cls = patient_labels.loc[pid, "class"]
            label = label_map[cls]

            save_path = os.path.join(bags_dir, f"{pid}.pt")
            if os.path.exists(save_path):
                records.append((pid, label))
                continue

            slice_paths = patient_slices[pid]

            dataset = SliceDataset(slice_paths, transform)
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            feats = []

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(DEVICE)
                    out = model(batch)
                    feats.append(out.cpu())

            bag_tensor = torch.cat(feats, dim=0)

            torch.save(bag_tensor, save_path)

            records.append((pid, label))

        # Save labels.csv
        df = pd.DataFrame(records, columns=["patient_id", "label"])
        df.to_csv(os.path.join(split_dir, "labels.csv"), index=False)


if __name__ == "__main__":
    meta_df = pd.read_csv('./meta_df.csv')
    main(meta_df)