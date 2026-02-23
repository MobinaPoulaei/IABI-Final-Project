import glob
import torch
import numpy as np
import torchvision.transforms as T
import PIL
from tqdm import tqdm
import os
import argparse
from utils.feature_extraction import resnet50_baseline


# ==================== ARGUMENT PARSER ====================
parser = argparse.ArgumentParser(description='OCT Feature Extraction')

parser.add_argument('--source_folder_dir', default='/kaggle/input/datasets/paultimothymooney/kermany2018/OCT2017 ',
                    type=str, help='Path to raw OCT image dataset root')
parser.add_argument('--save_folder_dir', default='/kaggle/working/Retinal',
                    type=str, help='Path to save extracted .npy feature files')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for feature extraction')

params = parser.parse_args()

# Splits and classes to process
splits = ['train', 'val', 'test']


# ==================== SETUP ====================
# Transform (resize to 224x224 for ResNet50, same as ICMIL)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Device and model setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50_baseline(True).to(device)
# model.load_state_dict(torch.load('/your/path/to/ResNet50/weights.pth'))  # Optional: load custom weights
model.eval()

print(f"Using device: {device}")
print(f"Source: {params.source_folder_dir}")
print(f"Destination: {params.save_folder_dir}")
print()

# Dictionary to store patient counts for logging
patient_counts = {}


def get_patient_id(filename):
    """Extract patient ID from filename. Example: CNV-10164-1.jpeg -> CNV-10164"""
    parts = os.path.basename(filename).rsplit('-', 1)
    if len(parts) > 1:
        return parts[0]
    return os.path.splitext(os.path.basename(filename))[0]


# ==================== PROCESSING ====================
for split in splits:
    source_split = os.path.join(params.source_folder_dir, split)

    if not os.path.exists(source_split):
        print(f"Split not found: {source_split}, skipping...")
        continue

    print(f"\n{'='*60}")
    print(f"Processing split: {split}")
    print(f"{'='*60}")

    # Get all classes
    classes = sorted([d for d in os.listdir(source_split)
                      if os.path.isdir(os.path.join(source_split, d))])

    for cls_name in classes:
        cls_source = os.path.join(source_split, cls_name)
        cls_dest   = os.path.join(params.save_folder_dir, split, cls_name)

        # Create destination directory
        os.makedirs(cls_dest, exist_ok=True)

        print(f"\nProcessing class: {cls_name}")

        # Group images by patient ID
        patient_groups = {}
        all_files = sorted(glob.glob(os.path.join(cls_source, "*.jpeg")))
        all_files.extend(sorted(glob.glob(os.path.join(cls_source, "*.jpg"))))
        all_files.extend(sorted(glob.glob(os.path.join(cls_source, "*.png"))))

        for fpath in all_files:
            pid = get_patient_id(fpath)
            patient_groups.setdefault(pid, []).append(fpath)

        print(f"Found {len(patient_groups)} patients in {cls_name}")

        # Store patient count for this split and class
        if split not in patient_counts:
            patient_counts[split] = {}
        patient_counts[split][cls_name] = len(patient_groups)

        # Process each patient (following ICMIL pattern)
        patient_ids = sorted(patient_groups.keys())

        for patient_id in tqdm(patient_ids, desc=f"Processing {cls_name}"):
            save_path = os.path.join(cls_dest, f'{patient_id}_resnet1024_feats.npy')

            # Skip if already processed
            if os.path.exists(save_path):
                continue

            # Get all slides (instances) for this patient
            imgs = sorted(patient_groups[patient_id])

            # Sort by slice number for consistency
            try:
                imgs.sort(key=lambda x: int(os.path.basename(x).rsplit('-', 1)[-1].split('.')[0]))
            except ValueError:
                pass

            feats = []

            # Process in batches (following ICMIL pattern exactly)
            for i in range(0, len(imgs), params.batch_size):
                imgnames     = imgs[i:i + params.batch_size]
                input_tensor = torch.FloatTensor().to(device)

                for imgname in imgnames:
                    img          = PIL.Image.open(imgname).convert('RGB')
                    input_tensor = torch.cat(
                        [input_tensor, transform(img).to(device).unsqueeze(0)], dim=0)

                with torch.no_grad():
                    feat = model(input_tensor)

                feats.extend(feat.cpu().data.numpy())

            # Save features
            feats = np.array(feats)
            np.save(save_path, feats)

    # Log summary for this split
    if split in patient_counts:
        print(f"\n{'='*60}")
        print(f"Summary for {split.upper()}:")
        print(f"{'='*60}")
        total = 0
        for cls_name in sorted(patient_counts[split].keys()):
            count  = patient_counts[split][cls_name]
            total += count
            print(f"  {cls_name:15s}: {count:6d} patients")
        print(f"  {'TOTAL':15s}: {total:6d} patients")
        print(f"{'='*60}")

print("\n" + "=" * 60)
print("Feature extraction completed!")
print("=" * 60)

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 60)
print("FINAL SUMMARY - Patient Counts")
print("=" * 60)
for split in ['train', 'val', 'test']:
    if split in patient_counts:
        print(f"\n{split.upper()}:")
        total = 0
        for cls_name in sorted(patient_counts[split].keys()):
            count  = patient_counts[split][cls_name]
            total += count
            print(f"  {cls_name:15s}: {count:6d} patients")
        print(f"  {'TOTAL':15s}: {total:6d} patients")
print("=" * 60)
