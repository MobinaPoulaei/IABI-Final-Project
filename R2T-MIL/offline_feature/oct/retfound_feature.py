import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import timm
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from torchvision import transforms

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

base_path = "../data/datasets/paultimothymooney/kermany2018/versions/2/OCT2017 "
retfound_ckpt_path = "./weights/RETFound_mae_natureOCT.pth"

target_classes = ["NORMAL", "CNV", "DME", "DRUSEN"]

label_map = {
    "NORMAL": 0,
    "CNV": 1,
    "DME": 2,
    "DRUSEN": 3,
}

# ------------------------------------------------------------
# Load RETFound backbone (ViT)
# ------------------------------------------------------------

def load_retfound_backbone(checkpoint_path: str) -> nn.Module:
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        num_classes=0,  # remove classification head
        global_pool="",
    )

    checkpoint = torch.load(checkpoint_path)

    # Handle different checkpoint formats
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)

    # Why: RETFound checkpoint may contain decoder weights (from MAE pretraining)
    # which we intentionally ignore.
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.eval()
    model.to(device)
    return model


model = load_retfound_backbone(retfound_ckpt_path)

# ------------------------------------------------------------
# RETFound preprocessing
# ------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats (ViT default)
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------
@torch.no_grad()
def extract_features(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("L")
    img = transform(img).unsqueeze(0).to(device)

    features = model.forward_features(img)  # [1, 197, 1024]

    cls_token = features[:, 0]  # CLS token
    return cls_token.cpu().squeeze(0)  # [1024]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    folder_path = os.path.join(base_path, "test")
    save_path = "./train_features" # do the same for val_features and test_features
    label_path = "./data_label"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    slide_dict = defaultdict(list)
    slide_labels = {}

    for class_name in target_classes:

        class_folder = os.path.join(folder_path, class_name)

        for fname in sorted(os.listdir(class_folder)):
            if not fname.lower().endswith(".jpeg"):
                continue

            name = os.path.splitext(fname)[0]
            parts = name.split("-")
            if len(parts) != 3:
                continue

            disease, patient_id, image_num = parts
            slide_id = f"{disease}-{patient_id}"

            full_path = os.path.join(class_folder, fname)
            slide_dict[slide_id].append(full_path)
            slide_labels[slide_id] = label_map[disease]

    records = []

    for slide_id, image_list in tqdm(
        slide_dict.items(), desc="Processing slides"
    ):
        feats = []

        for img_path in image_list:
            feat = extract_features(img_path)
            feats.append(feat)

        bag_tensor = torch.stack(feats)  # [num_images, 1024]

        disease = slide_id.split("-")[0]
        disease_folder = os.path.join(save_path, disease)
        os.makedirs(disease_folder, exist_ok=True)

        torch.save(
            bag_tensor,
            os.path.join(disease_folder, slide_id + ".pt"),
        )

        records.append(
            {
                "file_name": slide_id,
                "label": slide_labels[slide_id],
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(label_path, "oct_labels.csv"), index=False)
