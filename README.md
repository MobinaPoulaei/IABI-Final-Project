# Multiple Instance Learning for Medical Image Classification

This repository implements the **R2T-MIL** and **ICMIL** methods for medical image classification.

The project follows a **Multiple Instance Learning (MIL)** paradigm, where each patient is represented as a *bag* of feature embeddings extracted from multiple slices or scans. The model performs **patient-level classification** using concatenated embeddings.

---

## 📊 Supported Datasets

| Dataset | Modality | Feature Extractor | Task | Classes |
|----------|-----------|------------------|------|----------|
| **OCT** | Retinal OCT Images | RETFound | 4-Class | CNV, DME, DRUSEN, NORMAL |
| **CQ500** | Head CT | ResNet50 | Binary | ICH, NORMAL |

---

## 🏗️ R2T-MIL Project Workflow

The pipeline consists of two main stages:

### 1️⃣ Feature Extraction
High-dimensional medical images are converted into compact embeddings using pretrained backbone models.

### 2️⃣ MIL Classification
The R2T-MIL model processes patient-level embedding bags for final classification.

---

### 📂 Expected Data Structure

####  1️⃣ OCT

data/
└── oct/
├── train_data/
│ ├── features/
│ │ ├── CNV/
│ │ │ ├── CNV-13823.pt
│ │ │ └── ...
│ │ ├── DME/
│ │ ├── DRUSEN/
│ │ └── NORMAL/
│ └── label.csv
├── val_data/
│ ├── features/
│ └── label.csv
└── test_data/
├── features/
└── label.csv

####  2️⃣ CQ500

data/
└── cq500/
└── CQ500_ICH_VS_NORMAL_MIL/
├── train/
│ ├── features/
│ │ ├── CQ500CT1.pt
│ │ ├── CQ500CT2.pt
│ │ └── ...
│ └── label.csv
├── val/
│ ├── features/
│ └── label.csv
└── test/
├── features/
└── label.csv

---

### 🚀 Training & Evaluation

---

#### 1️⃣ OCT Dataset

##### Training
```bash
python main.py \
    --project mil_oct \
    --datasets oct \
    --dataset_root ./offline_feature/oct/train_data \
    --model_path checkpoints \
    --cv_fold 2 \
    --model rrtmil \
    --pool attn \
    --n_trans_layers 2 \
    --da_act tanh \
    --title oct_rrtmil \
    --epeg_k 15 \
    --crmsa_k 1 \
    --all_shortcut \
    --seed 2026 \
    --num_classes 4 \
    --num_epoch 15 \
    --loss ce \

```
##### Testing
```bash
python main.py \
    --project mil_oct \
    --datasets oct \
    --dataset_root ./offline_feature/oct/test_data \
    --model rrtmil \
    --pool attn \
    --n_trans_layers 2 \
    --da_act tanh \
    --title oct_rrtmil \
    --epeg_k 15 \
    --crmsa_k 1 \
    --all_shortcut \
    --seed 2026 \
    --num_classes 4 \
    --test_only \
    --test_model_path ./checkpoints/mil_oct/oct_rrtmil/fold_0_model_best_auc.pt
```

#### 2️⃣ CQ500 Dataset

##### Training 
```bash
python main.py \
    --project mil_cq500 \
    --datasets cq500 \
    --dataset_root ./offline_features/cq500/CQ500_ICH_VS_NORMAL_MIL/train \
    --model_path checkpoints \
    --cv_fold 3 \
    --model rrtmil \
    --pool attn \
    --n_trans_layers 2 \
    --da_act tanh \
    --title cq500_rrtmil \
    --epeg_k 15 \
    --crmsa_k 3 \
    --all_shortcut \
    --seed 2026 \
    --num_classes 2 \
    --num_epoch 15 \
    --loss bce \
```

##### Testing
```bash
python main.py \
    --project mil_cq500 \
    --datasets ocq500t \
    --dataset_root ./offline_feature/cq500/CQ500_ICH_VS_NORMAL_MIL/test_data \
    --model rrtmil \
    --pool attn \
    --n_trans_layers 2 \
    --da_act tanh \
    --title cq500_rrtmil \
    --epeg_k 15 \
    --crmsa_k 3 \
    --all_shortcut \
    --seed 2026 \
    --num_classes 2 \
    --test_only \
    --test_model_path ./checkpoints/mil_cq500/cq500_rrtmil/fold_0_model_best_auc.pt
```

---

## 🏗️ ICMIL Project Workflow

The practical implementation of the ICMIL framework consists of three sequential stages. 

**Important Argument Guidelines:**
- `--save_folder_dir`: The extracted features directory must strictly follow a structure similar to the `Retinal` dataset structure.
- `--checkpoint_path`: Must point to the exact model weights generated and saved during the Classifier Phase (Stage 2).
- `--num_cls`: Defines the classification type. Set to `2` for binary tasks or `4` for 4-class tasks.

### 1️⃣ Feature Extraction
In this initial stage, high-dimensional medical images (OCT scans) are processed using a backbone network to generate compact instance-level embeddings.
```bash
python 01_extract_features.py \
  --source_folder_dir '/path/to/OCT2017' \
  --save_folder_dir '/Retinal_Features' \
```

### 2️⃣ Train Classifier
Once features are extracted, the embedder is kept frozen. The system optimizes the bag-level classifier on the extracted feature bags.

```bash
python 02_train_classifier.py \
--data_path "/Retinal_Features" \
--num_cls 4 
```

### 3️⃣ Train Embedder
Here, the trained bag-level classifier acts as a teacher. Using a confidence-based mechanism, the system distills knowledge to fine-tune the instance-level embedder directly from the raw data.

```bash
python3 03_train_embedder.py \
--data_root "/path/to/OCT2017" \
--checkpoint_path "model_best_oct.pth" \
--num_cls 4
```
---
