# R2T-MIL: Multiple Instance Learning for Medical Image Classification

This repository implements the **R2T-MIL** method for medical image classification.

The project follows a **Multiple Instance Learning (MIL)** paradigm, where each patient is represented as a *bag* of feature embeddings extracted from multiple slices or scans. The model performs **patient-level classification** using concatenated embeddings.

---

## 🏗️ Project Workflow

The pipeline consists of two main stages:

### 1️⃣ Feature Extraction
High-dimensional medical images are converted into compact embeddings using pretrained backbone models.

### 2️⃣ MIL Classification
The R2T-MIL model processes patient-level embedding bags for final classification.

---

## 📊 Supported Datasets

| Dataset | Modality | Feature Extractor | Task | Classes |
|----------|-----------|------------------|------|----------|
| **OCT** | Retinal OCT Images | RETFound | 4-Class | CNV, DME, DRUSEN, NORMAL |
| **CQ500** | Head CT | ResNet50 | Binary | ICH, NORMAL |

---
