import os
import kagglehub

new_path = "./data"
os.environ["KAGGLEHUB_CACHE"] = new_path

oct_dataset = kagglehub.dataset_download("paultimothymooney/kermany2018")
print(f"Dataset downloaded to: {oct_dataset}")

cq500_dataset = kagglehub.dataset_download("crawford/qureai-headct")
print(f"Dataset downloaded to: {cq500_dataset}")
