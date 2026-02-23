import os
import kagglehub

new_path = "./data"
os.environ["KAGGLEHUB_CACHE"] = new_path

path_1 = kagglehub.dataset_download("paultimothymooney/kermany2018")
print(f"Dataset downloaded to: {path_1}")

path_2 = kagglehub.dataset_download("crawford/qureai-headct")
print(f"Dataset downloaded to: {path_2}")
