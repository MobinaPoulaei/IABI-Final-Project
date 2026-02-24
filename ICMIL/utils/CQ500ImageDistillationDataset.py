import torch
import random
import glob
import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
import torchvision.transforms as T


class CQ500ImageDistillationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path,
        meta_df,
        mode="train",
        level=0,
    ):

        super().__init__()

        assert mode in ["train", "val", "test"]

        self.path = path
        self.mode = mode

        self.meta_df = meta_df

        filenames = sorted(
            glob.glob(os.path.join(path, mode, "*", "*_resnet1024_feats.npy"))
        )
        self.patient_ids = set(
            [
                os.path.basename(fname).replace("_resnet1024_feats.npy", "")
                for fname in filenames
            ]
        )

        # keep only slices belonging to this split
        self.slice_df = meta_df[
            meta_df["PatientID"].isin(self.patient_ids)
        ].reset_index(drop=True)

        self.data = self.slice_df["path"].tolist()

        print(f"{mode}: {len(self.data)} slices")

        transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )

        transform1 = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop((256, 256), (0.5, 1)),
                T.ToTensor(),
            ]
        )
        transform2 = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop((256, 256), (0.5, 1)),
                T.ToTensor(),
            ]
        )

        self.transform = transform
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.data)

    def load_slice(self, path):

        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)[0]

        # normalize to uint8
        arr = arr.astype(np.float32)

        arr -= arr.min()
        arr /= arr.max() + 1e-8
        arr *= 255

        arr = arr.astype(np.uint8)

        pil_img = Image.fromarray(arr).convert("RGB")

        return pil_img

    def __getitem__(self, index):
        fname = self.data[index]
        img = self.load_slice(fname)

        npy = self.transform(img)
        npy1 = self.transform1(img)
        npy2 = self.transform2(img)

        return npy, npy1, npy2
