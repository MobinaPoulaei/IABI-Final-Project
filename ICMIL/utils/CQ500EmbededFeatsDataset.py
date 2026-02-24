import torch
import glob
import numpy as np
import os
import random


class CQ500EmbededFeatsDataset(torch.utils.data.Dataset):

    def __init__(self, path, mode="train", augment=True, level=0):

        super().__init__()

        assert mode in ["train", "val", "test"]

        self.path = path
        self.mode = mode
        self.augment_flag = augment and (mode == "train")

        self.data = []
        self.label = []
        self.patient_ids = []

        # class mapping
        self.class_map = {"Normal": 0, "ICH": 1}

        # collect filenames
        filenames = sorted(
            glob.glob(os.path.join(path, mode, "*", "*_resnet1024_feats.npy"))
        )

        print(f"\nLoading {mode} set: {len(filenames)} patients")

        for fname in filenames:
            feats = np.load(fname)

            # label from folder name
            class_name = fname.split(os.sep)[-2]

            label = self.class_map[class_name]

            patient_id = os.path.basename(fname).replace("_resnet1024_feats.npy", "")

            self.data.append(feats.astype(np.float32))
            self.label.append(label)
            self.patient_ids.append(patient_id)

    def __len__(self):

        return len(self.label)

    def augment(self, feats):

        # shuffle instances inside bag (MIL augmentation)
        idx = np.random.permutation(feats.shape[0])
        return feats[idx]

    def __getitem__(self, index):

        feats = self.data[index]

        if self.augment_flag:
            feats = self.augment(feats)

        return feats, self.label[index]


if __name__ == "__main__":

    dataset = CQ500EmbededFeatsDataset(
        "/kaggle/working/CQ500_ICH_VS_NORMAL", mode="train"
    )

    feats, label = dataset[0]

    print("Bag shape:", feats.shape)
    print("Label:", label)
