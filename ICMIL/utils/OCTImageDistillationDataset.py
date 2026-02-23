import torch
import glob
import random
import PIL
from PIL import Image
import torchvision.transforms as T


class OCTImageDistillationDataset(torch.utils.data.Dataset):
    """
    Minimal adaptation of RandomPatchDistillationDataset for OCT images.
    Assumes image structure:
        root_path/
            train/
                CNV/*.jpeg
                DME/*.jpeg
                DRUSEN/*.jpeg
                NORMAL/*.jpeg
            test/
                CNV/*.jpeg
                ...
    Train/Val are split from the 'train' folder (80/20).
    Test is read directly from the 'test' folder.
    Supports binary (num_cls=2) and 4-class (num_cls=4).
    """
    def __init__(self, path, mode='train', num_cls=2,
                 transform=T.Compose([
                     T.Resize((256, 256)),
                     T.ToTensor(),
                 ]),
                 transform1=T.Compose([
                     T.Resize((256, 256)),
                     T.RandomHorizontalFlip(),
                     T.RandomVerticalFlip(),
                     T.RandomResizedCrop((256, 256), (0.5, 1)),
                     T.ToTensor(),
                 ]),
                 transform2=T.Compose([
                     T.Resize((256, 256)),
                     T.RandomHorizontalFlip(),
                     T.RandomVerticalFlip(),
                     T.RandomResizedCrop((256, 256), (0.5, 1)),
                     T.ToTensor(),
                 ])):
        super().__init__()
        self.mode = mode
        self.num_cls = num_cls
        self.data = []
        self.labels = []
        self.transform = transform
        self.transform1 = transform1
        self.transform2 = transform2

        # Class mapping — depends on num_cls
        if num_cls == 2:
            self.class_map = {'CNV': 1, 'DME': 1, 'DRUSEN': 1, 'NORMAL': 0}
        else:
            self.class_map = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}

        # ------------------------------------------------------------------ #
        # TEST: read directly from 'test' folder, no splitting               #
        # TRAIN / VAL: read from 'train' folder, split 80/20                 #
        # ------------------------------------------------------------------ #
        if mode == 'test':
            all_files = []
            for class_name, label_idx in self.class_map.items():
                class_files = sorted(glob.glob(f'{path}/test/{class_name}/*.jpeg'))
                for fname in class_files:
                    all_files.append((fname, label_idx))

            if num_cls == 2:
                normal_files  = [(f, l) for f, l in all_files if 'NORMAL' in f]
                disease_files = [(f, l) for f, l in all_files if 'NORMAL' not in f]
                random.seed(552)
                target = min(len(normal_files), len(disease_files))
                normal_files  = random.sample(normal_files,  target)
                disease_files = random.sample(disease_files, target)
                all_files = normal_files + disease_files
            else:
                random.seed(552)

            random.shuffle(all_files)
            random.seed()
            selected = all_files

        else:  # train or val — both sourced from 'train' folder
            all_files = []
            for class_name, label_idx in self.class_map.items():
                class_files = sorted(glob.glob(f'{path}/train/{class_name}/*.jpeg'))
                for fname in class_files:
                    all_files.append((fname, label_idx))

            if num_cls == 2:
                normal_files  = [(f, l) for f, l in all_files if 'NORMAL' in f]
                disease_files = [(f, l) for f, l in all_files if 'NORMAL' not in f]
                random.seed(552)
                target = min(len(normal_files), len(disease_files))
                normal_files  = random.sample(normal_files,  target)
                disease_files = random.sample(disease_files, target)
                all_files = normal_files + disease_files
            else:
                random.seed(552)

            random.shuffle(all_files)
            random.seed()

            n_total = len(all_files)
            n_train = int(0.80 * n_total)

            if mode == 'train':
                selected = all_files[:n_train]
            else:  # val
                selected = all_files[n_train:]

        # ------------------------------------------------------------------ #
        # Store                                                                #
        # ------------------------------------------------------------------ #
        for fname, label in selected:
            self.data.append(fname)
            self.labels.append(label)

        # ------------------------------------------------------------------ #
        # Print statistics                                                     #
        # ------------------------------------------------------------------ #
        print(f'\n[{mode.upper()} SET] {len(self.data)} images total')
        if num_cls == 2:
            n_normal  = self.labels.count(0)
            n_disease = self.labels.count(1)
            print(f'  → NORMAL : {n_normal}')
            print(f'  → DISEASE: {n_disease}')
        else:
            reverse_map = {v: k for k, v in self.class_map.items()}
            for cls_idx in sorted(set(self.labels)):
                print(f'  → {reverse_map[cls_idx]}: {self.labels.count(cls_idx)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fname = self.data[index]
        label = self.labels[index]

        img  = Image.open(fname).convert('RGB')
        npy  = self.transform(img)
        npy1 = self.transform1(img)
        npy2 = self.transform2(img)

        return npy, npy1, npy2, label