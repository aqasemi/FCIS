import collections
import os
import random
import time
from tiseg.datasets.utils.draw import draw_graph
import cv2


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from fcis_model import FCISNet
from scipy.ndimage import convolve
from skimage import color
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

# Fix randomness
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

fix_all_seeds(2025)

# Configuration
TRAIN_CSV = "/kaggle/input/kaust-vs-kku-tournament-round-3/cells_segmentation/train.csv"
TRAIN_PATH = "/kaggle/input/kaust-vs-kku-tournament-round-3/cells_segmentation/train"
TEST_PATH = "/kaggle/input/kaust-vs-kku-tournament-round-3/cells_segmentation/test"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Data Hyperparameters
WIDTH = 256
HEIGHT = 256
PCT_IMAGES_VALIDATION = 0.075

# Modeling Hyperparameters
BATCH_SIZE = 2
NUM_EPOCHS = 12
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
NUM_CLASSES = 5  # 4 colors + background

# Inference Hyperparameters
MASK_THRESHOLD = 0.5

# Augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(train=True, height=HEIGHT, width=WIDTH):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height, width),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

def rle_decode(mask_rle, shape, color=1):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)

def get_inst_map(annos, height, width):
    inst_map = np.zeros((height, width), dtype=np.int32)
    for i, rle in enumerate(annos):
        mask = rle_decode(rle, (height, width))
        inst_map[mask > 0] = i + 1
    return inst_map

def get_tc_map(inst_map):
    # Implementation of get_tc_from_inst
    # This is a simplified version. The actual implementation might be more complex.
    # For now, we'll use a graph-based coloring algorithm.

    # Get unique instances
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    # Create adjacency matrix
    adj_dict = {inst_id: [] for inst_id in inst_list}
    for inst_id in inst_list:
        inst_mask = (inst_map == inst_id)
        # Dilate mask to find neighbors
        dilated_mask = cv2.dilate(inst_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        neighbors = np.unique(inst_map[dilated_mask > 0])
        for neighbor in neighbors:
            if neighbor != 0 and neighbor != inst_id:
                adj_dict[inst_id].append(neighbor)

    # Graph coloring
    color_map = draw_graph(adj_dict)

    tc_map = np.zeros_like(inst_map, dtype=np.int32)
    for inst_id, color in color_map.items():
        tc_map[inst_map == inst_id] = color

    return tc_map

def get_adj_matrix(inst_map):
    # Create adjacency matrix
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    adj_dict = {inst_id: [] for inst_id in inst_list}
    for inst_id in inst_list:
        inst_mask = (inst_map == inst_id)
        # Dilate mask to find neighbors
        dilated_mask = cv2.dilate(inst_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        neighbors = np.unique(inst_map[dilated_mask > 0])
        for neighbor in neighbors:
            if neighbor != 0 and neighbor != inst_id:
                adj_dict[inst_id].append(neighbor)
    return adj_dict


class CellDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.height = HEIGHT
        self.width = WIDTH

        self.image_info = collections.defaultdict(dict)
        temp_df = self.df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                'image_id': row['id'],
                'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                'annotations': row["annotation"]
            }

    def __getitem__(self, idx):
        info = self.image_info[idx]
        # 1) Load image
        img = Image.open(info['image_path']).convert("RGB")
        img_np = np.array(img)

        # 2) Decode instance map and generate tc_map and adjacency matrix
        inst_map = get_inst_map(info['annotations'], self.height, self.width)
        tc_map = get_tc_map(inst_map)
        adj_matrix = get_adj_matrix(inst_map)
        sem_map = (inst_map > 0).astype(np.uint8)

        # 3) Apply Albumentations transforms
        if self.transforms:
            augmented = self.transforms(
                image=img_np,
                masks=[sem_map, inst_map.astype(np.uint8), tc_map.astype(np.uint8)]
            )
            img = augmented['image']
            sem_map, inst_map, tc_map = augmented['masks']
        else:
            img = torch.from_numpy(img_np.transpose(2,0,1)).float()
            sem_map = torch.from_numpy(sem_map).long()
            inst_map = torch.from_numpy(inst_map).long()
            tc_map = torch.from_numpy(tc_map).long()

        # 4) Build target dict
        target = {
            'sem_gt': sem_map.long(),
            'inst_gt': inst_map.long(),
            'sem_gt_inner': tc_map.long(),
            'adj_gt': adj_matrix,
        }

        return img, target

    def __len__(self):
        return len(self.image_info)

# Split Data
df_base = pd.read_csv(TRAIN_CSV)
unique_ids = df_base['id'].unique()
train_ids, val_ids = train_test_split(unique_ids, test_size=PCT_IMAGES_VALIDATION, random_state=42, shuffle=True)
df_train = df_base[df_base['id'].isin(train_ids)]
df_val = df_base[df_base['id'].isin(val_ids)]

ds_train = CellDataset(TRAIN_PATH, df_train, transforms=get_transform(train=True))
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

ds_val = CellDataset(TRAIN_PATH, df_val, transforms=get_transform(train=False))
dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

# Model
def get_model():
    model = FCISNet(num_classes=NUM_CLASSES)
    return model

model = get_model()
model.to(DEVICE)
model.train();
