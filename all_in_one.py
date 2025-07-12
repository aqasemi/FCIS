# ==============================================================================
# Part 1: Imports and Configuration
# ==============================================================================

import collections
import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.ndimage import binary_fill_holes, convolve
from skimage import color, measure
from skimage.morphology import remove_small_objects
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16_bn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2


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


# ==============================================================================
# Part 2: Model Definition (FCISNet)
# ==============================================================================

def conv1x1(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return nn.Sequential(
        nn.Conv2d(in_dims, out_dims, 1, 1, 0),
        nn.BatchNorm2d(out_dims) if norm_cfg else nn.Identity(),
        nn.ReLU(inplace=True) if act_cfg else nn.Identity()
    )

def conv3x3(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return nn.Sequential(
        nn.Conv2d(in_dims, out_dims, 3, 1, 1),
        nn.BatchNorm2d(out_dims) if norm_cfg else nn.Identity(),
        nn.ReLU(inplace=True) if act_cfg else nn.Identity()
    )

def transconv4x4(in_dims, out_dims, bn):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_dims, out_channels=out_dims, kernel_size=(4, 4), stride=2, padding=1, bias=not bn),
        nn.BatchNorm2d(out_dims),
        nn.ReLU(inplace=True),
    )


class UNetLayer(nn.Module):

    def __init__(self, in_dims, skip_dims, feed_dims, num_convs=2, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_dims = in_dims
        self.skip_dims = skip_dims
        self.feed_dims = feed_dims

        self.up_conv = transconv4x4(in_dims, feed_dims, norm_cfg is not None)

        convs = [conv3x3(skip_dims + feed_dims, feed_dims, norm_cfg, act_cfg)]
        for _ in range(num_convs - 2):
            convs.append(conv3x3(feed_dims, feed_dims, norm_cfg, act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x, skip):
        x = self.up_conv(x)

        if x.shape != skip.shape:
            diff_h = skip.shape[-2] - x.shape[-2]
            diff_w = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2))

        x = torch.cat([x, skip], dim=1)
        out = self.convs(x)
        return out


class FCISHead(nn.Module):
    def __init__(self,
                 num_classes=None,
                 bottom_in_dim=512,
                 skip_in_dims=[64, 128, 256, 512, 512],
                 stage_dims=[16, 32, 64, 128, 256],
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_classes = num_classes
        self.bottom_in_dim = bottom_in_dim
        self.skip_in_dims = skip_in_dims
        self.stage_dims = stage_dims
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        num_layers = len(self.skip_in_dims)

        self.decode_layers = nn.ModuleList()
        for idx in range(num_layers - 1, -1, -1):
            if idx == num_layers - 1:
                # bottom, initial layer
                self.decode_layers.append(
                    UNetLayer(self.bottom_in_dim, self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg, act_cfg))
            else:
                self.decode_layers.append(
                    UNetLayer(self.stage_dims[idx + 1], self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg,
                              act_cfg))

        if self.num_classes is not None:
            self.cls_layer = nn.Conv2d(self.stage_dims[0], self.num_classes, kernel_size=1, stride=1)
            self.sem_layer = nn.Conv2d(self.num_classes-1, 1, kernel_size=1, stride=1)

    def forward(self, bottom_input, skip_inputs):
        # decode stage feed forward
        x = bottom_input
        skips = skip_inputs[::-1]

        decode_layers = self.decode_layers
        for skip, decode_stage in zip(skips, decode_layers):
            x = decode_stage(x, skip)

        out = x

        if self.num_classes is not None:
            inter_pred = self.cls_layer(out)
            back_pred = inter_pred[:, :1, ...]
            cls_pred = inter_pred[:, 1:, ...]

            fore_pred = self.sem_layer(cls_pred)
            sem_pred = torch.cat([back_pred, fore_pred], dim=1)

        return sem_pred, inter_pred

class FCISNet(nn.Module):
    def __init__(self, num_classes):
        super(FCISNet, self).__init__()
        self.num_classes = num_classes

        # Using torchvision's VGG16 with batch norm
        vgg_model = vgg16_bn(pretrained=True)
        self.backbone = nn.ModuleList(vgg_model.features)

        # The out_indices are adjusted to match the layers of the torchvision model
        self.out_indices = [6, 13, 23, 33, 43]

        self.head = FCISHead(
            num_classes=self.num_classes,
            bottom_in_dim=512,
            skip_in_dims=[64, 128, 256, 512, 512],
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'))

    def forward(self, img):
        img_feats = []
        x = img
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.out_indices:
                img_feats.append(x)

        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        sem_pred, inter_pred = self.head(bottom_feat, skip_feats)

        return sem_pred, inter_pred


# ==============================================================================
# Part 3: Loss Functions
# ==============================================================================

class BatchMultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes):
        super(BatchMultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target, weights=None):
        smooth = 1e-5
        input = F.softmax(input, dim=1)

        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if weights is None:
            weights = torch.ones(self.num_classes).to(input.device)

        dice_class = torch.zeros(self.num_classes).to(input.device)
        for i in range(self.num_classes):
            intersection = torch.sum(input[:, i] * target_one_hot[:, i])
            union = torch.sum(input[:, i]) + torch.sum(target_one_hot[:, i])
            dice_class[i] = (2 * intersection + smooth) / (union + smooth)

        dice_loss = 1 - torch.sum(weights * dice_class) / torch.sum(weights)
        return dice_loss

class GradientMSELoss(nn.Module):
    def __init__(self):
        super(GradientMSELoss, self).__init__()

    def forward(self, pred, gt, fore_gt):
        pred_grad_x = F.conv2d(pred, weight=torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32).to(pred.device), padding=1)
        pred_grad_y = F.conv2d(pred, weight=torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32).to(pred.device), padding=1)
        gt_grad_x = F.conv2d(gt, weight=torch.tensor([[[[-1, 0, 1]]]], dtype=torch.float32).to(gt.device), padding=1)
        gt_grad_y = F.conv2d(gt, weight=torch.tensor([[[[-1], [0], [1]]]], dtype=torch.float32).to(gt.device), padding=1)

        loss_x = F.mse_loss(pred_grad_x, gt_grad_x, reduction='none')
        loss_y = F.mse_loss(pred_grad_y, gt_grad_y, reduction='none')
        loss = (loss_x + loss_y) * fore_gt

        return loss.mean()

class OrthoLoss(nn.Module):
    def __init__(self, inst_gt, adj_gt, sem_gt, inter_pred, sample_ratio=0.7):
        super(OrthoLoss, self).__init__()
        self.inst_gt = inst_gt
        self.adj_gt = adj_gt
        self.sem_gt = sem_gt
        self.inter_pred = inter_pred
        self.sample_ratio = sample_ratio

    def calculate_ortho_loss(self):
        loss = 0
        for i in range(self.inst_gt.shape[0]):
            inst_gt = self.inst_gt[i]
            adj_gt = self.adj_gt[i]
            sem_gt = self.sem_gt[i]
            inter_pred = self.inter_pred[i]

            inst_list = list(torch.unique(inst_gt))
            if 0 in inst_list:
                inst_list.remove(0)

            for inst_id in inst_list:
                if inst_id.item() not in adj_gt:
                    continue

                neighbors = adj_gt[inst_id.item()]
                if not neighbors:
                    continue

                inst_mask = (inst_gt == inst_id)
                neighbor_mask = torch.zeros_like(inst_gt).bool()
                for neighbor_id in neighbors:
                    neighbor_mask |= (inst_gt == neighbor_id)

                inst_pixels = torch.where(inst_mask)
                neighbor_pixels = torch.where(neighbor_mask)

                if len(inst_pixels[0]) == 0 or len(neighbor_pixels[0]) == 0:
                    continue

                num_inst_pixels = int(len(inst_pixels[0]) * self.sample_ratio)
                num_neighbor_pixels = int(len(neighbor_pixels[0]) * self.sample_ratio)

                if num_inst_pixels == 0 or num_neighbor_pixels == 0:
                    continue

                inst_indices = random.sample(range(len(inst_pixels[0])), num_inst_pixels)
                neighbor_indices = random.sample(range(len(neighbor_pixels[0])), num_neighbor_pixels)

                inst_features = inter_pred[:, inst_pixels[0][inst_indices], inst_pixels[1][inst_indices]]
                neighbor_features = inter_pred[:, neighbor_pixels[0][neighbor_indices], neighbor_pixels[1][neighbor_indices]]

                dot_product = torch.matmul(inst_features.T, neighbor_features)
                ortho_loss = torch.mean(dot_product**2)
                loss += ortho_loss

        return {"ortho_loss": loss / self.inst_gt.shape[0]}


# ==============================================================================
# Part 4: Data Handling and Dataset Class
# ==============================================================================

def get_transform(train=True, height=HEIGHT, width=WIDTH):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height, width),
                A.Normalize(),
                ToTensorV2(),
            ],
            additional_targets={'mask2': 'mask', 'mask3': 'mask'}
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

def draw_graph(adj):
    color_map = {}
    for node in adj.keys():
        neighbor_colors = set(color_map.get(neighbor) for neighbor in adj[node] if neighbor in color_map)
        for color in range(1, 5):
            if color not in neighbor_colors:
                color_map[node] = color
                break
    return color_map

def get_tc_map(inst_map):
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    adj_dict = {inst_id: [] for inst_id in inst_list}
    for inst_id in inst_list:
        inst_mask = (inst_map == inst_id)
        dilated_mask = cv2.dilate(inst_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        neighbors = np.unique(inst_map[dilated_mask > 0])
        for neighbor in neighbors:
            if neighbor != 0 and neighbor != inst_id:
                adj_dict[inst_id].append(neighbor)

    color_map = draw_graph(adj_dict)

    tc_map = np.zeros_like(inst_map, dtype=np.int32)
    for inst_id, color in color_map.items():
        tc_map[inst_map == inst_id] = color

    return tc_map

def get_adj_matrix(inst_map):
    inst_list = list(np.unique(inst_map))
    if 0 in inst_list:
        inst_list.remove(0)

    adj_dict = {inst_id: [] for inst_id in inst_list}
    for inst_id in inst_list:
        inst_mask = (inst_map == inst_id)
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
        img = Image.open(info['image_path']).convert("RGB")
        img_np = np.array(img)

        inst_map = get_inst_map(info['annotations'], 520, 704)
        tc_map = get_tc_map(inst_map)
        adj_matrix = get_adj_matrix(inst_map)
        sem_map = (inst_map > 0).astype(np.uint8)

        if self.transforms:
            augmented = self.transforms(
                image=img_np,
                mask=sem_map,
                mask2=inst_map.astype(np.uint8),
                mask3=tc_map.astype(np.uint8)
            )
            img = augmented['image']
            sem_map, inst_map, tc_map = augmented['mask'], augmented['mask2'], augmented['mask3']
        else:
            img = torch.from_numpy(img_np.transpose(2,0,1)).float()
            sem_map = torch.from_numpy(sem_map).long()
            inst_map = torch.from_numpy(inst_map).long()
            tc_map = torch.from_numpy(tc_map).long()

        target = {
            'sem_gt': sem_map.long(),
            'inst_gt': inst_map.long(),
            'sem_gt_inner': tc_map.long(),
            'adj_gt': adj_matrix,
        }

        return img, target

    def __len__(self):
        return len(self.image_info)


# ==============================================================================
# Part 5: Training Loop
# ==============================================================================

def train_model(model, dl_train, dl_val, device, num_epochs, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
    sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=2)
    cls_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=NUM_CLASSES)

    n_batches, n_batches_val = len(dl_train), len(dl_val)
    validation_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"Starting epoch {epoch} of {num_epochs}")
        time_start = time.time()
        loss_accum = 0.0
        sem_ce_loss_accum, sem_dice_loss_accum = 0.0, 0.0
        cls_ce_loss_accum, cls_dice_loss_accum = 0.0, 0.0
        ortho_loss_accum = 0.0

        for batch_idx, (images, targets) in enumerate(dl_train, 1):
            images = torch.stack(images).to(device)
            sem_pred, inter_pred = model(images)

            sem_gt = torch.stack([t['sem_gt'] for t in targets]).to(device)
            inst_gt = torch.stack([t['inst_gt'] for t in targets]).to(device)
            sem_gt_inner = torch.stack([t['sem_gt_inner'] for t in targets]).to(device)
            adj_gt = [t['adj_gt'] for t in targets]

            sem_ce_loss = torch.mean(sem_ce_loss_calculator(sem_pred, sem_gt))
            sem_dice_loss = sem_dice_loss_calculator(sem_pred, sem_gt)

            cls_ce_loss = torch.mean(sem_ce_loss_calculator(inter_pred, sem_gt_inner) * (sem_gt > 0))
            cls_dice_loss = cls_dice_loss_calculator(inter_pred, sem_gt_inner)

            ortho_loss_calculator = OrthoLoss(inst_gt, adj_gt, sem_gt, inter_pred)
            ortho_loss = ortho_loss_calculator.calculate_ortho_loss()['ortho_loss']

            loss = sem_ce_loss + sem_dice_loss + cls_ce_loss + cls_dice_loss + ortho_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()
            sem_ce_loss_accum += sem_ce_loss.item()
            sem_dice_loss_accum += sem_dice_loss.item()
            cls_ce_loss_accum += cls_ce_loss.item()
            cls_dice_loss_accum += cls_dice_loss.item()
            ortho_loss_accum += ortho_loss.item()

            if batch_idx % 50 == 0:
                print(f"    [Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}")

        train_loss = loss_accum / n_batches
        train_sem_ce_loss = sem_ce_loss_accum / n_batches
        train_sem_dice_loss = sem_dice_loss_accum / n_batches
        train_cls_ce_loss = cls_ce_loss_accum / n_batches
        train_cls_dice_loss = cls_dice_loss_accum / n_batches
        train_ortho_loss = ortho_loss_accum / n_batches

        val_loss_accum = 0
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dl_val, 1):
                images = torch.stack(images).to(device)
                sem_pred, inter_pred = model(images)

                sem_gt = torch.stack([t['sem_gt'] for t in targets]).to(device)
                inst_gt = torch.stack([t['inst_gt'] for t in targets]).to(device)
                sem_gt_inner = torch.stack([t['sem_gt_inner'] for t in targets]).to(device)
                adj_gt = [t['adj_gt'] for t in targets]

                sem_ce_loss = torch.mean(sem_ce_loss_calculator(sem_pred, sem_gt))
                sem_dice_loss = sem_dice_loss_calculator(sem_pred, sem_gt)
                cls_ce_loss = torch.mean(sem_ce_loss_calculator(inter_pred, sem_gt_inner) * (sem_gt > 0))
                cls_dice_loss = cls_dice_loss_calculator(inter_pred, sem_gt_inner)
                ortho_loss_calculator = OrthoLoss(inst_gt, adj_gt, sem_gt, inter_pred)
                ortho_loss = ortho_loss_calculator.calculate_ortho_loss()['ortho_loss']

                val_loss = sem_ce_loss + sem_dice_loss + cls_ce_loss + cls_dice_loss + ortho_loss
                val_loss_accum += val_loss.item()

        val_loss = val_loss_accum / n_batches_val
        elapsed = time.time() - time_start
        validation_losses.append(val_loss)

        torch.save(model.state_dict(), f"fcis_model-e{epoch}.bin")
        prefix = f"[Epoch {epoch:2d} / {num_epochs:2d}]"
        print(prefix)
        print(f"{prefix} Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f} [{elapsed:.0f} secs]")
        print(f"{prefix} Sem CE: {train_sem_ce_loss:7.3f}, Sem Dice: {train_sem_dice_loss:7.3f}")
        print(f"{prefix} Cls CE: {train_cls_ce_loss:7.3f}, Cls Dice: {train_cls_dice_loss:7.3f}")
        print(f"{prefix} Ortho: {train_ortho_loss:7.3f}")
        print(prefix)

    return validation_losses

# ==============================================================================
# Part 6: Inference and Submission
# ==============================================================================

class CellTestDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.image_ids = [fname[:-4] for fname in os.listdir(self.image_dir) if fname.endswith('.png')]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + '.png')
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        if self.transforms:
            augmented = self.transforms(image=img_np)
            img_tensor = augmented['image']
        else:
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img)

        return {'image': img_tensor, 'image_id': image_id}

    def __len__(self):
        return len(self.image_ids)

def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))

def postprocess(pred):
    sem_id_list = list(np.unique(pred))
    inst_pred = np.zeros_like(pred).astype(np.int32)
    sem_pred = np.zeros_like(pred).astype(np.uint8)
    cur = 0
    for sem_id in sem_id_list:
        if sem_id == 0:
            continue
        sem_id_mask = pred == sem_id
        sem_id_mask = binary_fill_holes(sem_id_mask)
        sem_id_mask = remove_small_objects(sem_id_mask, 5)
        inst_sem_mask = measure.label(sem_id_mask)
        inst_sem_mask[inst_sem_mask > 0] += cur
        inst_pred[inst_sem_mask > 0] = 0
        inst_pred += inst_sem_mask
        cur += len(np.unique(inst_sem_mask))
        sem_pred[inst_sem_mask > 0] = sem_id
    return sem_pred, inst_pred

def generate_submission(model, test_loader, device):
    submission = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            sample = batch[0]
            img = sample['image'].to(device)
            image_id = sample['image_id']

            sem_logit, cls_logit = model(img.unsqueeze(0))

            sem_pred = sem_logit.argmax(dim=1)
            cls_pred = cls_logit.argmax(dim=1)
            cls_pred[sem_pred != 1] = 0

            cls_pred = cls_pred.cpu().numpy()[0]

            sem_pred, inst_pred = postprocess(cls_pred)

            if np.sum(inst_pred) == 0:
                submission.append((image_id, "-1"))
            else:
                for i in np.unique(inst_pred):
                    if i == 0:
                        continue
                    mask = (inst_pred == i)
                    rle = rle_encoding(mask)
                    submission.append((image_id, rle))

    df_sub = pd.DataFrame(submission, columns=['id','annotation'])
    df_sub.to_csv("submission.csv", index=False)
    print(df_sub.head())


# ==============================================================================
# Part 7: Visualization
# ==============================================================================

def colorize_seg_map(seg_map, palette=None):
    if palette is None:
        palette = np.random.randint(0, 255, size=(np.max(seg_map) + 1, 3), dtype=np.uint8)
        palette[0] = [0, 0, 0]

    color_map = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for i in np.unique(seg_map):
        color_map[seg_map == i] = palette[i]
    return color_map

def show_test_predictions(model, loader, device, num_samples=3, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), mask_alpha=0.4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= num_samples:
                break
            sample = batch[0]
            img_t = sample['image'].to(device)
            image_id = sample['image_id']

            sem_logit, cls_logit = model(img_t.unsqueeze(0))

            sem_pred = sem_logit.argmax(dim=1)
            cls_pred = cls_logit.argmax(dim=1)
            cls_pred[sem_pred != 1] = 0

            img_disp = img_t.cpu().clone()
            for c in range(3):
                img_disp[c] = img_disp[c] * std[c] + mean[c]
            img_np = img_disp.permute(1,2,0).numpy()
            img_np = np.clip(img_np, 0, 1)

            ax_img = axes[idx, 0]
            ax_img.imshow(img_np)
            ax_img.set_title(f"Image: {image_id}")
            ax_img.axis('off')

            ax_4color = axes[idx, 1]
            color_map = colorize_seg_map(cls_pred.cpu().numpy()[0])
            ax_4color.imshow(img_np)
            ax_4color.imshow(color_map, alpha=mask_alpha)
            ax_4color.set_title("4-Color Prediction")
            ax_4color.axis('off')

            ax_pred = axes[idx, 2]
            _, inst_pred = postprocess(cls_pred.cpu().numpy()[0])
            inst_map = colorize_seg_map(inst_pred)
            ax_pred.imshow(img_np)
            ax_pred.imshow(inst_map, alpha=mask_alpha, cmap='cividis')
            ax_pred.set_title("Predicted Mask")
            ax_pred.axis('off')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Part 8: Main Execution
# ==============================================================================

if __name__ == '__main__':
    # 1. Load data
    df_base = pd.read_csv(TRAIN_CSV)
    unique_ids = df_base['id'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=PCT_IMAGES_VALIDATION, random_state=42, shuffle=True)
    df_train = df_base[df_base['id'].isin(train_ids)]
    df_val = df_base[df_base['id'].isin(val_ids)]

    # 2. Create datasets and dataloaders
    ds_train = CellDataset(TRAIN_PATH, df_train, transforms=get_transform(train=True))
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    ds_val = CellDataset(TRAIN_PATH, df_val, transforms=get_transform(train=False))
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

    # 3. Initialize model
    model = FCISNet(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # 4. Train the model
    validation_losses = train_model(model, dl_train, dl_val, DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)

    # 5. Load the best model
    best_epoch = np.argmin(validation_losses) + 1
    model.load_state_dict(torch.load(f"fcis_model-e{best_epoch}.bin"))
    model.to(DEVICE)

    # 6. Create test dataset and dataloader
    ds_test = CellTestDataset(TEST_PATH, transforms=get_transform(train=False))
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x)

    # 7. Generate submission file
    generate_submission(model, test_loader, DEVICE)

    # 8. Visualize test predictions
    show_test_predictions(model, test_loader, DEVICE, num_samples=3)
