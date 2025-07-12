import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from skimage import measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Assuming model, TEST_PATH, DEVICE, MASK_THRESHOLD are defined

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

def get_transform(train=True, height=256, width=256):
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

ds_test = CellTestDataset(TEST_PATH, transforms=get_transform(train=False))
test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x)

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
    """model free post-process for both instance-level & semantic-level."""
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


submission = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference"):
        sample = batch[0]
        img = sample['image'].to(DEVICE)
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
