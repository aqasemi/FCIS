import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import math
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Assuming model, test_loader, DEVICE, MASK_THRESHOLD are defined

def colorize_seg_map(seg_map, palette=None):
    if palette is None:
        # Generate a random color palette
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

            # Forward pass
            sem_logit, cls_logit = model(img_t.unsqueeze(0))

            sem_pred = sem_logit.argmax(dim=1)
            cls_pred = cls_logit.argmax(dim=1)
            cls_pred[sem_pred != 1] = 0

            # Un-normalize for display
            img_disp = img_t.cpu().clone()
            for c in range(3):
                img_disp[c] = img_disp[c] * std[c] + mean[c]
            img_np = img_disp.permute(1,2,0).numpy()
            img_np = np.clip(img_np, 0, 1)

            # Plot original
            ax_img = axes[idx, 0]
            ax_img.imshow(img_np)
            ax_img.set_title(f"Image: {image_id}")
            ax_img.axis('off')

            # Plot 4-color prediction
            ax_4color = axes[idx, 1]
            color_map = colorize_seg_map(cls_pred.cpu().numpy()[0])
            ax_4color.imshow(img_np)
            ax_4color.imshow(color_map, alpha=mask_alpha)
            ax_4color.set_title("4-Color Prediction")
            ax_4color.axis('off')

            # Plot final instance segmentation
            ax_pred = axes[idx, 2]
            _, inst_pred = postprocess(cls_pred.cpu().numpy()[0])
            inst_map = colorize_seg_map(inst_pred)
            ax_pred.imshow(img_np)
            ax_pred.imshow(inst_map, alpha=mask_alpha, cmap='cividis')
            ax_pred.set_title("Predicted Mask")
            ax_pred.axis('off')

    plt.tight_layout()
    plt.show()

# You will need the postprocess function from the inference script
def postprocess(pred):
    """model free post-process for both instance-level & semantic-level."""
    from skimage import measure
    from skimage.morphology import remove_small_objects
    from scipy.ndimage import binary_fill_holes
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

show_test_predictions(model, test_loader, DEVICE, num_samples=3)
