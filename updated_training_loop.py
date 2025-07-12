import torch
import torch.nn as nn
from losses import BatchMultiClassDiceLoss, OrthoLoss
import time

# Assuming model, dl_train, dl_val, DEVICE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY are defined

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=2)
cls_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=5)

n_batches, n_batches_val = len(dl_train), len(dl_val)

validation_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Starting epoch {epoch} of {NUM_EPOCHS}")

    time_start = time.time()
    loss_accum = 0.0
    sem_ce_loss_accum = 0.0
    sem_dice_loss_accum = 0.0
    cls_ce_loss_accum = 0.0
    cls_dice_loss_accum = 0.0
    ortho_loss_accum = 0.0


    for batch_idx, (images, targets) in enumerate(dl_train, 1):

        # Predict
        images = list(image.to(DEVICE) for image in images)
        images = torch.stack(images)

        sem_pred, inter_pred = model(images)

        # Move targets to device
        sem_gt = torch.stack([t['sem_gt'] for t in targets]).to(DEVICE)
        inst_gt = torch.stack([t['inst_gt'] for t in targets]).to(DEVICE)
        sem_gt_inner = torch.stack([t['sem_gt_inner'] for t in targets]).to(DEVICE)
        adj_gt = [t['adj_gt'] for t in targets] # list of dicts

        # Calculate losses
        sem_ce_loss = torch.mean(sem_ce_loss_calculator(sem_pred, sem_gt))
        sem_dice_loss = sem_dice_loss_calculator(sem_pred, sem_gt)

        cls_ce_loss = torch.mean(sem_ce_loss_calculator(inter_pred, sem_gt_inner) * (sem_gt > 0))
        cls_dice_loss = cls_dice_loss_calculator(inter_pred, sem_gt_inner)

        ortho_loss_calculator = OrthoLoss(inst_gt, adj_gt, sem_gt, inter_pred)
        ortho_loss = ortho_loss_calculator.calculate_ortho_loss()['ortho_loss']

        loss = sem_ce_loss + sem_dice_loss + cls_ce_loss + cls_dice_loss + ortho_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        loss_accum += loss.item()
        sem_ce_loss_accum += sem_ce_loss.item()
        sem_dice_loss_accum += sem_dice_loss.item()
        cls_ce_loss_accum += cls_ce_loss.item()
        cls_dice_loss_accum += cls_dice_loss.item()
        ortho_loss_accum += ortho_loss.item()

        if batch_idx % 50 == 0:
            print(f"    [Batch {batch_idx:3d} / {n_batches:3d}] Batch train loss: {loss.item():7.3f}")


    # Train losses
    train_loss = loss_accum / n_batches
    train_sem_ce_loss = sem_ce_loss_accum / n_batches
    train_sem_dice_loss = sem_dice_loss_accum / n_batches
    train_cls_ce_loss = cls_ce_loss_accum / n_batches
    train_cls_dice_loss = cls_dice_loss_accum / n_batches
    train_ortho_loss = ortho_loss_accum / n_batches

    # Validation
    val_loss_accum = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dl_val, 1):
            images = list(image.to(DEVICE) for image in images)
            images = torch.stack(images)

            sem_pred, inter_pred = model(images)

            sem_gt = torch.stack([t['sem_gt'] for t in targets]).to(DEVICE)
            inst_gt = torch.stack([t['inst_gt'] for t in targets]).to(DEVICE)
            sem_gt_inner = torch.stack([t['sem_gt_inner'] for t in targets]).to(DEVICE)
            adj_gt = [t['adj_gt'] for t in targets]

            sem_ce_loss = torch.mean(sem_ce_loss_calculator(sem_pred, sem_gt))
            sem_dice_loss = sem_dice_loss_calculator(sem_pred, sem_gt)

            cls_ce_loss = torch.mean(sem_ce_loss_calculator(inter_pred, sem_gt_inner) * (sem_gt > 0))
            cls_dice_loss = cls_dice_loss_calculator(inter_pred, sem_gt_inner)

            ortho_loss_calculator = OrthoLoss(inst_gt, adj_gt, sem_gt, inter_pred)
            ortho_loss = ortho_loss_calculator.calculate_ortho_loss()['ortho_loss']

            val_loss = sem_ce_loss + sem_dice_loss + cls_ce_loss + cls_dice_loss + ortho_loss
            val_loss_accum += val_loss.item()

    # Validation losses
    val_loss = val_loss_accum / n_batches_val
    elapsed = time.time() - time_start

    validation_losses.append(val_loss)

    torch.save(model.state_dict(), f"fcis_model-e{epoch}.bin")
    prefix = f"[Epoch {epoch:2d} / {NUM_EPOCHS:2d}]"
    print(prefix)
    print(f"{prefix} Train loss: {train_loss:7.3f}. Val loss: {val_loss:7.3f} [{elapsed:.0f} secs]")
    print(f"{prefix} Sem CE Loss: {train_sem_ce_loss:7.3f}, Sem Dice Loss: {train_sem_dice_loss:7.3f}")
    print(f"{prefix} Cls CE Loss: {train_cls_ce_loss:7.3f}, Cls Dice Loss: {train_cls_dice_loss:7.3f}")
    print(f"{prefix} Ortho Loss: {train_ortho_loss:7.3f}")
    print(prefix)
