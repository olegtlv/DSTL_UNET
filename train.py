import torch
import torch.nn as nn
import torch.optim as optim
import wandb
device = 'cuda'
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from tools import show_one_image_with_pred
from training_tools import DiceLoss

# Define loss function and optimizer
criterion_BCE = nn.BCELoss()
criterion_Dice = DiceLoss()
scaler = GradScaler()


def train_one_epoch(model, dataloader, optimizer, combined_loss, epoch, train_loss_hist_batch, optimizer_hist_batch, scaler):
    model.train()
    total_loss = 0
    num_batches = 0
    batch_num = 0

    for batch in dataloader:
        inputs = batch['image'].to('cuda')
        targets = batch['mask'].to('cuda')

        optimizer.zero_grad()

        with autocast(device_type='cuda'):
            outputs = model(inputs)
            if isinstance(outputs, tuple):  # if model returns extra outputs
                outputs = outputs[0]
            loss_comb = combined_loss(outputs, targets, epoch)

        scaler.scale(loss_comb).backward()         # ✅ Backward pass
        scaler.step(optimizer)                     # ✅ Optimizer step
        scaler.update()                            # ✅ Update the scale

        train_loss_hist_batch.append(loss_comb.item())
        optimizer_hist_batch.append(optimizer.param_groups[0]['lr'])

        if batch_num % 25 == 0:
            print("batch_num: ", batch_num)
            print("    batch loss: %.4f" % float(loss_comb))
            print('    LR ENC: ', optimizer.param_groups[0]['lr'],
                  '    LR_DEC: ', optimizer.param_groups[6]['lr'])

        total_loss += loss_comb.item()
        batch_num += 1
        num_batches += 1

        wandb.log({
            "train/batch_loss": loss_comb.item(),
            "train/lr_encoder": optimizer.param_groups[0]['lr'],
            "train/lr_decoder": optimizer.param_groups[6]['lr'],
            "train/batch_num": batch_num + 1,
            "epoch": epoch
        })

    epoch_loss = total_loss / num_batches
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch})

    return epoch_loss


def calc_validation_loss_one_epoch(model, dataloader, combined_loss, epoch):
    model.eval()
    num_batches_test = 0
    total_loss_test = 0
    #     total_loss_test_BCE = 0
    #     total_loss_test_Dice = 0

    with torch.no_grad():
        for batch in dataloader:
            num_batches_test += 1
            inputs_test, targets_test = batch['image'].to(device), batch['mask'].to(device)
            # outputs_test, _, _ = model(inputs_test)
            outputs_test = model(inputs_test)
            loss_test = combined_loss(outputs_test, targets_test, epoch)
            total_loss_test += loss_test.item()

    epoch_loss_test = total_loss_test / num_batches_test

    # Log validation loss
    wandb.log({"val/epoch_loss": epoch_loss_test, "epoch": epoch})

    # Log prediction image once per epoch
    img = show_one_image_with_pred(inputs_test, outputs_test, targets_test, return_img=True)
    wandb.log({"val/prediction_example": wandb.Image(img), "epoch": epoch})

    return epoch_loss_test


