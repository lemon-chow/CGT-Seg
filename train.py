import argparse
import logging
import os
import random
import csv
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import numpy as np
import pandas as pd

# import wandb
from evaluate import evaluate
from unet import UNet, UResnet34, UResnet50, UResnet101, UResnet152, SEUNet
from utils.data_loading import BasicDataset, LungDataset
from utils.dice_score import dice_loss

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

# dir_img = Path('./data/2020-TMI-InfNet-Dataset638/image')
# dir_mask = Path('./data/2020-TMI-InfNet-Dataset638/object_gt')
dir_img = Path('./data/image_1')
dir_mask = Path('./data/mask')
dir_checkpoint = Path('./checkpoints/')
experiment_result = Path('./results/')  # new

# def calculate_iou(pred, target):
#     intersection = np.logical_and(target, pred)
#     union = np.logical_or(target, pred)
#     iou_score = np.sum(intersection) / np.sum(union)
#     return iou_score
#
#
# def calculate_pr_curve(pred, target):
#     precision, recall, _ = precision_recall_curve(target.ravel(), pred.ravel())
#     pr_auc = auc(recall, precision)
#     return precision, recall, pr_auc
#
#
# def calculate_confusion_matrix(pred, traget, normalize=False):
#     cm = confusion_matrix(traget.ravel(), pred.ravel())
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     return cm


# val_iou_scores = []
# val_precisions = []
# val_recalls = []
# def my_collate_fn(batch):
#     images = [item['image'] for item in batch]
#     masks = [item['mask'] for item in batch]
#     images = torch.stack(images, dim=0)
#     masks = torch.stack(masks, dim=0)
#     return {'image': images, 'mask': masks}


def train_model(
        model,
        device,
        epochs: int = 8,
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        debug = False,
        loss_weight=[1.0, 2.6],
        classes_weight=[1.0, 1.7, 2.4, 2.8]
):
    # 1. Create dataset
    try:
        dataset = LungDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    if debug:
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        subset_indices = indices[:102]
        dataset = Subset(dataset, subset_indices)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    results_file = experiment_result / 'training_results.csv'
    # ################
    #     with open(results_file, 'w', newline='') as file:
    #         writer = csv.writer(file)
    #         # 添加列名以记录准确率
    #         writer.writerow(
    #             ['epoch', 'train_loss', 'train_accuracy', 'validation_dice', 'iou_score', 'pr_auc', 'precision', 'recall',
    #              'confusion'])

    ############
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    classes_weight = torch.tensor(classes_weight, dtype=torch.float32).to(device)    
    criterion = nn.CrossEntropyLoss(classes_weight) if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0

    results_df = pd.DataFrame(
        columns=['epoch', 'epoch_loss', 'train_accuracy', 'val_score', 'iou_score', 'pr_auc',
                 'precision', 'recall', 'confusion'])

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        train_accuracy = 0
        val_score = []
        iou_scores = []
        pr_aucs = []
        precisions = []
        recalls = []
        # confusion = np.zeros((2, 2), dtype=int)
        confusion = np.zeros((model.n_classes, model.n_classes), dtype=int)
        num = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                num = num + 1
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_masks = true_masks.squeeze(1)  # Squeeze out the channel dimension

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # loss1
                        weighted_cross_entropy = criterion(masks_pred, true_masks) * loss_weight[0]

                        # loss2
                        dice = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        ) * loss_weight[1]
                        loss = weighted_cross_entropy + dice

                        print(weighted_cross_entropy)
                        print(dice)


                if model.n_classes == 1:
                    pred = torch.sigmoid(masks_pred) > 0.5
                else:
                    pred = masks_pred.argmax(dim=1)
                correct += (pred == true_masks).sum().item()
                total += true_masks.numel()

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                train_accuracy_0 = correct / total
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'train_accuracy': train_accuracy_0,
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                train_accuracy += train_accuracy_0

                # Evaluation round
                evaluate_times = 3
                division_step = (n_train // (evaluate_times * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score_0, iou_score_0, pr_auc_0, precision_0, recall_0, confusion_0 = evaluate(model, val_loader, device, amp)

                        scheduler.step(val_score_0)

                        val_score.append(val_score_0)
                        iou_scores.append(iou_score_0)
                        pr_aucs.append(pr_auc_0)
                        precisions.append(precision_0)
                        recalls.append(recall_0)
                        confusion += confusion_0

            train_accuracy /= num
            val_score = np.mean(val_score)
            iou_score_avg = np.mean(iou_scores)
            pr_auc_avg = np.mean(pr_aucs)
            precision_avg = np.mean(precisions)
            recall_avg = np.mean(recalls)
            # confusion /= num
            epoch_loss /= num

            results_df = pd.concat([results_df, pd.DataFrame({
                                'epoch': epoch,
                                'epoch_loss': epoch_loss,
                                'train_accuracy': train_accuracy,
                                'val_score': val_score,
                                'iou_score': iou_score_avg,
                                'pr_auc': pr_auc_avg,
                                'precision': precision_avg,
                                'recall': recall_avg,
                                'confusion': str(confusion)
            }, index=[0])])

            # 将DataFrame保存为CSV文件
            results_df.to_csv(results_file, index=False)

            # # 将 DataFrame 存储为 .npy 文件
            # results_np = results_df.to_numpy()  # 将 DataFrame 转换为 NumPy 数组
            # np.save(results_file, results_np)

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            if not debug:
                state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=8, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--debug', '-d', type=bool, default=False, help='Debug mode')
    parser.add_argument('--loss-weight', '-lw', nargs='+', type=float, help="List of loss weights")
    parser.add_argument('--classes-weight', '-cw', nargs='+', type=float, help="List of class weights")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = UResnet34(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # model = SEUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            debug=args.debug,
            loss_weight=args.loss_weight,
            classes_weight=args.classes_weight            
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            debug=args.debug,
            loss_weight=args.loss_weight,
            classes_weight=args.classes_weight   
        )