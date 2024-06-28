import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_avg = 0
    iou_avg = 0
    auc_avg = 0
    precision_avg = 0
    recall_avg = 0
    confusion_matrices = np.zeros((net.n_classes, net.n_classes), dtype=int)
    # jaccard_coeff_multi_scores = []

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_avg += dice_coeff(mask_pred, mask_true.float(), reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                mask_true = mask_true.squeeze(1)
                # convert to one-hot format
                mask_true_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_avg += multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_true_one_hot[:, 1:], reduce_batch_first=False)

            # Calculate IoU
            mask_pred_softmax = F.softmax(mask_pred, dim=1)
            iou = calculate_iou(mask_pred_softmax, mask_true, net.n_classes)
            iou_avg += iou

            # Flatten and convert tensors to numpy arrays for calculating other metrics
            mask_pred_np = mask_pred.argmax(dim=1).flatten().cpu().numpy()
            mask_true_np = mask_true.flatten().cpu().numpy()

            # Calculate AUC
            try:
                auc = roc_auc_score(mask_true_one_hot.cpu().numpy().reshape(-1), mask_pred_softmax.cpu().numpy().reshape(-1), average='macro', multi_class='ovr')
                auc_avg += auc
            except ValueError as e:
                print(f"Skipping AUC calculation: {e}")
                auc = None  # or an appropriate placeholder value

            precision = precision_score(mask_true_np, mask_pred_np, average='weighted', zero_division=1)
            recall = recall_score(mask_true_np, mask_pred_np, average='weighted', zero_division=1)
            confusion = confusion_matrix(mask_true_np, mask_pred_np, labels=[i for i in range(net.n_classes)])
           
            precision_avg += precision
            recall_avg += recall
            confusion_matrices += confusion

    net.train()

    dice_avg = dice_avg / max(num_val_batches, 1)
    iou_avg = iou_avg / max(num_val_batches, 1)
    auc_avg = auc_avg / max(num_val_batches, 1)
    precision_avg = precision_avg / max(num_val_batches, 1)
    recall_avg = recall_avg / max(num_val_batches, 1)

    return dice_avg.item(), iou_avg, auc_avg, precision_avg, recall_avg, confusion_matrices
    # return dice_score / num_val_batches, iou_scores, auc_scores, precision_scores, recall_scores, confusion_matrices


def calculate_iou(preds, targs, n_classes, eps=1e-8):
    """
    Calculate IoU for multi-class segmentation tasks.
    preds: Predictions from the model, expected to be logits or probabilities after softmax [N, C, H, W]
    targs: Ground truth labels [N, H, W]
    n_classes: Number of classes
    eps: A small value to prevent division by zero
    """
    # Convert predictions to class indices [N, H, W]
    preds = torch.argmax(preds, dim=1)
    
    # Initialize IoU for each class
    iou_per_class = torch.zeros(n_classes, device=preds.device)
    
    for cls in range(n_classes):
        # True Positives: Predictions and targets both are class 'cls'
        true_positive = ((preds == cls) & (targs == cls)).sum(dim=[1, 2])
        # False Positives: Predictions are class 'cls' but targets are not
        false_positive = ((preds == cls) & (targs != cls)).sum(dim=[1, 2])
        # False Negatives: Targets are class 'cls' but predictions are not
        false_negative = ((preds != cls) & (targs == cls)).sum(dim=[1, 2])
        
        # Calculate IoU for class 'cls'
        iou = true_positive / (true_positive + false_positive + false_negative + eps)
        iou_per_class[cls] = iou.mean()
    
    # Return the average IoU across all classes
    return iou_per_class.mean().item()

