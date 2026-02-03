import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_iou(preds, labels, num_classes):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0

def compute_pixel_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == labels).sum().item()
    total = torch.numel(labels)
    return correct / total

def compute_f1_score(preds, labels, num_classes):
    preds = torch.argmax(preds, dim=1)
    preds_np = preds.view(-1).cpu().numpy()
    labels_np = labels.view(-1).cpu().numpy()
    return f1_score(labels_np, preds_np, average='macro', labels=list(range(num_classes)), zero_division=0)

def compute_metrics(outputs, labels, num_classes):
    iou = compute_iou(outputs, labels, num_classes)
    acc = compute_pixel_accuracy(outputs, labels)
    f1 = compute_f1_score(outputs, labels, num_classes)
    return {
        "IoU": iou,
        "Pixel Accuracy": acc,
        "F1 Score": f1
    }