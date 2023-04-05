import numpy as np


def iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) score between two binary masks.

    Args:
        pred_mask: Predicted binary segmentation mask as a numpy array.
        true_mask: Ground truth binary segmentation mask as a numpy array.

    Returns:
        IoU score as a float value between 0 and 1.
    """
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def dice(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Calculate Dice coefficient score between two binary masks.

    Args:
        pred_mask: Predicted binary segmentation mask as a numpy array.
        true_mask: Ground truth binary segmentation mask as a numpy array.

    Returns:
        Dice coefficient score as a float value between 0 and 1.
    """
    intersection = np.logical_and(pred_mask, true_mask)
    dice_score = (2 * np.sum(intersection)) / (np.sum(pred_mask) + np.sum(true_mask))
    return dice_score
