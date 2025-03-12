import numpy as np

def compute_iou_per_class(pred_mask, gt_mask, num_classes):
    """Calculate per-class IoU for a single predicted mask vs. a single ground-truth mask.

    Args:
        pred_mask (np.ndarray): Predicted segmentation mask.
        gt_mask (np.ndarray): Ground-truth segmentation mask.
        num_classes (int): Total number of segmentation classes.

    Returns:
        dict: Dictionary mapping class_id to IoU value.
    """
    iou_dict = {}
    for cls_id in range(num_classes):
        pred_cls = (pred_mask == cls_id).astype(np.uint8)
        gt_cls = (gt_mask == cls_id).astype(np.uint8)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        iou_dict[cls_id] = (intersection / union) if union else 0.0
    return iou_dict


def compute_instance_iou(pred_mask, gt_mask, instance_id):
    """Calculate IoU for a *single ground-truth instance* vs. predicted mask.

      pred_mask, gt_mask: same shape (binary or instance-labeled)
      instance_id       : the ground-truth label ID we care about
    Returns: IoU (float)
    """
    # Extract the ground-truth region for 'instance_id'
    gt_instance = (gt_mask == instance_id).astype(np.uint8)

    # If your prediction is binary (foreground=1, background=0), do (pred_mask > 0).
    # If you have instance-labeled predictions, do (pred_mask == instance_id) instead.
    pred_instance = (pred_mask > 0).astype(np.uint8)

    intersection = np.logical_and(gt_instance, pred_instance).sum()
    union = np.logical_or(gt_instance, pred_instance).sum()

    return float(intersection) / union if union else 0.0