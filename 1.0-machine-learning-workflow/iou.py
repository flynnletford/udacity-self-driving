import numpy as np

from utils import get_data, check_results


def calculate_ious(gt_bboxes, pred_bboxes):
    """
    calculate ious between 2 sets of bboxes 
    args:
    - gt_bboxes [array]: Nx4 ground truth array
    - pred_bboxes [array]: Mx4 pred array
    returns:
    - iou [array]: NxM array of ious
    """
    ious = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i,j] = calculate_iou(gt_bbox, pred_bbox)
    return ious

def calculate_intersection_area(gt_bbox, pred_bbox):   
    gx0 = gt_bbox[0]
    gy0 = gt_bbox[1]
    gx1 = gt_bbox[2]
    gy1 = gt_bbox[3]

    px0 = pred_bbox[0]
    py0 = pred_bbox[1]
    px1 = pred_bbox[2]
    py1 = pred_bbox[3]

    left = max(gx0, px0)
    right = min(gx1, px1)
    bottom = max(gy0, py0)
    top = min(gy1, py1)

    width = right - left
    height = top - bottom

    if width <= 0 or height <= 0:
        return 0.0
    
    intersection = width * height

    return intersection

def calculate_union(gt_bbox, pred_bbox, intersection_area):
    gt_width = gt_bbox[2] - gt_bbox[0]
    gt_height = gt_bbox[3] - gt_bbox[1]

    pred_width = pred_bbox[2] - pred_bbox[0]
    pred_height = pred_bbox[3] - pred_bbox[1]

    gt_area = gt_width * gt_height
    pred_area = pred_width * pred_height

    union = gt_area + pred_area - intersection_area

    return union

def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """

    intersection_area = calculate_intersection_area(gt_bbox, pred_bbox)
    union = calculate_union(gt_bbox, pred_bbox, intersection_area)

    iou = intersection_area / union

    return iou


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    check_results(ious)