import numpy as np

def box_iou_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0
    x1max, y1max = box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0
    x2max, y2max = box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    iou = intersection / union
    return iou 



def assign(gt_boxes, gt_labels, anchors, box_responsible_flags, pos_iou_thr = 0.5, neg_iou_thr = 0.3):
    """
    step 1: 计算gt_boxes与anchor的overlaps
    step 2: 筛选出

    """
    num_gts = len(gt_boxes)
    assigned_gt_inds = -1 * np.ones(shape=(len(anchors)))
    assigned_gt_labels = -1 * np.ones(shape=(len(anchors)))
    overlaps = box_iou_xywh(gt_boxes, anchors)
    max_overlaps = max(overlaps, axis=0)
    max_overlaps_index = np.argmax(overlaps, axis=0)
    # step 1 计算负样本
    neg_inds = max_overlaps < neg_iou_thr
    assigned_gt_inds[neg_inds] = 0
    # step 2 计算正样本
    overlaps[:,  ~box_responsible_flags] = -1.
    max_overlaps = max(overlaps, axis=0)
    max_overlaps_index = np.argmax(overlaps, axis=0)
    # row_index = np.arange(len(gt_boxes), dtype=int)
    pos_inds = box_responsible_flags & (max_overlaps > pos_iou_thr)
    assigned_gt_inds[pos_inds] = max_overlaps_index[pod_inds] + 1


    # step 4 对于每一个gt,计算与之具有最大overlap的anchor，且其overlap大于min_pos_iou，将其赋值给anchor
    for i in range(num_gts):
        





    assigned_gt_labels[]





    








    pod_inds = box_responsible_flags[max_overlaps_index] & (max_overlaps > pos_iou_thr)
    assigned_gt_inds[pod_inds] = max_overlaps_index[pod_inds] + 1

    




