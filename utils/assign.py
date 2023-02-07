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

def box_iox_xyxy(box1, box2):
    """
    box1: nx4
    box2: mx4
    """
    area1 = (box1[:, 3]-box1[:, 1])*(box1[:, 2]-box1[:, 0])
    area2 = (box2[:, 3]-box2[:, 1])*(box2[:, 2]-box2[:, 0])
    box1 = box1[:, None, :]
    box2 = box2[None, :, :]
    x1 = np.maximum(box1[:, 0], box2[:, 0])
    y1 = np.maximum(box1[:, 1], box2[:, 1])
    x2 = np.minimum(box1[:, 2], box2[:, 2])
    y2 = np.minimum(box1[:, 3], box2[:, 3])
    inter_w = np.maximum(x2 - x1, 0)
    inter_h = np.maximum(y2 - y1, 0)
    overlaps = inter_w * inter_h
    union =  area1[:, None] + area2[None, :] - inter_w * inter_h + 1e-9
    return overlaps/union


def grid_assign(gt_boxes, gt_labels, anchors, box_responsible_flags, pos_iou_thr = 0.5, neg_iou_thr = 0.3, min_pos_iou=0.0):
    """
    step 1: 计算gt_boxes与anchor的overlaps
    step 2: 首先将所有的anchor的筛选值赋值为-1（即忽略该anchor）
    step 3: 计算每一个anchor与所有gt_boxes的iou，将iou小于neg_iou_thr的anchor认定为负样本，赋值为0
    step 4: 对于每一个box_responsible_flags为true的anchor，其最大的iou大于pos_iou_thr被认为是正样本
    step 5: 对于每一个gt, 计算其与box_responsible_flags为true的所有anchor的overlap,并筛选出最大值，当最大值大于min_pos_iou，将这个gt赋值给anchor
    step 6: 对于上述step4至step5的操作所获得的正样本赋予对应的gt_boxes的标签
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
    gt_max_overlaps = max(overlaps, axis=1)
    gt_max_overlaps_index = np.argmax(overlaps, axis=1)
    # row_index = np.arange(len(gt_boxes), dtype=int)
    pos_inds = box_responsible_flags & (max_overlaps > pos_iou_thr)
    assigned_gt_inds[pos_inds] = max_overlaps_index[pos_inds] + 1


    # step 4 对于每一个gt,计算与之具有最大overlap的anchor，且其overlap大于min_pos_iou，将其赋值给anchor
    for i in range(num_gts):
        max_iou = gt_max_overlaps[i]
        anchor_id = gt_max_overlaps_index[i]
        if(max_iou>min_pos_iou):
            if(box_responsible_flags[anchor_id]):
                assigned_gt_inds[anchor_id] = i + 1

    pos_inds = assigned_gt_inds > 0
    assigned_gt_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds]-1]
    
    return assigned_gt_inds, assigned_gt_labels

def is_in_targetbox(boxes, target_boxes):
    """
    判断boxes的中心是否在target_boxes中
    """
    cnt_x_boxes, cnt_y_boxes = (boxes[:, 2] + boxes[:, 0]) / 2.0, (boxes[:, 3] + boxes[:, 1]) / 2.0
    tl_x = cnt_x_boxes[:, None] - target_boxes[:, 0][None, :]
    tl_y = cnt_y_boxes[:, None] - target_boxes[:, 1][None, :]
    br_x = target_boxes[:, 2][None, :] - cnt_x_boxes[:, None] 
    br_y = target_boxes[:, 3][None, :] - cnt_y_boxes[:, None]
    is_in_target = (tl_x > 0)&(tl_y > 0)&(br_x > 0)&(br_y > 0)
    return is_in_target




def Single_TaskAlignedAssigner(pred_bboxes, pred_cls, gt_boxes, cls_ids, overlap_beta=6, score_alpha=1, topk=3):
    """
    args:
    step 1: 计算预测的pred_bboxes与 gt_boxes之间的overlaps
    step 2: 根据gt_boxes的标签cls_ids，计算预测的bboxes对每一类gt_boxes的分类预测分数cls_scores
    step 3: 计算预测的pred_bboxes与gt_boxes之间的度量函数alignment_metrics，度量函数定义为分类预测分数的指数与overlap指数的乘积
    step 4: 基于度量函数alignment_metrics，对每一个gt_boxes选择它的topk 预测的bboxes,作为它的候选框
    step 5: 对于step 4得到的候选框，如果他的中心没有落在其绑定的gt的范围内，则将其滤除
    step 6: 对于step 5得到的候选框，如果某一候选框与多个gt_boxes绑定，则选择度量函数最大的那一个进行绑定    
    pred_bboxes: nx4
    pred_cls: nxc
    gt_bboxes: mx4
    cls_ids: mx1
    returns:
    candicate_ids nx1
    """
    num_priors, _ = pred_bboxes.shape
    num_gts = len(cls_ids)
    candicate_ids = -1 * np.ones(shape=(num_priors)) # nx1
    candicate_matrix = -1 * np.ones(shape=(num_priors, num_gts))
    overlaps = box_iox_xyxy(pred_bboxes, gt_boxes) # nxm
    cls_scores = pred_cls[:, cls_ids] # nxm
    alignment_metrics = cls_scores**score_alpha*overlaps**overlap_beta # nxm step 3
    # step 4
    ind = np.argsort(alignment_metrics, axis=0)
    ind = np.take(ind, np.arange(topk), axis=0)
    x_ind = np.repeat(np.range(topk), num_gts)
    candicate_matrix[x_ind, ind] = 1 
    is_in_target = is_in_targetbox(pred_bboxes, gt_boxes)
    candicate_matrix[~is_in_target] = -1 # step 5
    # step 6
    sum_candicate_matrix = (candicate_matrix==1).sum(axis=1)>1
    sub_candicate_matrix = candicate_matrix[sum_candicate_matrix, :]
    index = np.argmax(sub_candicate_matrix, axis=1)
    candicate_matrix[sum_candicate_matrix, :] = -1
    candicate_matrix[sum_candicate_matrix, index] = 1
    bbox_ids, gt_box_ids = np.argwhere(candicate_matrix==1) 
    candicate_ids[bbox_ids] = gt_box_ids
    return candicate_ids



def TaskAlignedAssigner(preds, gts):
    """
    step 1: 计算预测的bboxes 与 gts之间的overlaps
    step 2: 根据gts的标签，计算预测的bboxes对每一类gts的分类预测分数score
    step 3: 计算预测的bboxes与gts之间的度量函数，度量函数定义为分类预测分数的指数与overlap指数的乘积
    step 4: 基于度量函数，对每一个gt选择它的topk 预测的bboxes,作为它的候选框
    step 5: 对于step 4得到的候选框，如果他的中心没有落在其绑定的gt的范围内，则将其滤除
    step 6: 对于step 5得到的候选框，如果某一候选框与多个gt绑定，则选择度量函数最大的那一个进行绑定
    args:
    preds: list {anchor: batch*H*W*Nx4, boxes_type: xyxy(default) or xywh, preds_bboxes:batch*H*W*Nx4, cls_scores: batch*H*W*Nxcls_num}
    gts: list {gt_boxes: num_gts*4, boxes_type: boxes_type: xyxy(default) or xywh, cls_ids: [num_gts], img_ids: [num_gts]}
    """
    # batch_size, anchor_nums = preds.shape
    pred_bboxes = preds['preds_bboxes']
    cls_scores = preds['cls_scores']
    gt_boxes = gts['gt_boxes']
    cls_ids = gts['cls_ids']


def SimOTAAssigner():
    """
    step 1: 计算预测框bboxes
    step 2:
    step 3:
    """




   





    










    




