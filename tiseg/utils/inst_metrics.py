from collections import OrderedDict

import numpy as np
from skimage import measure
from scipy.optimize import linear_sum_assignment

from .misc import get_bounding_box


def pre_eval_bin_aji(inst_pred, inst_gt):
    # make instance id contiguous
    inst_pred = measure.label(inst_pred.copy())
    inst_gt = measure.label(inst_gt.copy())

    pred_id_list = list(np.unique(inst_pred))
    gt_id_list = list(np.unique(inst_gt))

    if 0 not in pred_id_list:
        pred_id_list.insert(0, 0)

    if 0 not in gt_id_list:
        gt_id_list.insert(0, 0)

    # Remove background class
    pred_masks = []
    for p in pred_id_list:
        p_mask = (inst_pred == p).astype(np.uint8)
        pred_masks.append(p_mask)

    gt_masks = []
    for g in gt_id_list:
        g_mask = (inst_gt == g).astype(np.uint8)
        gt_masks.append(g_mask)

    # prefill with value
    pairwise_intersection = np.zeros([len(gt_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)
    pairwise_union = np.zeros([len(gt_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise
    for gt_id in gt_id_list:  # 0-th is background
        if gt_id == 0:
            continue
        g_mask = gt_masks[gt_id]
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(g_mask)
        g_mask_crop = g_mask[rmin1:rmax1, cmin1:cmax1]
        g_mask_crop = g_mask_crop.astype("int")
        p_mask_crop = inst_pred[rmin1:rmax1, cmin1:cmax1]
        pred_target_overlap = p_mask_crop[g_mask_crop > 0]
        pred_target_overlap_id = list(np.unique(pred_target_overlap))
        for pred_id in pred_target_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            g_mask_union_crop = (g_mask[rmin:rmax, cmin:cmax] > 0).astype(np.uint8)
            p_mask_union_crop = (p_mask[rmin:rmax, cmin:cmax] > 0).astype(np.uint8)
            total = (g_mask_union_crop + p_mask_union_crop).sum()
            intersect = (g_mask_union_crop * p_mask_union_crop).sum()
            # print(total, intersect)
            pairwise_intersection[gt_id - 1, pred_id - 1] = intersect
            pairwise_union[gt_id - 1, pred_id - 1] = total - intersect

    pairwise_iou = pairwise_intersection / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each target, dont care
    # about reusing pred instance multiple times
    if pairwise_iou.shape[0] * pairwise_iou.shape[1] == 0:
        return 0., 0.
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_gt = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_gt]
    overall_inter = (pairwise_intersection[paired_gt, paired_pred]).sum()
    overall_union = (pairwise_union[paired_gt, paired_pred]).sum()

    paired_gt = list(paired_gt + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # It seems that only unpaired Predictions need to be added into union.
    unpaired_gt = np.array([idx for idx in gt_id_list if (idx not in paired_gt) and (idx != 0)])
    for gt_id in unpaired_gt:
        overall_union += gt_masks[gt_id].sum()
    unpaired_pred = np.array([idx for idx in pred_id_list if (idx not in paired_pred) and (idx != 0)])
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    return overall_inter, overall_union


def pre_eval_aji(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class, num_classes, reduce_zero_label=True):
    pred_sem_ids = list(pred_id_list_per_class.keys())
    gt_sem_ids = list(gt_id_list_per_class.keys())

    union_sem_ids = list(set(pred_sem_ids + gt_sem_ids))

    overall_inter = np.zeros((num_classes), dtype=np.float32)
    overall_union = np.zeros((num_classes), dtype=np.float32)
    for sem_id in union_sem_ids:
        # NOTE: this part overall union is about mismatching between semantic map & instance map.
        if sem_id == 0:
            pred_id_list = pred_id_list_per_class[sem_id]
            gt_id_list = gt_id_list_per_class[sem_id]
            overall_union[sem_id] += sum([np.sum(inst_pred == pred_id) for pred_id in pred_id_list if pred_id != 0])
            overall_union[sem_id] += sum([np.sum(inst_gt == gt_id) for gt_id in gt_id_list if gt_id != 0])
            continue

        if sem_id in pred_id_list_per_class and sem_id in gt_id_list_per_class:
            pred_id_list = pred_id_list_per_class[sem_id]
            gt_id_list = gt_id_list_per_class[sem_id]
            pred_inst_map = sum([((inst_pred == pred_id).astype(np.int32) * (idx + 1))
                                 for idx, pred_id in enumerate(pred_id_list)])
            gt_inst_map = sum([((inst_gt == gt_id).astype(np.int32) * (idx + 1))
                               for idx, gt_id in enumerate(gt_id_list)])

            res = pre_eval_bin_aji(pred_inst_map, gt_inst_map)
            overall_inter[sem_id] += res[0]
            overall_union[sem_id] += res[1]
        # NOTE: this part overall union is about semantic results mismatching between prediction & ground truth.
        elif sem_id in pred_id_list_per_class:
            pred_id_list = pred_id_list_per_class[sem_id]
            overall_union[sem_id] += sum([np.sum(inst_pred == pred_id) for pred_id in pred_id_list if pred_id != 0])
        elif sem_id in gt_id_list_per_class:
            gt_id_list = gt_id_list_per_class[sem_id]
            overall_union[sem_id] += sum([np.sum(inst_gt == gt_id) for gt_id in gt_id_list if gt_id != 0])

    if reduce_zero_label:
        overall_inter = overall_inter[1:]
        overall_union = overall_union[1:]

    return overall_inter, overall_union


def pre_eval_bin_pq(inst_pred, inst_gt, match_iou=0.5):
    assert match_iou >= 0.0, "Cant' be negative"

    # make instance id contiguous
    inst_pred = measure.label(inst_pred.copy())
    inst_gt = measure.label(inst_gt.copy())

    pred_id_list = list(np.unique(inst_pred))
    gt_id_list = list(np.unique(inst_gt))

    if 0 not in pred_id_list:
        pred_id_list.insert(0, 0)

    if 0 not in gt_id_list:
        gt_id_list.insert(0, 0)

    # Remove background class
    pred_masks = []
    for p in pred_id_list:
        p_mask = (inst_pred == p).astype(np.uint8)
        pred_masks.append(p_mask)

    gt_masks = []
    for g in gt_id_list:
        g_mask = (inst_gt == g).astype(np.uint8)
        gt_masks.append(g_mask)

    # prefill with value
    pairwise_iou = np.zeros([len(gt_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64)

    # caching pairwise iou
    for gt_id in gt_id_list[1:]:  # 0-th is background
        g_mask = gt_masks[gt_id]

        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(g_mask)
        g_mask_crop = g_mask[rmin1:rmax1, cmin1:cmax1]
        g_mask_crop = g_mask_crop.astype("int")
        p_mask_crop = inst_pred[rmin1:rmax1, cmin1:cmax1]
        pred_gt_overlap = p_mask_crop[g_mask_crop > 0]
        pred_gt_overlap_id = np.unique(pred_gt_overlap)
        pred_gt_overlap_id = list(pred_gt_overlap_id)
        for pred_id in pred_gt_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            g_mask_union_crop = g_mask[rmin:rmax, cmin:cmax]
            p_mask_union_crop = p_mask[rmin:rmax, cmin:cmax]
            total = (g_mask_union_crop + p_mask_union_crop).sum()
            inter = (g_mask_union_crop * p_mask_union_crop).sum()
            iou = inter / (total - inter)
            pairwise_iou[gt_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_gt, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_gt, paired_pred]
        paired_gt += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        # Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_gt, paired_pred = linear_sum_assignment(-pairwise_iou)
        # extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_gt, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_gt = list(paired_gt[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_gt = [idx for idx in gt_id_list[1:] if idx not in paired_gt]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_gt)
    fp = len(unpaired_pred)
    fn = len(unpaired_gt)
    iou = paired_iou.sum()

    return tp, fp, fn, iou


def pre_eval_pq(inst_pred, inst_gt, pred_id_list_per_class, gt_id_list_per_class, num_classes, reduce_zero_label=True):
    pred_sem_ids = list(pred_id_list_per_class.keys())
    gt_sem_ids = list(gt_id_list_per_class.keys())

    union_sem_ids = list(set(pred_sem_ids + gt_sem_ids))

    tp = np.zeros((num_classes), dtype=np.float32)
    fp = np.zeros((num_classes), dtype=np.float32)
    fn = np.zeros((num_classes), dtype=np.float32)
    iou = np.zeros((num_classes), dtype=np.float32)
    for sem_id in union_sem_ids:
        # NOTE: this part overall union is about mismatching between semantic map & instance map.
        if sem_id == 0:
            pred_id_list = pred_id_list_per_class[sem_id]
            gt_id_list = gt_id_list_per_class[sem_id]
            fp[sem_id] += len(pred_id_list)
            fn[sem_id] += len(gt_id_list)
            # assert(len(gt_id_list) == 0)
            # print('pred', len(pred_id_list), 'gt', len(gt_id_list))
            continue

        if sem_id in pred_id_list_per_class and sem_id in gt_id_list_per_class:
            pred_id_list = pred_id_list_per_class[sem_id]
            gt_id_list = gt_id_list_per_class[sem_id]
            pred_inst_map = sum([((inst_pred == pred_id).astype(np.int32) * (idx + 1))
                                 for idx, pred_id in enumerate(pred_id_list)])
            gt_inst_map = sum([((inst_gt == gt_id).astype(np.int32) * (idx + 1))
                               for idx, gt_id in enumerate(gt_id_list)])

            res = pre_eval_bin_pq(pred_inst_map, gt_inst_map)
            tp[sem_id] += res[0]
            fp[sem_id] += res[1]
            fn[sem_id] += res[2]
            iou[sem_id] += res[3]
        # NOTE: this part overall union is about semantic results mismatching between prediction & ground truth.
        elif sem_id in pred_id_list_per_class:
            pred_id_list = pred_id_list_per_class[sem_id]
            fp[sem_id] += len(pred_id_list)
        elif sem_id in gt_id_list_per_class:
            gt_id_list = gt_id_list_per_class[sem_id]
            fn[sem_id] += len(gt_id_list)

    if reduce_zero_label:
        tp = tp[1:]
        fp = fp[1:]
        fn = fn[1:]
        iou = iou[1:]

    return tp, fp, fn, iou


def binary_aggregated_jaccard_index(inst_pred, inst_gt):
    """Two-class Aggregated Jaccard Index Calculation.

    0 is set as background pixels and we will ignored.

    Args:
        inst_pred (numpy.ndarray): Prediction instance map.
        inst_gt (numpy.ndarray): Ground truth instance map.
    """
    return aggregated_jaccard_index(inst_pred, inst_gt, inst_pred > 0, inst_gt > 0, 2)


def aggregated_jaccard_index(inst_pred, inst_gt, sem_pred, sem_gt, num_classes):
    """Class Wise Aggregated Jaccard Index Calculation.

    0 is set as background pixels and we will ignored.

    Args:
        inst_pred (numpy.ndarray): Prediction instance map.
        inst_gt (numpy.ndarray): Ground truth instance map.
        sem_pred (numpy.ndarry): Related prediction semantic map which provides instances classification infomation.
        sem_gt (numpy.ndarry): Related ground truth semantic map.
        num_classes (int): The number of classification.
    """
    i, u = pre_eval_aji(inst_pred, inst_gt, sem_pred, sem_gt, num_classes)
    # remove background class (index 0)
    i = i[1:]
    u = u[1:]
    if np.sum(i) == 0. or np.sum(u) == 0.:
        return 0.
    aji_score = np.sum(i) / np.sum(u)
    return aji_score


def binary_panoptic_quality(inst_pred, inst_gt, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    res = pre_eval_bin_pq(inst_pred, inst_gt, match_iou)

    # remove background class (index 0)
    tp = res[0]
    fp = res[1]
    fn = res[2]
    iou = res[3]

    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = iou / (tp + 1.0e-6)

    return dq, sq, dq * sq


def panoptic_quality(inst_pred, inst_gt, sem_pred, sem_gt, num_classes, match_iou=0.5):
    tp, fp, fn, iou = pre_eval_pq(inst_pred, inst_gt, sem_pred, sem_gt, num_classes, match_iou)

    tp = tp[1:]
    fp = fp[1:]
    fn = fn[1:]
    iou = iou[1:]

    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)
    iou = np.sum(iou)

    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = iou / (tp + 1.0e-6)

    return dq, sq, dq * sq


def binary_inst_dice(inst_pred, inst_gt, match_iou=0.5):
    """Two-class instance level dice score."""
    res = pre_eval_bin_pq(inst_pred, inst_gt, match_iou)
    tp = res[0]
    fp = res[1]
    fn = res[2]

    return 2 * tp / (2 * tp + fp + fn)


def pre_eval_to_bin_aji(pre_eval_results, nan_to_num=None):
    """Convert aji pre-eval overall intersection & pre-eval overall union to aji score."""
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 2

    # [0]: overall intersection
    # [1]: overall union
    inst_inter = [np.sum(x) for x in pre_eval_results[0]]
    inst_union = [np.sum(x) for x in pre_eval_results[1]]

    ret_metrics = {'Aji': sum(inst_inter) / sum(inst_union)}

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})

    return ret_metrics


def pre_eval_to_imw_aji(pre_eval_results, nan_to_num=None):
    """Convert aji pre-eval overall intersection & pre-eval overall union to aji score."""
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 2

    inst_inter = [np.sum(x) for x in pre_eval_results[0]]
    inst_union = [np.sum(x) for x in pre_eval_results[1]]

    ret_metrics = {}
    ret_metrics['Aji'] = []
    for i, u in zip(inst_inter, inst_union):
        ret_metrics['Aji'].append(i / u)

    ret_metrics['Aji'] = np.array(ret_metrics['Aji'])

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})

    return ret_metrics


def pre_eval_to_aji(pre_eval_results, nan_to_num=None):
    """Convert aji pre-eval overall intersection & pre-eval overall union to aji score."""
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 2

    # [0]: overall intersection
    # [1]: overall union
    overall_inter = sum(pre_eval_results[0])
    overall_union = sum(pre_eval_results[1])

    aji = overall_inter / overall_union

    ret_metrics = {'Aji': aji}

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})

    return ret_metrics


def pre_eval_to_bin_pq(pre_eval_results, nan_to_num=None, analysis_mode=False):
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    # [0]: tp (The number of inst with iou > 0.5)
    # [1]: fp (The number of inst in prediction with iou < 0.5)
    # [2]: fn (The numebr of inst in ground truth with iou < 0.5)
    # [3]: iou (The sum value of paired inst iou)
    tp = sum([np.sum(x) for x in pre_eval_results[0]])
    fp = sum([np.sum(x) for x in pre_eval_results[1]])
    fn = sum([np.sum(x) for x in pre_eval_results[2]])
    iou = sum([np.sum(x) for x in pre_eval_results[3]])

    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = iou / (tp + 1.0e-6)

    pq = dq * sq

    ret_metrics = {'DQ': dq, 'SQ': sq, 'PQ': pq}

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})

    if analysis_mode:
        analysis = {'pq_TP': tp, 'pq_FP': fp, 'pq_FN': fn, 'pq_IoU': np.round(iou, 2)}
        ret_metrics.update(analysis)

    return ret_metrics


def pre_eval_to_imw_pq(pre_eval_results, nan_to_num=None):
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    # [0]: tp (The number of inst with iou > 0.5)
    # [1]: fp (The number of inst in prediction with iou < 0.5)
    # [2]: fn (The numebr of inst in ground truth with iou < 0.5)
    # [3]: iou (The sum value of paired inst iou)
    tp_list = [np.sum(x) for x in pre_eval_results[0]]
    fp_list = [np.sum(x) for x in pre_eval_results[1]]
    fn_list = [np.sum(x) for x in pre_eval_results[2]]
    iou_list = [np.sum(x) for x in pre_eval_results[3]]

    ret_metrics = {}
    ret_metrics['DQ'] = []
    ret_metrics['SQ'] = []
    ret_metrics['PQ'] = []
    for tp, fp, fn, iou in zip(tp_list, fp_list, fn_list, iou_list):
        # get the F1-score i.e DQ
        dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
        # get the SQ, no paired has 0 iou so not impact
        sq = iou / (tp + 1.0e-6)
        pq = dq * sq
        ret_metrics['DQ'].append(dq)
        ret_metrics['SQ'].append(sq)
        ret_metrics['PQ'].append(pq)

    ret_metrics['DQ'] = np.array(ret_metrics['DQ'])
    ret_metrics['SQ'] = np.array(ret_metrics['SQ'])
    ret_metrics['PQ'] = np.array(ret_metrics['PQ'])

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})

    return ret_metrics


def pre_eval_to_pq(pre_eval_results, nan_to_num=None, analysis_mode=False):
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    # [0]: tp (The number of inst with iou > 0.5)
    # [1]: fp (The number of inst in prediction with iou < 0.5)
    # [2]: fn (The numebr of inst in ground truth with iou < 0.5)
    # [3]: iou (The sum value of paired inst iou)
    tp = sum(pre_eval_results[0])
    fp = sum(pre_eval_results[1])
    fn = sum(pre_eval_results[2])
    iou = sum(pre_eval_results[3])

    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = iou / (tp + 1.0e-6)
    pq = dq * sq

    ret_metrics = {'DQ': dq, 'SQ': sq, 'PQ': pq}

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})

    if analysis_mode:
        analysis = {'pq_TP': tp, 'pq_FP': fp, 'pq_FN': fn, 'pq_IoU': np.round(iou, 2)}
        ret_metrics.update(analysis)

    return ret_metrics


def pre_eval_to_imw_inst_dice(pre_eval_results, nan_to_num=None):
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    # [0]: tp (The number of inst with iou > 0.5)
    # [1]: fp (The number of inst in prediction with iou < 0.5)
    # [2]: fn (The numebr of inst in ground truth with iou < 0.5)
    # [3]: iou (The sum value of paired inst iou)
    tp_list = pre_eval_results[0]
    fp_list = pre_eval_results[1]
    fn_list = pre_eval_results[2]
    _ = pre_eval_results[3]

    ret_metrics = {}
    ret_metrics['InstDice'] = []
    for tp, fp, fn in zip(tp_list, fp_list, fn_list):
        if tp + fp + fn == 0:
            inst_dice = 1.0
        else:
            inst_dice = 2 * tp / (2 * tp + fp + fn)
        ret_metrics['InstDice'].append(inst_dice)

    ret_metrics['InstDice'] = np.array(ret_metrics['InstDice'])

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})
    return ret_metrics


def pre_eval_to_inst_dice(pre_eval_results, nan_to_num=None):
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4

    # [0]: tp (The number of inst with iou > 0.5)
    # [1]: fp (The number of inst in prediction with iou < 0.5)
    # [2]: fn (The numebr of inst in ground truth with iou < 0.5)
    # [3]: iou (The sum value of paired inst iou)
    tp = sum(pre_eval_results[0])
    fp = sum(pre_eval_results[1])
    fn = sum(pre_eval_results[2])

    ret_metrics = {}
    if tp + fp + fn == 0:
        ret_metrics['InstDice'] = 1.0
    else:
        ret_metrics['InstDice'] = 2 * tp / (2 * tp + fp + fn)

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})

    return ret_metrics
