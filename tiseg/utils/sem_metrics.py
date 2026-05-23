from collections import OrderedDict

import mmcv
import numpy as np
import torch


def to_ndarray(val):
    if isinstance(val, torch.Tensor):
        return val.numpy()
    else:
        return np.array(val)


# TODO: Add doc string & comments
def pre_eval_all_semantic_metric(pred_label, target_label, num_classes, ignore_index=255, reduce_zero_label=True):
    """Generate pre eval results for all semantic metrics."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    mask = target_label != ignore_index
    pred_label = pred_label[mask]
    target_label = target_label[mask]

    TP = target_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(FN.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    TN_per_class = Pred_per_class.sum() - (TP_per_class + FP_per_class + FN_per_class)

    if reduce_zero_label:
        TP_per_class = TP_per_class[1:]
        FP_per_class = FP_per_class[1:]
        FN_per_class = FN_per_class[1:]
        TN_per_class = TN_per_class[1:]
        Pred_per_class = Pred_per_class[1:]
        GT_per_class = GT_per_class[1:]

    ret_package = (TP_per_class, TN_per_class, FP_per_class, FN_per_class, Pred_per_class, GT_per_class)

    return ret_package


def accuracy(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class accuracy calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = target_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(FN.float(), bins=(num_classes), min=0, max=num_classes - 1)

    TN_per_class = pred_label.numel() - (TP_per_class + FP_per_class + FN_per_class)

    accuracy = (TP_per_class + TN_per_class) / pred_label.numel()

    accuracy = np.nan_to_num(accuracy.numpy(), nan_to_num)

    return accuracy


def precision_recall(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class precision-recall calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]
    FP = pred_label[pred_label != target_label]
    FN = target_label[pred_label != target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FP_per_class = torch.histc(FP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    FN_per_class = torch.histc(FN.float(), bins=(num_classes), min=0, max=num_classes - 1)

    precision = TP_per_class / (TP_per_class + FP_per_class)
    recall = TP_per_class / (TP_per_class + FN_per_class)

    precision = np.nan_to_num(precision.numpy(), nan_to_num)
    recall = np.nan_to_num(recall.numpy(), nan_to_num)

    return precision, recall


def dice_similarity_coefficient(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class dice calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    dice = 2 * TP_per_class / (Pred_per_class + GT_per_class)

    dice = np.nan_to_num(dice.numpy(), nan_to_num)

    return dice


def intersect_and_union(pred_label, target_label, num_classes, nan_to_num=None):
    """multi-class iou calculation."""
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(target_label, str):
        target_label = torch.from_numpy(mmcv.imread(target_label, flag='unchanged', backend='pillow'))
    else:
        target_label = torch.from_numpy(target_label)

    TP = pred_label[pred_label == target_label]

    TP_per_class = torch.histc(TP.float(), bins=(num_classes), min=0, max=num_classes - 1)
    Pred_per_class = torch.histc(pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    GT_per_class = torch.histc(target_label.float(), bins=(num_classes), min=0, max=num_classes - 1)

    iou = TP_per_class / (Pred_per_class + GT_per_class - TP_per_class)

    iou = np.nan_to_num(iou.numpy(), nan_to_num)

    return iou


def pre_eval_to_imw_sem_metrics(pre_eval_results, metrics=['IoU'], nan_to_num=None):
    """
    """
    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 6

    TP_list = [torch.sum(x) for x in pre_eval_results[0]]
    TN_list = [torch.sum(x) for x in pre_eval_results[1]]
    FP_list = [torch.sum(x) for x in pre_eval_results[2]]
    FN_list = [torch.sum(x) for x in pre_eval_results[3]]
    # prediction area
    P_list = [torch.sum(x) for x in pre_eval_results[4]]
    # Ground Truth area
    G_list = [torch.sum(x) for x in pre_eval_results[5]]

    ret_metrics = {}
    if 'Accuracy' in metrics:
        ret_metrics['Accuracy'] = []
        for TP, TN, G in zip(TP_list, TN_list, G_list):
            ret_metrics['Accuracy'].append(to_ndarray((TP + TN) / (G.sum())))
    if 'IoU' in metrics:
        ret_metrics['IoU'] = []
        for TP, P, G in zip(TP_list, P_list, G_list):
            ret_metrics['IoU'].append(to_ndarray(TP / (G + P - TP)))
    if 'Dice' in metrics:
        ret_metrics['Dice'] = []
        for TP, P, G in zip(TP_list, P_list, G_list):
            denom = G + P
            # Binary foreground Dice: if both GT and prediction are empty,
            # the foreground segmentation is exactly correct.
            if torch.sum(denom) == 0:
                ret_metrics['Dice'].append(np.array(1.0))
            else:
                ret_metrics['Dice'].append(to_ndarray(2 * TP / denom))
    if 'Recall' in metrics:
        ret_metrics['Recall'] = []
        for TP, FN in zip(TP_list, FN_list):
            ret_metrics['Recall'].append(to_ndarray(TP / (TP + FN)))
    if 'Precision' in metrics:
        ret_metrics['Precision'] = []
        for TP, FP in zip(TP_list, FP_list):
            ret_metrics['Precision'].append(to_ndarray(TP / (TP + FP)))

    for key in ret_metrics.keys():
        ret_metrics[key] = to_ndarray(ret_metrics[key])

    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})
    return ret_metrics


def pre_eval_to_sem_metrics(pre_eval_results, metrics=['IoU'], nan_to_num=None, beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 6

    total_area_TP = sum(pre_eval_results[0])
    total_area_TN = sum(pre_eval_results[1])
    total_area_FP = sum(pre_eval_results[2])
    total_area_FN = sum(pre_eval_results[3])
    total_area_pred_label = sum(pre_eval_results[4])
    total_area_label = sum(pre_eval_results[5])

    ret_metrics = total_area_to_sem_metrics(total_area_TP, total_area_TN, total_area_FP, total_area_FN,
                                            total_area_pred_label, total_area_label, metrics, nan_to_num)

    return ret_metrics


def total_area_to_sem_metrics(total_area_TP,
                              total_area_TN,
                              total_area_FP,
                              total_area_FN,
                              total_area_pred_label,
                              total_area_label,
                              metrics=['IoU'],
                              nan_to_num=None):
    """Calculate evaluation metrics
    Args:
        total_area_TP (torch.Tensor): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_FP (torch.Tensor): The false positive pixels histogram on
            all classes.
        total_area_FN (torch.Tensor): The prediction histogram on all
            classes.
        total_area_pred_label (torch.Tensor): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'IoU' and 'Dice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        dict: Contains selected metric value.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['Accuracy', 'IoU', 'Dice', 'Recall', 'Precision']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    ret_metrics = {}
    for metric in metrics:
        if metric == 'Accuracy':
            acc = (total_area_TP + total_area_TN) / total_area_label.sum()
            ret_metrics['Accuracy'] = acc
        elif metric == 'IoU':
            iou = total_area_TP / (total_area_pred_label + total_area_label - total_area_TP)
            ret_metrics['IoU'] = iou
        elif metric == 'Dice':
            dice = 2 * total_area_TP / (total_area_pred_label + total_area_label)
            ret_metrics['Dice'] = dice
        elif metric == 'Recall':
            recall = total_area_TP / (total_area_TP + total_area_FN)
            ret_metrics['Recall'] = recall
        elif metric == 'Precision':
            precision = total_area_TP / (total_area_TP + total_area_FP)
            ret_metrics['Precision'] = precision

    # convert torch to numpy
    ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num)
             for metric, metric_value in ret_metrics.items()})
    return ret_metrics
