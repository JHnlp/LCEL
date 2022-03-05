"""
=======================================================================
 Copyright (c) 2019 PolyU CBS LLT Group. All Rights Reserved

@Author      :  Jinghang GU
@Contect     :  gujinghangnlp@gmail.com
@Time        :  2020/5/19
@Description :
=======================================================================
"""
import logging
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, issparse

logger = logging.getLogger(__name__)

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, classification_report

    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False


def simple_accuracy(pred, target):
    acc = pred == target
    acc = acc.astype('float').mean()
    return round(acc, 6)


def example_based_precision(pred, target, epsilon=1e-9):
    if issparse(pred) or issparse(target):
        intersection = pred.multiply(target).sum(axis=1)
        p = intersection / (pred.sum(axis=1) + epsilon)
        p = p.mean()
    else:
        intersection = (pred * target).sum(axis=1, keepdims=True)
        p = intersection / (pred.sum(axis=1, keepdims=True) + epsilon)
        p = p.mean()
    return round(p, 6)


def example_based_recall(pred, target, epsilon=1e-9):
    if issparse(pred) or issparse(target):
        intersection = pred.multiply(target).sum(axis=1)
        r = intersection / (target.sum(axis=1) + epsilon)
        r = r.mean()
    else:
        intersection = (pred * target).sum(axis=1, keepdims=True)
        r = intersection / (target.sum(axis=1, keepdims=True) + epsilon)
        r = r.mean()
    return round(r, 6)


def example_based_f1(pred, target, epsilon=1e-9):
    p = example_based_precision(pred, target, epsilon=epsilon)
    r = example_based_recall(pred, target, epsilon=epsilon)
    f = 2 * p * r / (p + r + epsilon)
    return round(f, 6)


def label_level_micro_precision(pred, target, epsilon=1e-9):
    """
        MiP: only calculate the Precision of true positive labels.
    """
    intersections = pred[target == 1].sum()
    ebp = intersections / (pred.sum() + epsilon)
    return round(ebp, 6)


def label_level_micro_recall(pred, target, epsilon=1e-9):
    """
        MiR: only calculate the Recall of true positive labels.
    """
    intersections = pred[target == 1].sum()
    ebr = intersections / (target.sum() + epsilon)
    return round(ebr, 6)


def label_level_micro_f1(pred, target, epsilon=1e-9):
    """
        MiF: only calculate the F1 score of true positive labels.
    """
    p = label_level_micro_precision(pred, target, epsilon=epsilon)
    r = label_level_micro_recall(pred, target, epsilon=epsilon)
    f = 2 * p * r / (p + r + epsilon)
    return round(f, 6)


def label_level_macro_precision(pred, target, epsilon=1e-9):
    if issparse(pred) or issparse(target):
        intersection = pred.multiply(target).sum(axis=0)
        p = intersection / (pred.sum(axis=0) + epsilon)
        valid_labels = (pred + target).sum(axis=0) > 0
        p = p.sum() / (valid_labels.sum() + epsilon)
    else:
        intersection = (pred * target).sum(axis=0)
        p = intersection / (pred.sum(axis=0) + epsilon)
        valid_labels = (pred + target).sum(axis=0) > 0
        p = p.sum() / (valid_labels.sum() + epsilon)
    return round(p, 6)


def label_level_macro_recall(pred, target, epsilon=1e-9):
    if issparse(pred) or issparse(target):
        intersection = pred.multiply(target).sum(axis=0)
        r = intersection / (target.sum(axis=0) + epsilon)
        valid_labels = (pred + target).sum(axis=0) > 0
        r = r.sum() / (valid_labels.sum() + epsilon)
    else:
        intersection = (pred * target).sum(axis=0)
        r = intersection / (target.sum(axis=0) + epsilon)
        valid_labels = (pred + target).sum(axis=0) > 0
        r = r.sum() / (valid_labels.sum() + epsilon)
    return round(r, 6)


def label_level_macro_f1(pred, target, epsilon=1e-9):
    if issparse(pred) or issparse(target):
        intersection = pred.multiply(target).sum(axis=0)
        p = intersection / (pred.sum(axis=0) + epsilon)
        r = intersection / (target.sum(axis=0) + epsilon)
        f = 2 * np.multiply(p, r) / (p + r + epsilon)
    else:
        intersection = (pred * target).sum(axis=0)
        p = intersection / (pred.sum(axis=0) + epsilon)
        r = intersection / (target.sum(axis=0) + epsilon)
        f = 2 * p * r / (p + r + epsilon)

    valid_labels = (pred + target).sum(axis=0) > 0
    f = f.sum() / (valid_labels.sum() + epsilon)
    return round(f, 6)


def acc_and_f1(pred, target):
    acc = simple_accuracy(pred, target)
    f1 = f1_score(y_true=target, y_pred=pred, average='macro')
    print(acc)
    print('\n')
    print(f1)
    print('\n')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(pred, target):
    pearson_corr = pearsonr(pred, target)[0]
    spearman_corr = spearmanr(pred, target)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, pred, target):
    # assert len(pred) == len(target)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(target, pred)}
    elif task_name == "mrpc":
        return acc_and_f1(pred, target)
    elif task_name == "sts-b":
        return pearson_and_spearman(pred, target)
    elif task_name == "qqp":
        return acc_and_f1(pred, target)
    elif task_name == "tnews":
        return {"acc": simple_accuracy(pred, target)}
    elif task_name in ["covid19", "bioasq", "litcovid"]:
        return {"Acc": simple_accuracy(pred, target),
                "EBP": example_based_precision(pred, target),
                "EBR": example_based_recall(pred, target),
                "EBF": example_based_f1(pred, target),
                "MaP": label_level_macro_precision(pred, target),
                "MaR": label_level_macro_recall(pred, target),
                "MaF": label_level_macro_f1(pred, target),
                "MiP": label_level_micro_precision(pred, target),
                "MiR": label_level_micro_recall(pred, target),
                "MiF": label_level_micro_f1(pred, target)}
    else:
        raise KeyError(task_name)


def classifiction_metric(preds, labels, label_list):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

    acc = accuracy_score(labels, preds)
    labels_list = [i for i in range(len(label_list))]
    report = classification_report(
        labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)

    return acc, report


if __name__ == "__main__":
    import torch, pprint, numpy as np

    pred = [[1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 1]]
    gold = [[1, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 1]]

    pred = np.array(pred)
    gold = np.array(gold)

    pred_sparse = csr_matrix(pred)
    gold_sparse = csr_matrix(gold)

    ebp = example_based_precision(pred_sparse, gold_sparse)
    ebr = example_based_recall(pred_sparse, gold_sparse)
    ebf = example_based_f1(pred_sparse, gold_sparse)
    mip = label_level_micro_precision(pred_sparse, gold_sparse)
    mir = label_level_micro_recall(pred_sparse, gold_sparse)
    mif = label_level_micro_f1(pred_sparse, gold_sparse)
    map = label_level_macro_precision(pred_sparse, gold_sparse)
    mar = label_level_macro_recall(pred_sparse, gold_sparse)
    maf = label_level_macro_f1(pred_sparse, gold_sparse)

    pprint.pprint([ebp, ebr, ebf, map, mar, maf, mip, mir, mif])

    pred = np.array(pred)
    gold = np.array(gold)

    ebp = example_based_precision(pred, gold)
    ebr = example_based_recall(pred, gold)
    ebf = example_based_f1(pred, gold)
    mip = label_level_micro_precision(pred, gold)
    mir = label_level_micro_recall(pred, gold)
    mif = label_level_micro_f1(pred, gold)
    map = label_level_macro_precision(pred, gold)
    mar = label_level_macro_recall(pred, gold)
    maf = label_level_macro_f1(pred, gold)
    pprint.pprint([ebp, ebr, ebf, map, mar, maf, mip, mir, mif])

    '''
    [0.4000000000000001, 0.5555555539814815, 0.43333333333333335, 0.4867724861711598, 
    0.6666666666666666, 0.47222222222222215, 0.5277777747453704, 0.75, 0.5454545454545454, 0.6315789424930748]
    Accuracy: 0.4000000000000001
    EBP : 0.5555555539814815
    EBR : 0.43333333333333335
    EBF : 0.4867724861711598
    MaP : 0.6666666666666666
    MaR : 0.47222222222222215
    MaF : 0.5277777747453704
    MiP : 0.75
    MiR : 0.5454545454545454
    MiF : 0.6315789424930748
    '''
pass
