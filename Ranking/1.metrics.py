from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    # допишите ваш код здесь
    _, idxs = ys_pred.sort(descending=True)
    ys_true_new = ys_true[idxs]
    n = 0
    for i in range(len(ys_true_new) - 1):
        for j in range(i, len(ys_true_new)):
            n += int(ys_true_new[j] > ys_true_new[i])
    return n


def compute_gain(y_value: float, gain_scheme: str) -> float:
    # допишите ваш код здесь
    if gain_scheme == 'exp2':
        return 2 ** y_value - 1
    elif gain_scheme == 'const':
        return y_value
    else:
        raise KeyError('gain_scheme is not correct')


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    # допишите ваш код здесь
    _, idxs = ys_pred.sort(descending=True)
    ys_true_new = ys_true[idxs]
    s = 0
    for i in range(len(ys_true_new)):
        s += compute_gain(ys_true_new[i].item(), gain_scheme) / log2(i + 2)
    return s


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    # допишите ваш код здесь
    return dcg(ys_true, ys_pred, gain_scheme) / dcg(ys_true, ys_true, gain_scheme)


def precision_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    if ys_pred.sum().item() == 0 or ys_true.sum().item == 0:
        return -1
    _, idxs = ys_pred.sort(descending=True)
    ys_true_new = ys_true[idxs]
    k1 = min(len(ys_true_new), k)
    tp = ys_true_new[:k1].sum().item() # корректно, так как считаем релевантным все к
    tp_fp = min((ys_true.sum().item(), k)) # не совсем корректно, так как позитивный исход д.б. = к
    return tp / tp_fp


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    # допишите ваш код здесь
    _, idxs = ys_pred.sort(descending=True)
    ys_true_new = ys_true[idxs]
    idx = (ys_true_new == 1).nonzero()[0].item()
    return 1 / (idx + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15 ) -> float:
    _, idxs = ys_pred.sort(descending=True)
    ys_true_new = ys_true[idxs]

    # для 0 документа
    plook = 1
    prel = ys_true_new[0].item()
    pfound = prel * plook

    for ys in ys_true_new[1:]:
        plook = plook * (1 - prel) * (1 - p_break)
        prel = ys.item()
        pfound += plook * prel
    return pfound


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if ys_pred.sum().item() == 0 or ys_true.sum().item == 0:
        return -1
    _, idxs = ys_pred.sort(descending=True)
    ys_true_new = ys_true[idxs]
    metric = 0
    for k in range(len(ys_true_new)):
        if ys_true_new[k].item() != 0:
            tp = ys_true_new[:k+1].sum().item()
            tp_fp = k+1
            metric += tp / tp_fp
    return metric / ys_true_new.sum().item()
