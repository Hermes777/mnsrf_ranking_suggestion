###############################################################################
# Author: Wasi Ahmad
# Project: ARC-I: Convolutional Matching Model
# Date Created: 7/28/2017
#
# File Description: This script contains ranking evaluation functions.
###############################################################################

import torch, numpy


def mean_average_precision(logits, target):
    """
    Compute mean average precision.
    :param logits: 2d tensor [batch_size x num_clicks_per_query]
    :param target: 2d tensor [batch_size x num_clicks_per_query]
    :return: mean average precision [a float value]
    """
    assert logits.size() == target.size()
    sorted, indices = torch.sort(logits, 1, descending=True)
    map = 0
    for i in range(indices.size(0)):
        average_precision = 0
        num_rel = 0
        for j in range(indices.size(1)):
            if target[i, indices[i, j].data[0]].data[0] == 1:
                num_rel += 1
                average_precision += num_rel / (j + 1)
        average_precision = average_precision / num_rel
        map += average_precision

    return map / indices.size(0)


def NDCG(logits, target, k):
    """
    Compute normalized discounted cumulative gain.
    :param logits: 2d tensor [batch_size x rel_docs_per_query]
    :param target: 2d tensor [batch_size x rel_docs_per_query]
    :return: mean average precision [a float value]
    """
    assert logits.size() == target.size()
    assert logits.size(1) >= k, 'NDCG@K cannot be computed, invalid value of K.'

    sorted, indices = torch.sort(logits, 1, descending=True)
    NDCG = 0
    for i in range(indices.size(0)):
        DCG_ref = 0
        num_rel_docs = torch.nonzero(target[i].data).size(0)
        for j in range(indices.size(1)):
            if j == k:
                break
            if target[i, indices[i, j].data[0]].data[0] == 1:
                DCG_ref += 1 / numpy.log2(j + 2)
        DCG_gt = 0
        for j in range(num_rel_docs):
            if j == k:
                break
            DCG_gt += 1 / numpy.log2(j + 2)
        NDCG += DCG_ref / DCG_gt

    return NDCG / indices.size(0)
