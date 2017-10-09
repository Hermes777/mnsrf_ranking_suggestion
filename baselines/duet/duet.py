###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script implements the deep semantic similarity model.
###############################################################################

import torch.nn as nn
import torch.nn.functional as f
from nn_model import LocalModel, DistributedModel


class DUET(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, args):
        """"Constructor of the class."""
        super(DUET, self).__init__()
        self.dictionary = dictionary
        self.config = args
        self.local_model = LocalModel(self.config)
        self.distributed_model = DistributedModel(self.config, len(self.dictionary))

    @staticmethod
    def compute_loss(logits, target):
        """
        Compute negative log-likelihood loss for a batch of predictions.
        :param logits: 2d tensor [batch_size x num_rel_docs_per_query]
        :param target: 2d tensor [batch_size x num_rel_docs_per_query]
        :return: average negative log-likelihood loss over the input mini-batch [autograd Variable]
        """
        loss = -(logits * target).sum(1).squeeze(1)
        return loss.mean()

    def forward(self, batch_queries, batch_docs, click_labels):
        """
        Forward function of the dssm model. Return average loss for a batch of queries.
        :param batch_queries: 2d tensor [batch_size x vocab_size]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x vocab_size]
        :param click_labels: 2d tensor [batch_size x num_rel_docs_per_query]
        :return: average loss over batch [autograd Variable]
        """
        local_score = self.local_model(batch_queries, batch_docs)
        distributed_score = self.distributed_model(batch_queries, batch_docs)
        total_score = local_score + distributed_score
        softmax_prob = f.log_softmax(total_score)
        return self.compute_loss(softmax_prob, click_labels)
