###############################################################################
# Author: Wasi Ahmad
# Project: Convolutional Latent Semantic Model
# Date Created: 7/18/2017
#
# File Description: This script implements the deep semantic similarity model.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as f


class CLSM(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, args):
        """"Constructor of the class."""
        super(CLSM, self).__init__()
        self.dictionary = dictionary
        self.config = args
        self.convolution = nn.Conv1d(len(dictionary), self.config.nhid, 3)
        self.output = nn.Linear(self.config.nhid, self.config.nhid_output)
        self.tanh = nn.Tanh()

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

    @staticmethod
    def cosine_similarity(x1, x2, dim=1, eps=1e-8):
        """
        Returns cosine similarity between x1 and x2, computed along dim.
        # taken from http://pytorch.org/docs/master/_modules/torch/nn/functional.html#cosine_similarity
        :param x1: (Variable): First input.
        :param x2: (Variable): Second input (of size matching x1).
        :param dim: (int, optional): Dimension of vectors. Default: 1
        :param eps: Small value to avoid division by zero. Default: 1e-8
        :return: 
        """
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    def forward(self, batch_queries, batch_docs, click_labels):
        """
        Forward function of the dssm model. Return average loss for a batch of queries.
        :param batch_queries: 4d tensor [batch_size x num_trigrams x 3 x max_num_letter_trigrams]
        :param batch_docs: 5d tensor [batch_size x num_rel_docs_per_query x num_trigrams x 3 x max_num_letter_trigrams]
        :param click_labels: 2d tensor [batch_size x num_rel_docs_per_query]
        :return: average loss over batch [autograd Variable]
        """
        # query encoding
        query_rep = self.convolution(batch_queries.transpose(1, 2)).transpose(1, 2)
        latent_query_rep = torch.max(query_rep, 1)[0].squeeze()
        # document encoding
        doc_rep = self.convolution(batch_docs.view(-1, *batch_docs.size()[2:]).transpose(1, 2)).transpose(1, 2)
        latent_doc_rep = torch.max(doc_rep, 1)[0].squeeze().view(*batch_docs.size()[:2], -1)
        # compute loss
        latent_query_rep = latent_query_rep.unsqueeze(1).expand(*latent_doc_rep.size())
        softmax_prob = f.log_softmax(self.cosine_similarity(latent_query_rep, latent_doc_rep, 2))
        return self.compute_loss(softmax_prob, click_labels)
