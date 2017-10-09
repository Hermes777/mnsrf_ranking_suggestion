###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script contains the implementation of local and distributed model.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as f


class LocalModel(nn.Module):
    """Implementation of the local model."""

    def __init__(self, args):
        """"Constructor of the class."""
        super(LocalModel, self).__init__()
        self.config = args
        self.conv1d = nn.Conv1d(self.config.max_doc_length, self.config.nfilters, self.config.local_filter_size)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(self.config.dropout)
        self.fc1 = nn.Linear(self.config.max_query_length, 1)
        self.fc2 = nn.Linear(self.config.nfilters, self.config.nfilters)
        self.fc3 = nn.Linear(self.config.nfilters, 1)

    def forward(self, batch_queries, batch_clicks):
        output_size = batch_clicks.size()[:2]
        batch_queries = batch_queries.unsqueeze(1).expand(batch_queries.size(0), batch_clicks.size(1),
                                                          *batch_queries.size()[1:])
        batch_queries = batch_queries.contiguous().view(-1, *batch_queries.size()[2:]).float()
        batch_clicks = batch_clicks.view(-1, *batch_clicks.size()[2:]).transpose(1, 2).float()
        interaction_feature = self.tanh(torch.bmm(batch_queries, batch_clicks).transpose(1, 2))
        convolved_feature = self.tanh(self.conv1d(interaction_feature))
        mapped_feature1 = self.tanh(self.fc1(convolved_feature.view(-1, convolved_feature.size(2)))).squeeze(1)
        mapped_feature1 = mapped_feature1.view(*convolved_feature.size()[:-1])
        mapped_feature2 = self.drop(self.tanh(self.fc2(mapped_feature1)))
        score = self.tanh(self.fc3(mapped_feature2)).view(*output_size)
        return score


class DistributedModel(nn.Module):
    """Implementation of the distributed model."""

    def __init__(self, args, vocab_size):
        """"Constructor of the class."""
        super(DistributedModel, self).__init__()
        self.config = args
        self.conv1d = nn.Conv1d(vocab_size, self.config.nfilters, self.config.dist_filter_size)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(self.config.dropout)
        self.fc1_query = nn.Linear(self.config.nfilters, self.config.nfilters)
        self.conv1d_doc = nn.Conv1d(self.config.nfilters, self.config.nfilters, 1)
        self.fc2 = nn.Linear(self.config.max_doc_length - self.config.pool_size - 1, 1)
        self.fc3 = nn.Linear(self.config.nfilters, self.config.nfilters)
        self.fc4 = nn.Linear(self.config.nfilters, 1)

    def forward(self, batch_queries, batch_clicks):
        output_size = batch_clicks.size()[:2]
        batch_queries = batch_queries.transpose(1, 2).float()
        batch_clicks = batch_clicks.view(-1, *batch_clicks.size()[2:]).transpose(1, 2).float()
        # apply convolution 1d
        convolved_query_features = self.tanh(self.conv1d(batch_queries))
        convolved_doc_features = self.tanh(self.conv1d(batch_clicks))
        # apply max-pooling 1d
        maxpooled_query_features = f.max_pool1d(convolved_query_features, convolved_query_features.size(2)).squeeze(2)
        maxpooled_doc_features = f.max_pool1d(convolved_doc_features, self.config.pool_size, 1)
        # apply fc to query and convolution 1d to document representation
        query_rep = self.tanh(self.fc1_query(maxpooled_query_features))
        doc_rep = self.tanh(self.conv1d_doc(maxpooled_doc_features))
        # do hadamard (element-wise) product
        query_rep = query_rep.unsqueeze(2).expand(*query_rep.size(), doc_rep.size(2)).unsqueeze(1)
        query_rep = query_rep.expand(query_rep.size(0), output_size[1], *query_rep.size()[2:])
        query_rep = query_rep.contiguous().view(-1, *query_rep.size()[2:])
        query_doc_sim = query_rep * doc_rep
        # apply fc2
        mapped_features = self.tanh(self.fc2(query_doc_sim.view(-1, query_doc_sim.size(2)))).squeeze(1)
        mapped_features = mapped_features.view(*query_doc_sim.size()[:-1])
        # apply fc3 and dropout
        mapped_features_2 = self.drop(self.tanh(self.fc3(mapped_features)))
        # apply fc4
        score = self.tanh(self.fc4(mapped_features_2)).view(*output_size)
        return score
