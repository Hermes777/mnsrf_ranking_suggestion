###############################################################################
# Author: Wasi Ahmad
# Project: Match Tensor: a Deep Relevance Model for Search
# Date Created: 7/28/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch
import torch.nn as nn
from nn_layer import EmbeddingLayer, Encoder, ExactMatchChannel


class MatchTensor(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(MatchTensor, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args
        self.num_directions = 2 if self.config.bidirection else 1

        self.embedding = EmbeddingLayer(len(self.dictionary), self.config)
        self.linear_projection = nn.Linear(self.config.emsize, self.config.featsize)
        self.query_encoder = Encoder(self.config.featsize, self.config.nhid_query, True, self.config)
        self.document_encoder = Encoder(self.config.featsize, self.config.nhid_doc, True, self.config)
        self.query_projection = nn.Linear(self.config.nhid_query * self.num_directions, self.config.nchannels)
        self.document_projection = nn.Linear(self.config.nhid_doc * self.num_directions, self.config.nchannels)

        self.exact_match_channel = ExactMatchChannel()
        self.conv1 = nn.Conv2d(self.config.nchannels + 1, self.config.nfilters, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(self.config.nchannels + 1, self.config.nfilters, (3, 5), padding=(1, 2))
        self.conv3 = nn.Conv2d(self.config.nchannels + 1, self.config.nfilters, (3, 7), padding=(1, 3))
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(self.config.nfilters * 3, self.config.match_filter_size, (1, 1))
        self.output = nn.Linear(self.config.match_filter_size, 1)

        # Initializing the weight parameters for the embedding layer.
        self.embedding.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    def forward(self, batch_queries, batch_docs):
        """
        Forward function of the match tensor model. Return average loss for a batch of sessions.
        :param batch_queries: 2d tensor [batch_size x max_query_length]
        :param batch_docs: 3d tensor [batch_size x num_rel_docs_per_query x max_document_length]
        :return: average loss over batch [autograd Variable]
        """
        embedded_queries = self.embedding(batch_queries)
        embedded_docs = self.embedding(batch_docs.view(-1, batch_docs.size(-1)))
        embedded_queries = self.linear_projection(embedded_queries.view(-1, embedded_queries.size(-1)))
        embedded_docs = self.linear_projection(embedded_docs.view(-1, embedded_docs.size(-1)))

        embedded_queries = embedded_queries.view(*batch_queries.size(), self.config.featsize)
        embedded_docs = embedded_docs.view(-1, batch_docs.size()[-1], self.config.featsize)

        if self.config.model == 'LSTM':
            encoder_hidden, encoder_cell = self.query_encoder.init_weights(embedded_queries.size(0))
            output, hidden = self.query_encoder(embedded_queries, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = self.query_encoder.init_weights(embedded_queries.size(0))
            output, hidden = self.query_encoder(embedded_queries, encoder_hidden)

        embedded_queries = self.query_projection(output.view(-1, output.size()[-1])).view(*batch_queries.size(), -1)
        embedded_queries = embedded_queries.unsqueeze(1).expand(embedded_queries.size(0), batch_docs.size(1),
                                                                *embedded_queries.size()[1:])
        embedded_queries = embedded_queries.contiguous().view(-1, *embedded_queries.size()[2:])

        if self.config.model == 'LSTM':
            encoder_hidden, encoder_cell = self.document_encoder.init_weights(embedded_docs.size(0))
            output, hidden = self.document_encoder(embedded_docs, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = self.document_encoder.init_weights(embedded_docs.size(0))
            output, hidden = self.document_encoder(embedded_docs, encoder_hidden)

        embedded_docs = self.document_projection(output.view(-1, output.size()[-1]))
        embedded_docs = embedded_docs.view(-1, batch_docs.size(2), embedded_docs.size()[-1])

        embedded_queries = embedded_queries.unsqueeze(2).expand(*embedded_queries.size()[:2], batch_docs.size()[-1],
                                                                embedded_queries.size(2))
        embedded_docs = embedded_docs.unsqueeze(1).expand(embedded_docs.size(0), batch_queries.size()[-1],
                                                          *embedded_docs.size()[1:])

        query_document_product = embedded_queries * embedded_docs
        exact_match = self.exact_match_channel(batch_queries, batch_docs).unsqueeze(3)
        query_document_product = torch.cat((query_document_product, exact_match), 3)
        query_document_product = query_document_product.transpose(2, 3).transpose(1, 2)

        convoluted_feat1 = self.conv1(query_document_product)
        convoluted_feat2 = self.conv2(query_document_product)
        convoluted_feat3 = self.conv3(query_document_product)
        convoluted_feat = self.relu(torch.cat((convoluted_feat1, convoluted_feat2, convoluted_feat3), 1))
        convoluted_feat = self.conv(convoluted_feat).transpose(1, 2).transpose(2, 3)

        max_pooled_feat = torch.max(convoluted_feat, 2)[0].squeeze()
        max_pooled_feat = torch.max(max_pooled_feat, 1)[0].squeeze()
        return self.output(max_pooled_feat).squeeze().view(*batch_docs.size()[:2])
