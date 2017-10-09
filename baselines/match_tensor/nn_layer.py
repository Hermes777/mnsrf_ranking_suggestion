###############################################################################
# Author: Wasi Ahmad
# Project: Match Tensor: a Deep Relevance Model for Search
# Date Created: 7/28/2017
#
# File Description: This script contains code related to the neural network layers.
###############################################################################

import torch, helper
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable


class EmbeddingLayer(nn.Module):
    """Embedding class which includes only an embedding layer."""

    def __init__(self, input_size, config):
        """"Constructor of the class"""
        super(EmbeddingLayer, self).__init__()
        self.drop = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(input_size, config.emsize)
        # self.embedding.weight.requires_grad = False

    def forward(self, input_variable):
        """"Defines the forward computation of the embedding layer."""
        return self.drop(self.embedding(input_variable))

    def init_embedding_weights(self, dictionary, embeddings_index, embedding_dim):
        """Initialize weight parameters for the embedding layer."""
        pretrained_weight = np.empty([len(dictionary), embedding_dim], dtype=float)
        for i in range(len(dictionary)):
            if dictionary.idx2word[i] in embeddings_index:
                pretrained_weight[i] = embeddings_index[dictionary.idx2word[i]]
            else:
                pretrained_weight[i] = helper.initialize_out_of_vocab_words(embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))


class Encoder(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, hidden_size, bidirection, config):
        """"Constructor of the class"""
        super(Encoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirection = bidirection
        self.drop = nn.Dropout(self.config.dropout)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout,
                                                      bidirectional=self.bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout, bidirectional=self.bidirection)

    def forward(self, input, hidden):
        """"Defines the forward computation of the encoder"""
        output = input
        for i in range(self.config.nlayers):
            output, hidden = self.rnn(output, hidden)
            output = self.drop(output)
        return output, hidden

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        num_directions = 2 if self.bidirection else 1
        if self.config.model == 'LSTM':
            return Variable(weight.new(self.config.nlayers * num_directions, bsz, self.hidden_size).zero_()), Variable(
                weight.new(self.config.nlayers * num_directions, bsz, self.hidden_size).zero_())
        else:
            return Variable(weight.new(self.n_layers * num_directions, bsz, self.hidden_size).zero_())


class ExactMatchChannel(nn.Module):
    """Exact match channel layer for the match tensor"""

    def __init__(self):
        """"Constructor of the class"""
        super(ExactMatchChannel, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1))
        # Initializing the value of alpha
        init.uniform(self.alpha)

    def forward(self, batch_query, batch_docs):
        """"Computes the exact match channel"""
        query_tensor = batch_query.unsqueeze(1).expand(batch_query.size(0), batch_docs.size(1), batch_query.size(1))
        query_tensor = query_tensor.contiguous().view(-1, query_tensor.size(2))
        doc_tensor = batch_docs.view(-1, batch_docs.size(2))

        query_tensor = query_tensor.unsqueeze(2).expand(*query_tensor.size(), batch_docs.size(2))
        doc_tensor = doc_tensor.unsqueeze(1).expand(doc_tensor.size(0), batch_query.size(1), doc_tensor.size(1))

        exact_match = (query_tensor == doc_tensor).float()
        return exact_match * self.alpha.expand(exact_match.size())
