###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 6/20/2017
#
# File Description: This script contains code related to the neural network layers.
###############################################################################

import torch, helper
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


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

    def __init__(self, input_size, hidden_size, config):
        """"Constructor of the class"""
        super(Encoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(self.config.dropout)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout,
                                                      bidirectional=self.config.bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout, bidirectional=self.config.bidirection)

    def forward(self, input, hidden):
        """"Defines the forward computation of the encoder"""
        output = input
        for i in range(self.config.nlayers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        num_directions = 2 if self.config.bidirection else 1
        if self.config.model == 'LSTM':
            return Variable(weight.new(self.config.nlayers * num_directions, bsz, self.hidden_size).zero_()), Variable(
                weight.new(self.config.nlayers * num_directions, bsz, self.hidden_size).zero_())
        else:
            return Variable(weight.new(self.n_layers * num_directions, bsz, self.hidden_size).zero_())


class Decoder(nn.Module):
    """Decoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, hidden_size, output_size, config):
        """"Constructor of the class"""
        super(Decoder, self).__init__()
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(self.config.dropout)
        self.out = nn.Linear(hidden_size, output_size)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(self.input_size, self.hidden_size, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout)

    def forward(self, input, hidden):
        """"Defines the forward computation of the decoder"""
        output = input
        for i in range(self.config.nlayers):
            output, hidden = self.rnn(output, hidden)
            output = self.drop(output)
        output = F.log_softmax(self.out(output.squeeze(1)))
        return output, hidden
