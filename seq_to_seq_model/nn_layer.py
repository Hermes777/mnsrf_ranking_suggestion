###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 5/20/2017
#
# File Description: This script contains code related to different neural
# network layer classes.
###############################################################################

import helper, torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    """Embedding class which includes only an embedding layer."""

    def __init__(self, input_size, emsize, dropout, train_embeddings=False):
        """"Constructor of the class"""
        super(EmbeddingLayer, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, emsize)
        if not train_embeddings:
            self.embedding.weight.requires_grad = False

    def forward(self, input_variable):
        """"Defines the forward computation of the embedding layer."""
        embedded = self.embedding(input_variable)
        embedded = self.drop(embedded)
        return embedded

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


class RNN(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, model, input_size, hidden_size, num_layers, dropout, bidirection=False):
        """"Constructor of the class"""
        super(RNN, self).__init__()
        self.model = model
        self.emsize = input_size
        self.nhid = hidden_size
        self.nlayers = num_layers
        self.dropout = dropout
        self.num_directions = 2 if bidirection else 1
        self.drop = nn.Dropout(self.dropout)

        if self.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.model)(input_size, self.nhid, self.nlayers,
                                               batch_first=True, dropout=self.dropout,
                                               bidirectional=bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, self.nhid, self.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.dropout, bidirectional=bidirection)

    def forward(self, input, hidden):
        """"Defines the forward computation of the encoder"""
        output = input
        for i in range(self.nlayers):
            output, hidden = self.rnn(output, hidden)
            output = self.drop(output)
        return output, hidden

    def init_weights(self, bsz):
        weight = next(self.parameters()).data
        if self.model == 'LSTM':
            return Variable(
                weight.new(self.nlayers * self.num_directions, bsz, self.nhid).zero_()), Variable(
                weight.new(self.nlayers * self.num_directions, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.n_layers * self.num_directions, bsz, self.nhid).zero_())


class ApplyAttention(nn.Module):
    """Decoder class of a sequence-to-sequence network"""

    def __init__(self, output_size, hidden_size, method='general', attention_type='global'):
        """"Constructor of the class"""
        super(ApplyAttention, self).__init__()
        self.nhid = hidden_size
        self.method = method
        self.attn_combine = nn.Linear(self.nhid * 2, self.nhid)
        self.out = nn.Linear(self.nhid, output_size)
        if attention_type == 'global':
            """global attention mechanism described in paper - http://aclweb.org/anthology/D15-1166"""
            if self.method == 'general':
                self.weight = nn.Parameter(torch.Tensor(self.nhid, self.nhid))
                init.xavier_normal(self.weight)
            elif self.method == 'concat':
                self.attn = nn.Linear(self.nhid * 2, self.nhid)
                self.weight = nn.Parameter(torch.Tensor(1, self.nhid))
                init.xavier_normal(self.weight)

    def forward(self, decoder_out, encoder_outputs):
        """"Defines the forward computation of the attention mechanism"""
        attn_weights = self.compute_attention_weights(decoder_out, encoder_outputs)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        attention_combine = self.attn_combine(torch.cat((context_vector.squeeze(1), decoder_out.squeeze(1)), 1))
        output = F.log_softmax(self.out(attention_combine))
        return output, context_vector, attn_weights

    def compute_attention_weights(self, decoder_out, encoder_outputs):
        if self.method == 'dot':
            score = torch.bmm(decoder_out, torch.transpose(encoder_outputs, 1, 2))
            return F.softmax(score.squeeze(1))
        elif self.method == 'general':
            weighted_encoder_output = torch.bmm(self.weight.expand(encoder_outputs.size(0), *self.weight.size()),
                                                torch.transpose(encoder_outputs, 1, 2))
            score = torch.bmm(decoder_out, weighted_encoder_output)
            return F.softmax(score.squeeze(1))
        elif self.method == 'concat':
            concatenated_rep = torch.cat((decoder_out.expand(decoder_out.size(0), encoder_outputs.size(1),
                                                             decoder_out.size(2)), encoder_outputs), 2)
            attn_applied = torch.tanh(self.attn(concatenated_rep.view(-1, concatenated_rep.size(2))))
            score = torch.sum(torch.mul(self.weight.expand(*attn_applied.size()), attn_applied), 1)
            return F.softmax(score.view(decoder_out.size(0), -1))
