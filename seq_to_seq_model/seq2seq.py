###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper, nn_layer
import torch.nn as nn
from torch.autograd import Variable


class Sequence2Sequence(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(Sequence2Sequence, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args
        self.embedding = nn_layer.EmbeddingLayer(len(dictionary), self.config.emsize, self.config.dropout)
        self.encoder = nn_layer.RNN(self.config.model, self.config.emsize, self.config.nhid, self.config.nlayers,
                                    self.config.dropout, True)
        self.decoder = nn_layer.RNN(self.config.model, self.config.emsize + self.config.nhid, self.config.nhid,
                                    self.config.nlayers, self.config.dropout)
        self.attention = nn_layer.ApplyAttention(len(dictionary), self.config.nhid)

        # Initializing the weight parameters for the embedding layer.
        self.embedding.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    @staticmethod
    def compute_loss(logits, target, seq_idx, length, regularization_param=None):
        # logits: batch x vocab_size, target: batch x 1
        losses = -torch.gather(logits, dim=1, index=target.unsqueeze(1))
        # mask: batch x 1
        mask = helper.mask(length, seq_idx)
        losses = losses * mask.float()
        num_non_zero_elem = torch.nonzero(mask.data).size()
        if not num_non_zero_elem:
            loss = losses.sum()
        else:
            loss = losses.sum() / num_non_zero_elem[0]
        if regularization_param:
            regularized_loss = logits.exp().mul(logits).sum(1).squeeze() * regularization_param
            loss += regularized_loss.mean()
        return loss

    def forward(self, batch_sentence1, batch_sentence2, length):
        """"Defines the forward computation of the question classifier."""
        embedded = self.embedding(batch_sentence1)
        if self.config.model == 'LSTM':
            init_hidden, init_cell = self.encoder.init_weights(batch_sentence1.size(0))
            encoder_output, encoder_hidden = self.encoder(embedded, (init_hidden, init_cell))
        else:
            init_hidden = self.encoder.init_weights(batch_sentence1.size(0))
            encoder_output, encoder_hidden = self.encoder(embedded, init_hidden)

        if self.config.bidirection:
            encoder_hidden = torch.mean(encoder_hidden[0], 0), torch.mean(encoder_hidden[1], 0)
            encoder_output = torch.div(
                torch.add(encoder_output[:, :, 0:self.config.nhid],
                          encoder_output[:, :, self.config.nhid:2 * self.config.nhid]), 2)

        # Initialize hidden states of decoder with the last hidden states of the encoder
        decoder_hidden = encoder_hidden
        context_vector = Variable(torch.zeros(batch_sentence2.size(0), 1, self.config.nhid))
        if self.config.cuda:
            context_vector = context_vector.cuda()

        loss = 0
        for idx in range(batch_sentence2.size(1) - 1):
            # Use the real target outputs as each next input (teacher forcing)
            input_variable = batch_sentence2[:, idx]
            target_variable = batch_sentence2[:, idx + 1]

            embedded_input = self.embedding(input_variable).unsqueeze(1)
            embedded_input = torch.cat((embedded_input, context_vector), 2)
            decoder_output, decoder_hidden = self.decoder(embedded_input, decoder_hidden)
            output, context_vector, attn_weights = self.attention(decoder_output, encoder_output)
            loss += self.compute_loss(output, target_variable, idx, length)

        return loss
