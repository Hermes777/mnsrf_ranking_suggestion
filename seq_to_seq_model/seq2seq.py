###############################################################################
# Author: Wasi Ahmad
# Project: Learning Vision to Language
# Date Created: 4/02/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper
import torch.nn as nn
from torch.autograd import Variable
from encoder import Encoder
from decoder import Decoder, GloballyAttentiveDecoder


class Sequence2Sequence(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(Sequence2Sequence, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args
        self.encoder = Encoder(len(self.dictionary), self.config)
        self.decoder = GloballyAttentiveDecoder(len(self.dictionary), self.config)
        # self.decoder = Decoder(len(self.dictionary), self.config)
        self.criterion = nn.NLLLoss()  # Negative log-likelihood loss

        # Initializing the weight parameters for the embedding layer in the encoder and decoder.
        self.encoder.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)
        self.decoder.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    def compute_loss(self, logits, target, seq_idx, length, regularization_param=None):
        # logits: batch x vocab_size, target: batch x 1
        losses = -torch.gather(logits, dim=1, index=target.unsqueeze(1))
        # mask: batch x 1
        mask = helper.mask(length, seq_idx)
        losses = losses * mask.float()
        loss = losses.mean()
        if regularization_param:
            regularized_loss = logits.exp().mul(logits).sum(1).squeeze() * regularization_param
            loss += regularized_loss.mean()
        return loss

    def forward(self, batch_sentence1, batch_sentence2, length):
        """"Defines the forward computation of the question classifier."""
        if self.config.model == 'LSTM':
            encoder_hidden, encoder_cell = self.encoder.init_weights(batch_sentence1.size(0))
            output, hidden = self.encoder(batch_sentence1, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = self.encoder.init_weights(batch_sentence1.size(0))
            output, hidden = self.encoder(batch_sentence1, encoder_hidden)

        if self.config.bidirection:
            hidden = torch.mean(hidden[0], 0), torch.mean(hidden[1], 0)
            output = torch.div(
                torch.add(output[:, :, 0:self.config.nhid], output[:, :, self.config.nhid:2 * self.config.nhid]), 2)

        # Initialize hidden states of decoder with the last hidden states of the encoder
        decoder_hidden = hidden
        decoder_context = Variable(torch.zeros(batch_sentence2.size(0), 1, self.config.nhid))
        if self.config.cuda:
            decoder_context = decoder_context.cuda()

        loss = 0
        for idx in range(batch_sentence2.size(1) - 1):
            # Use the real target outputs as each next input (teacher forcing)
            input_variable = batch_sentence2[:, idx]
            target_variable = batch_sentence2[:, idx + 1]
            decoder_output, decoder_hidden, decoder_context, decoder_attention = self.decoder(input_variable,
                                                                                              decoder_hidden,
                                                                                              decoder_context,
                                                                                              output)
            # decoder_output, decoder_hidden = self.decoder(input_variable, decoder_hidden)
            loss += self.compute_loss(decoder_output, target_variable, idx, length, 0.1)

        return loss
