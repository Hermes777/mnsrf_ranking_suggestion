###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 6/20/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper
import torch.nn as nn
from torch.autograd import Variable
from nn_layer import EmbeddingLayer, Encoder, Decoder


class Sequence2Sequence(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(Sequence2Sequence, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args
        self.embedding = EmbeddingLayer(len(self.dictionary), self.config)
        self.query_encoder = Encoder(self.config.emsize, self.config.nhid_query, self.config)
        self.session_encoder = Encoder(self.config.nhid_query, self.config.nhid_session, self.config)
        self.decoder = Decoder(self.config.emsize, self.config.nhid_session, len(self.dictionary), self.config)

        # Initializing the weight parameters for the embedding layer.
        self.embedding.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    @staticmethod
    def compute_loss(logits, target, seq_idx, length):
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
        return loss

    def forward(self, batch_session, length):
        """"Defines the forward computation of the question classifier."""
        embedded_input = self.embedding(batch_session.view(-1, batch_session.size(-1)))
        if self.config.model == 'LSTM':
            encoder_hidden, encoder_cell = self.query_encoder.init_weights(embedded_input.size(0))
            output, hidden = self.query_encoder(embedded_input, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = self.query_encoder.init_weights(embedded_input.size(0))
            output, hidden = self.query_encoder(embedded_input, encoder_hidden)

        if self.config.bidirection:
            output = torch.div(
                torch.add(output[:, :, 0:self.config.nhid_query],
                          output[:, :, self.config.nhid_query:2 * self.config.nhid_query]), 2)

        session_input = output[:, -1, :].contiguous().view(batch_session.size(0), batch_session.size(1), -1)
        # session level encoding
        sess_hidden = self.session_encoder.init_weights(session_input.size(0))
        hidden_states, cell_states = [], []
        for idx in range(session_input.size(1)):
            sess_output, sess_hidden = self.session_encoder(session_input[:, idx, :].unsqueeze(1), sess_hidden)
            if self.config.bidirection:
                hidden_states.append(torch.mean(sess_hidden[0], 0))
                cell_states.append(torch.mean(sess_hidden[1], 0))
            else:
                hidden_states.append(sess_hidden[0])
                cell_states.append(sess_hidden[1])

        hidden_states = torch.stack(hidden_states, 2).squeeze(0)
        cell_states = torch.stack(cell_states, 2).squeeze(0)
        hidden_states = hidden_states[:, :-1, :].contiguous().view(-1, hidden_states.size(-1)).unsqueeze(0)
        cell_states = cell_states[:, :-1, :].contiguous().view(-1, cell_states.size(-1)).unsqueeze(0)

        decoder_input = batch_session[:, 1:, :].contiguous().view(-1, batch_session.size(-1))
        target_length = length[:, 1:].contiguous().view(-1)
        input_variable = Variable(torch.LongTensor(decoder_input.size(0)).fill_(
            self.dictionary.word2idx[self.dictionary.start_token]))
        if self.config.cuda:
            input_variable = input_variable.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()

        # Initialize hidden states of decoder with the last hidden states of the session encoder
        decoder_hidden = (hidden_states, cell_states)
        loss = 0
        for idx in range(decoder_input.size(1)):
            if idx != 0:
                input_variable = decoder_input[:, idx - 1]
            embedded_decoder_input = self.embedding(input_variable).unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(embedded_decoder_input, decoder_hidden)
            target_variable = decoder_input[:, idx]
            loss += self.compute_loss(decoder_output, target_variable, idx, target_length)

        return loss
