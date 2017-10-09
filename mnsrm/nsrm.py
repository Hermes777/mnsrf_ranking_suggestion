###############################################################################
# Author: Wasi Ahmad
# Project: Neural Session Relevance Model
# Date Created: 7/15/2017
#
# File Description: This script contains code related to the sequence-to-sequence
# network.
###############################################################################

import torch, helper
import torch.nn as nn
from torch.autograd import Variable
from nn_layer import EmbeddingLayer, Encoder, Decoder


class NSRM(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embedding_index, args):
        """"Constructor of the class."""
        super(NSRM, self).__init__()
        self.dictionary = dictionary
        self.embedding_index = embedding_index
        self.config = args
        self.num_directions = 2 if self.config.bidirection else 1
        self.embedding = EmbeddingLayer(len(self.dictionary), self.config)
        self.query_encoder = Encoder(self.config.emsize, self.config.nhid_query, True, self.config)
        self.document_encoder = Encoder(self.config.emsize, self.config.nhid_doc, True, self.config)
        self.session_encoder = Encoder(self.config.nhid_query * self.num_directions, self.config.nhid_session,
                                       False, self.config)
        self.projection = nn.Linear((self.config.nhid_query * self.num_directions) + self.config.nhid_session,
                                    self.config.nhid_doc * self.num_directions)
        self.decoder = Decoder(self.config.emsize, self.config.nhid_session, len(self.dictionary), self.config)

        # Initializing the weight parameters for the embedding layer.
        self.embedding.init_embedding_weights(self.dictionary, self.embedding_index, self.config.emsize)

    @staticmethod
    def compute_decoding_loss(logits, target, seq_idx, length):
        """
        Compute negative log-likelihood loss for a batch of predictions.
        :param logits: 2d tensor [batch_size x vocab_size]
        :param target: 2d tensor [batch_size x 1]
        :param seq_idx: an integer represents the current index of the sequences
        :param length: 1d tensor [batch_size], represents each sequences' true length
        :return: total loss over the input mini-batch [autograd Variable] and number of loss elements
        """
        losses = -torch.gather(logits, dim=1, index=target.unsqueeze(1))
        mask = helper.mask(length, seq_idx)  # mask: batch x 1
        losses = losses * mask.float()
        num_non_zero_elem = torch.nonzero(mask.data).size()
        if not num_non_zero_elem:
            return losses.sum(), 0
        else:
            return losses.sum(), num_non_zero_elem[0]

    @staticmethod
    def compute_click_loss(logits, target):
        """
        Compute logistic loss for a batch of clicks. Return average loss for the input mini-batch.
        :param logits: 2d tensor [batch_size x num_clicks_per_query]
        :param target: 2d tensor [batch_size x num_clicks_per_query]
        :return: average loss over batch [autograd Variable]
        """
        # taken from https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L695
        neg_abs = - logits.abs()
        loss = logits.clamp(min=0) - logits * target + (1 + neg_abs.exp()).log()
        return loss.mean()

    def forward(self, batch_session, length, batch_clicks, click_labels):
        """
        Forward function of the neural click model. Return average loss for a batch of sessions.
        :param batch_session: 3d tensor [batch_size x session_length x max_query_length]
        :param length: 2d tensor [batch_size x session_length]
        :param batch_clicks: 4d tensor [batch_size x session_length x num_rel_docs_per_query x max_document_length]
        :param click_labels: 3d tensor [batch_size x session_length x num_rel_docs_per_query]
        :return: average loss over batch [autograd Variable]
        """
        # query level encoding
        embedded_queries = self.embedding(batch_session.view(-1, batch_session.size(-1)))
        if self.config.model == 'LSTM':
            encoder_hidden, encoder_cell = self.query_encoder.init_weights(embedded_queries.size(0))
            output, hidden = self.query_encoder(embedded_queries, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = self.query_encoder.init_weights(embedded_queries.size(0))
            output, hidden = self.query_encoder(embedded_queries, encoder_hidden)

        encoded_queries = torch.max(output, 1)[0].squeeze(1)
        # encoded_queries = batch_size x num_queries_in_a_session x hidden_size
        encoded_queries = encoded_queries.view(*batch_session.size()[:-1], -1)

        # document level encoding
        embedded_clicks = self.embedding(batch_clicks.view(-1, batch_clicks.size(-1)))
        if self.config.model == 'LSTM':
            encoder_hidden, encoder_cell = self.document_encoder.init_weights(embedded_clicks.size(0))
            output, hidden = self.document_encoder(embedded_clicks, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = self.document_encoder.init_weights(embedded_clicks.size(0))
            output, hidden = self.document_encoder(embedded_clicks, encoder_hidden)

        encoded_clicks = torch.max(output, 1)[0].squeeze(1)
        # encoded_clicks = batch_size x num_queries_in_a_session x num_rel_docs_per_query x hidden_size
        encoded_clicks = encoded_clicks.view(*batch_clicks.size()[:-1], -1)

        # session level encoding
        sess_hidden = self.session_encoder.init_weights(encoded_queries.size(0))
        sess_output = Variable(torch.zeros(self.config.batch_size, 1, self.config.nhid_session))
        if self.config.cuda:
            sess_output = sess_output.cuda()
        hidden_states, cell_states = [], []
        click_loss = 0
        for idx in range(encoded_queries.size(1)):
            combined_rep = torch.cat((sess_output.squeeze(), encoded_queries[:, idx, :]), 1)
            combined_rep = self.projection(combined_rep)
            combined_rep = combined_rep.unsqueeze(1).expand(*encoded_clicks[:, idx, :, :].size())
            click_score = torch.sum(torch.mul(combined_rep, encoded_clicks[:, idx, :, :]), 2).squeeze(2)
            click_loss += self.compute_click_loss(click_score, click_labels[:, idx, :])
            # update session state using query representations
            sess_output, sess_hidden = self.session_encoder(encoded_queries[:, idx, :].unsqueeze(1), sess_hidden)
            hidden_states.append(sess_hidden[0])
            cell_states.append(sess_hidden[1])

        click_loss = click_loss / encoded_queries.size(1)

        hidden_states = torch.stack(hidden_states, 2).squeeze(0)
        cell_states = torch.stack(cell_states, 2).squeeze(0)

        # decoding in sequence-to-sequence learning
        hidden_states = hidden_states[:, :-1, :].contiguous().view(-1, hidden_states.size(-1)).unsqueeze(0)
        cell_states = cell_states[:, :-1, :].contiguous().view(-1, cell_states.size(-1)).unsqueeze(0)
        decoder_input = batch_session[:, 1:, :].contiguous().view(-1, batch_session.size(-1))
        target_length = length[:, 1:].contiguous().view(-1)
        input_variable = Variable(
            torch.LongTensor(decoder_input.size(0)).fill_(self.dictionary.word2idx[self.dictionary.start_token]))
        if self.config.cuda:
            input_variable = input_variable.cuda()
            hidden_states = hidden_states.cuda()
            cell_states = cell_states.cuda()

        # Initialize hidden states of decoder with the last hidden states of the session encoder
        decoder_hidden = (hidden_states, cell_states)
        decoding_loss = 0
        total_local_decoding_loss_element = 0
        for idx in range(decoder_input.size(1)):
            if idx != 0:
                input_variable = decoder_input[:, idx - 1]
            embedded_decoder_input = self.embedding(input_variable).unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(embedded_decoder_input, decoder_hidden)
            target_variable = decoder_input[:, idx]
            local_loss, num_local_loss = self.compute_decoding_loss(decoder_output, target_variable, idx, target_length)
            decoding_loss += local_loss
            total_local_decoding_loss_element += num_local_loss

        if total_local_decoding_loss_element > 0:
            decoding_loss = decoding_loss / total_local_decoding_loss_element

        return click_loss + decoding_loss
