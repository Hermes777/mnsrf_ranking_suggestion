###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 6/20/2017
#
# File Description: This script contains code to train the model.
###############################################################################

import time, helper, torch

import torch.nn as nn
from torch.nn.utils import clip_grad_norm


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, dictionary, embeddings_index, config, best_loss):
        self.model = model
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.config = config
        self.criterion = nn.NLLLoss()  # Negative log-likelihood loss
        self.lr = config.lr

        self.optimizer = optimizer
        self.best_dev_loss = best_loss
        self.times_no_improvement = 0
        self.stop = False
        self.train_losses = []
        self.dev_losses = []

    def train_epochs(self, train_batches, dev_batches, start_epoch, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(start_epoch, start_epoch + n_epochs):
            if not self.stop:
                self.train(train_batches, dev_batches, (epoch + 1))
                helper.save_plot(self.train_losses, self.config.save_path, 'training', epoch + 1)
                helper.save_plot(self.dev_losses, self.config.save_path, 'dev', epoch + 1)
            else:
                break

    def train(self, train_batches, dev_batches, epoch_no):
        # Turn on training mode which enables dropout.
        self.model.train()

        start = time.time()
        print_loss_total = 0
        plot_loss_total = 0

        num_batches = len(train_batches)
        print('epoch %d started' % epoch_no)

        for batch_no in range(1, num_batches + 1):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            train_sessions, length = helper.session_to_tensor(train_batches[batch_no - 1], self.dictionary)
            if self.config.cuda:
                train_sessions = train_sessions.cuda()
                length = length.cuda()

            loss = self.model(train_sessions, length)
            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            clip_grad_norm(self.model.parameters(), self.config.clip)
            self.optimizer.step()

            print_loss_total += loss.data[0]
            plot_loss_total += loss.data[0]

            if batch_no % self.config.print_every == 0:
                print_loss_avg = print_loss_total / self.config.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_loss_avg))

            if batch_no % self.config.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.config.plot_every
                self.train_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if batch_no % self.config.dev_every == 0:
                dev_loss = self.validate(dev_batches)
                self.dev_losses.append(dev_loss)
                print('validation loss = %.4f' % dev_loss)
                if self.best_dev_loss == -1 or self.best_dev_loss > dev_loss:
                    self.best_dev_loss = dev_loss
                    helper.save_checkpoint({
                        'epoch': epoch_no,
                        'state_dict': self.model.state_dict(),
                        'best_loss': self.best_dev_loss,
                        'optimizer': self.optimizer.state_dict(),
                    }, self.config.save_path + 'model_best.pth.tar')
                else:
                    self.times_no_improvement += 1
                    # no improvement in validation loss for last n times, so stop training
                    if self.times_no_improvement == 20:
                        self.stop = True
                        break

    def validate(self, dev_batches):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        dev_loss = 0
        num_batches = len(dev_batches)
        for batch_no in range(1, num_batches + 1):
            dev_sessions, length = helper.session_to_tensor(dev_batches[batch_no - 1], self.dictionary)
            if self.config.cuda:
                dev_sessions = dev_sessions.cuda()
                length = length.cuda()

            loss = self.model(dev_sessions, length)
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            dev_loss += loss.data[0]

        # Turn on training mode at the end of validation.
        self.model.train()

        return dev_loss / num_batches
