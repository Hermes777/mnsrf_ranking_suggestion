###############################################################################
# Author: Wasi Ahmad
# Project: Deep Semantic Similarity Model
# Date Created: 7/18/2017
#
# File Description: This script contains all the command line arguments.
###############################################################################

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='seq2seq_language_model')
    parser.add_argument('--data', type=str, default='../data/session_with_clicks_v5/',
                        help='location of the data corpus')
    parser.add_argument('--nhid_output', type=int, default=128,
                        help='number of hidden units in the output layer')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units in the hidden layers')
    parser.add_argument('--lr', type=float, default=.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper limit of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                        help='batch size')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA for computation')
    parser.add_argument('--print_every', type=int, default=1000,
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=1000,
                        help='plotting interval')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='resume from last checkpoint (default: none)')
    parser.add_argument('--save_path', type=str, default='../output_dssm/',
                        help='path to save the final model')

    args = parser.parse_args()
    return args
