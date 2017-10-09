###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################

import util, data, train, os, numpy, helper
import torch
from torch import optim
from preprocessing import Vocabulary
from duet import DUET

args = util.get_args()
# if output directory doesn't exist, create it
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# Set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Create dictionary and save it for future computation
###############################################################################

# vocab = Vocabulary()
# vocab.form_vocabulary(args.data, 'session_train.txt')
# print('dictionary size - ', len(vocab))
# vocab.save_vocabulary(args.save_path, 'vocab.csv')

###############################################################################
# Load data
###############################################################################

dictionary = data.Dictionary(5)
dictionary.load_dictionary(args.save_path, 'vocab.csv', 5000)
print('vocabulary size = ', len(dictionary))

train_corpus = data.Corpus(args.data, 'session_train.txt', dictionary)
print('train set size = ', len(train_corpus.data))

dev_corpus = data.Corpus(args.data, 'session_dev.txt', dictionary)
print('dev set size = ', len(dev_corpus.data))

###############################################################################
# Build the model
###############################################################################

model = DUET(dictionary, args)
optimizer = optim.SGD(model.parameters(), args.lr)
best_loss = -1

param_dict = helper.count_parameters(model)
print('Number of trainable parameters = ', numpy.sum(list(param_dict.values())))

# for training on multiple GPUs. use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    if len(cuda_visible_devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=cuda_visible_devices)
if args.cuda:
    model = model.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

###############################################################################
# Train the model
###############################################################################

train = train.Train(model, optimizer, dictionary, args, best_loss)
train.train_epochs(train_corpus, dev_corpus, args.start_epoch, args.epochs)
