###############################################################################
# Author: Wasi Ahmad
# Project: Convolutional Latent Semantic Model
# Date Created: 7/18/2017
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################

import util, helper, data, train, os, numpy
import torch
from torch import optim
from clsm import CLSM

args = util.get_args()
# if output directory doesn't exist, create it
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# set the random seed manually for reproducibility.
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

dictionary = data.Dictionary()
train_corpus = data.Corpus(args.data, 'session_train.txt', dictionary)
print('train set size = ', len(train_corpus.data))
print('vocabulary size = ', len(dictionary))
dev_corpus = data.Corpus(args.data, 'session_dev.txt', dictionary, is_test_corpus=True)
print('dev set size = ', len(dev_corpus.data))

# save the dictionary object to use during testing
helper.save_object(dictionary, args.save_path + 'dictionary.p')

# ###############################################################################
# # Build the model
# ###############################################################################

model = CLSM(dictionary, args)
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

# ###############################################################################
# # Train the model
# ###############################################################################

train = train.Train(model, optimizer, dictionary, args, best_loss)
train.train_epochs(train_corpus, dev_corpus, args.start_epoch, args.epochs)
