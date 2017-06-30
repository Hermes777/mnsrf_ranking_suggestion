###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 6/20/2017
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################

import util, helper, data, train, os
import torch
from torch import optim
from seq2seq import Sequence2Sequence

args = util.get_args()
# Set the random seed manually for reproducibility.
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
train_corpus = data.Corpus(args.data, 'session_train.txt', dictionary, args.max_length)
dev_corpus = data.Corpus(args.data, 'session_dev.txt', dictionary, args.max_length)
# test_corpus = data.Corpus(args.data, 'session_test.txt', dictionary, args.max_length, is_test_corpus=True)
print('Train set size = ', len(train_corpus))
print('Max session length in train corpus = ', train_corpus.max_session_length)
print('Dev set size = ', len(dev_corpus))
print('Max session length in dev corpus = ', dev_corpus.max_session_length)
print('Vocabulary size = ', len(dictionary))

# save the dictionary object to use during testing
helper.save_object(dictionary, args.save_path + 'dictionary.p')

# embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file)
# helper.save_word_embeddings('../data/glove/', 'glove.840B.300d.s2s.txt', embeddings_index, dictionary.idx2word)

embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, 'glove.840B.300d.s2s.txt')
print('Number of OOV words = ', len(dictionary) - len(embeddings_index))

# Splitting the data in batches
train_batches = helper.batchify(train_corpus.data, args.batch_size)
print('Number of train batches = ', len(train_batches))
dev_batches = helper.batchify(dev_corpus.data, args.batch_size)
print('Number of dev batches = ', len(dev_batches))

# ###############################################################################
# # Build the model
# ###############################################################################

model = Sequence2Sequence(dictionary, embeddings_index, args)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
best_loss = -1

# for training on multiple GPUs. use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
# if 'CUDA_VISIBLE_DEVICES' in os.environ:
#     cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
#     model = torch.nn.DataParallel(model, device_ids=cuda_visible_devices)
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

train = train.Train(model, optimizer, dictionary, embeddings_index, args, best_loss)
train.train_epochs(train_batches, dev_batches, args.start_epoch, args.epochs)
