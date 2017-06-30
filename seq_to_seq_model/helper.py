###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 5/20/2017
#
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments.
###############################################################################

import re, os, glob, pickle, string, math, time, util, torch, shutil
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk import wordpunct_tokenize
from numpy.linalg import norm
from collections import OrderedDict
from torch.autograd import Variable


def normalize_word_embedding(v):
    return np.array(v) / norm(np.array(v))


def load_word_embeddings(directory, file):
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        try:
            values = line.split()
            word = values[0]
            embeddings_index[word] = normalize_word_embedding([float(x) for x in values[1:]])
        except ValueError as e:
            print(e)
    f.close()
    return embeddings_index


def save_word_embeddings(directory, file, embeddings_index, words):
    f = open(os.path.join(directory, file), 'w')
    for word in words:
        if word in embeddings_index:
            f.write(word + '\t' + '\t'.join(str(x) for x in embeddings_index[word]) + '\n')
    f.close()


def save_model_states(model, loss, epoch, snapshot_prefix):
    """Save a deep learning network's states in a file."""
    snapshot_path = snapshot_prefix + '_loss_{:.6f}_epoch_{}_model.pt'.format(loss, epoch)
    with open(snapshot_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
            os.remove(f)


def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)


def load_model_states(model, filename):
    assert os.path.exists(filename)
    with open(filename, 'rb') as f:
        model.load_state_dict(torch.load(f))


def load_model_states_from_checkpoint(model, filename, tag):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint[tag])


def load_model_states_without_dataparallel(model, filename):
    """Load a previously saved model states."""
    assert os.path.exists(filename)
    with open(filename, 'rb') as f:
        state_dict = torch.load(f, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def save_object(obj, filename):
    """Save an object into file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    """Load object from file."""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def tokenize_and_normalize(s):
    """Tokenize and normalize string."""
    token_list = []
    tokens = wordpunct_tokenize(s.lower())
    token_list.extend([x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)])
    return token_list


def initialize_out_of_vocab_words(dimension):
    """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
    return np.random.normal(size=dimension)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def mask(sequence_length, seq_idx):
    batch_size = sequence_length.size(0)
    seq_range = torch.LongTensor([seq_idx])
    seq_range_expand = seq_range.expand(batch_size)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    return seq_range_expand < sequence_length


def batchify(data, bsz):
    """Transform data into batches."""
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    batched_data = [[data[bsz * i + j] for j in range(bsz)] for i in range(nbatch)]
    return batched_data


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))


def save_plot(points, filepath, filetag, epoch):
    """Generate and save the plot"""
    path_prefix = os.path.join(filepath, filetag + '_loss_plot_')
    path = path_prefix + 'epoch_{}.png'.format(epoch)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(path)
    plt.close(fig)  # close the figure
    for f in glob.glob(path_prefix + '*'):
        if f != path:
            os.remove(f)


def show_plot(points):
    """Generates plots"""
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def sentence_to_tensor(sentence, max_sent_length, dictionary):
    """Convert a sequence of words to a tensor of word indices."""
    sen_rep = torch.LongTensor(max_sent_length).zero_()
    for i in range(len(sentence)):
        word = sentence[i]
        if word in dictionary.word2idx:
            sen_rep[i] = dictionary.word2idx[word]
        else:
            sen_rep[i] = dictionary.word2idx[dictionary.unknown_token]
    return sen_rep


def instances_to_tensors(instances, dictionary, max_sent_length):
    """Convert a list of sequences to a list of tensors."""
    all_sentences1 = torch.LongTensor(len(instances), max_sent_length)
    all_sentences2 = torch.LongTensor(len(instances), max_sent_length)
    for i in range(len(instances)):
        all_sentences1[i] = sentence_to_tensor(instances[i].sentence1, max_sent_length, dictionary)
        all_sentences2[i] = sentence_to_tensor(instances[i].sentence2, max_sent_length, dictionary)
    return Variable(all_sentences1), Variable(all_sentences2)


def queries_to_tensors(instances, dictionary):
    max_query_length1 = 0
    max_query_length2 = 0
    for item in instances:
        if max_query_length1 < len(item.sentence1):
            max_query_length1 = len(item.sentence1)
        if max_query_length2 < len(item.sentence2):
            max_query_length2 = len(item.sentence2)

    all_sentences1 = torch.LongTensor(len(instances), max_query_length1)
    all_sentences2 = torch.LongTensor(len(instances), max_query_length2)
    length = torch.LongTensor(len(instances))
    for i in range(len(instances)):
        all_sentences1[i] = sentence_to_tensor(instances[i].sentence1, max_query_length1, dictionary)
        all_sentences2[i] = sentence_to_tensor(instances[i].sentence2, max_query_length2, dictionary)
        length[i] = len(instances[i].sentence2) - 1
    return Variable(all_sentences1), Variable(all_sentences2), Variable(length)


def show_attention_plot(input_sentence, output_words, attentions):
    """Shows attention as a graphical plot"""
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def save_attention_plot(input_sentence, output_words, attentions, filename):
    """Save attention as a graphical plot"""
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(filename)
    plt.close(fig)  # close the figure
