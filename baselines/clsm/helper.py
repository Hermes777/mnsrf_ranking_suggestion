###############################################################################
# Author: Wasi Ahmad
# Project: Convolutional Latent Semantic Model
# Date Created: 7/18/2017
#
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments.
###############################################################################

import os, glob, pickle, math, time, torch, numpy
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict
from torch.autograd import Variable


def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)


def load_model_states_from_checkpoint(model, filename, tag):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint[tag])


def load_model_states_without_dataparallel(model, filename, tag):
    """Load a previously saved model states."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    new_state_dict = OrderedDict()
    for k, v in checkpoint[tag].items():
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


def count_parameters(model):
    param_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_dict[name] = numpy.prod(param.size())
    return param_dict


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
    numpy.random.shuffle(data)
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    batched_data = [[data[bsz * i + j] for j in range(bsz)] for i in range(nbatch)]
    return batched_data


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


def sequence_to_tensors(sequence, dict_size):
    """Convert a sequence of words to a tensor of word indices."""
    seq_rep = numpy.zeros(shape=(len(sequence), dict_size))
    for i in range(len(sequence)):
        for j in range(len(sequence[i])):
            seq_rep[i][sequence[i][j]] += 1
    return seq_rep


def batch_to_tensor(batch, dict_size):
    query_tensor = numpy.ndarray(shape=(len(batch), len(batch[0].letter_trigrams), dict_size))
    query_clicks = numpy.ndarray(shape=(len(batch), len(batch[0].rel_docs), len(batch[0].rel_docs[0].letter_trigrams),
                                        dict_size))
    click_labels = numpy.ndarray(shape=(len(batch), len(batch[0].rel_docs)))

    for i in range(len(batch)):
        query_tensor[i] = sequence_to_tensors(batch[i].letter_trigrams, dict_size)
        # shuffle the list of relevant documents
        numpy.random.shuffle(batch[i].rel_docs)
        for j in range(len(batch[i].rel_docs)):
            query_clicks[i, j] = sequence_to_tensors(batch[i].rel_docs[j].letter_trigrams, dict_size)
            click_labels[i, j] = 1 if batch[i].rel_docs[j].is_clicked else 0

    query_tensor = torch.from_numpy(query_tensor).type(torch.FloatTensor)
    query_clicks = torch.from_numpy(query_clicks).type(torch.FloatTensor)
    click_labels = torch.from_numpy(click_labels).type(torch.FloatTensor)
    return Variable(query_tensor), Variable(query_clicks), Variable(click_labels)
