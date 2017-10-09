###############################################################################
# Author: Wasi Ahmad
# Project: Deep Semantic Similarity Model
# Date Created: 7/18/2017
#
# File Description: This script provides general purpose utility functions that
# may come in handy at any point in the experiments.
###############################################################################

import os
import numpy as np
from numpy.linalg import norm


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
