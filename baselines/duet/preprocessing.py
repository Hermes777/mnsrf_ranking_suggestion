###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script handles preprocessing and creation of vocabulary.
###############################################################################

import os, operator


class Vocabulary(object):
    def __init__(self, order_n_gram=5):
        self.order_n_gram = order_n_gram
        self.word2freq = {}

    def form_vocabulary(self, path, filename):
        """Creates the vocabulary."""
        assert os.path.exists(os.path.join(path, filename))
        with open(os.path.join(path, filename), 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.split('\t')
                    query_text = tokens[0]
                    query_letter_n_grams = self.get_letter_n_grams(query_text, self.order_n_gram)
                    self.add_letter_n_grams(query_letter_n_grams)
                    for i in range(2, len(tokens), 3):
                        doc_body = tokens[i]
                        doc_letter_n_grams = self.get_letter_n_grams(doc_body, self.order_n_gram)
                        self.add_letter_n_grams(doc_letter_n_grams)

    def get_letter_n_grams(self, text, n):
        letter_n_grams = []
        tokens = text.split()
        for i in range(len(tokens)):
            if tokens[i] != '<unk>':
                for j in range(1, n + 1):
                    if j > len(tokens[i]):
                        break
                    else:
                        # create letter_n_grams where n = j
                        letter_n_grams.extend(self.find_letter_ngrams(tokens[i], j))

        return letter_n_grams

    @staticmethod
    def find_letter_ngrams(word, n):
        return [''.join(list(a)) for a in zip(*[word[i:] for i in range(n)])]

    def add_letter_n_grams(self, n_grams):
        for token in n_grams:
            if token not in self.word2freq:
                self.word2freq[token] = 1
            else:
                self.word2freq[token] += 1

    def contains(self, word):
        return True if word in self.word2freq else False

    def save_vocabulary(self, path, filename):
        assert os.path.exists(path)
        sorted_x = sorted(self.word2freq.items(), key=operator.itemgetter(1), reverse=True)
        with open(os.path.join(path, filename), 'w') as f:
            for word, freq in sorted_x:
                f.write(word + ',' + str(freq) + '\n')

    def __len__(self):
        return len(self.word2freq)
