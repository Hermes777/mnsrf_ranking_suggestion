###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os


class Dictionary(object):
    def __init__(self, order_n_gram):
        self.order_n_gram = order_n_gram
        self.word2idx = {}
        self.idx2word = []

    def load_dictionary(self, path, filename, top_n_words):
        with open(os.path.join(path, filename), 'r') as f:
            counter = 0
            for line in f:
                token = line.strip().split(',')[0]
                self.idx2word.append(token)
                self.word2idx[token] = len(self.idx2word) - 1
                counter += 1
                if counter == top_n_words:
                    break

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Document(object):
    def __init__(self):
        self.letter_n_grams = []
        self.is_clicked = False

    def add_tokens(self, tokens, dictionary):
        for word_letter_n_grams in tokens:
            n_grams = []
            for letter_n_grams in word_letter_n_grams:
                if dictionary.contains(letter_n_grams):
                    n_grams.append(dictionary.word2idx[letter_n_grams])
            self.letter_n_grams.append(n_grams)

    def set_clicked(self):
        self.is_clicked = True


class Query(object):
    def __init__(self):
        self.letter_n_grams = []
        self.rel_docs = []

    def add_tokens(self, tokens, dictionary):
        for word_letter_n_grams in tokens:
            n_grams = []
            for letter_n_grams in word_letter_n_grams:
                if dictionary.contains(letter_n_grams):
                    n_grams.append(dictionary.word2idx[letter_n_grams])
            self.letter_n_grams.append(n_grams)

    def add_rel_doc(self, doc):
        if isinstance(doc, Document):
            self.rel_docs.append(doc)
        else:
            print('Unknown document type!')


class Corpus(object):
    def __init__(self, path, filename, dictionary):
        self.data = self.parse(os.path.join(path, filename), dictionary)

    def parse(self, path, dictionary):
        """Parses the content of a file."""
        assert os.path.exists(path)

        queries = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.split('\t')
                    query_letter_n_grams = self.get_letter_n_grams(tokens[0], dictionary.order_n_gram)
                    query = Query()
                    query.add_tokens(query_letter_n_grams, dictionary)
                    for i in range(2, len(tokens), 3):
                        doc_letter_n_grams = self.get_letter_n_grams(tokens[i], dictionary.order_n_gram)
                        doc = Document()
                        doc.add_tokens(doc_letter_n_grams, dictionary)
                        if int(tokens[i + 1]) == 1:
                            doc.set_clicked()
                        query.add_rel_doc(doc)
                    queries.append(query)

        return queries

    def get_letter_n_grams(self, text, n):
        """Returns 2d list, each element contains letter_n_grams for a word."""
        letter_n_grams = []
        tokens = text.split()
        for i in range(len(tokens)):
            if tokens[i] != '<unk>':
                token_letter_n_grams = []
                for j in range(1, n + 1):
                    if j > len(tokens[i]):
                        break
                    else:
                        # create letter_n_grams where n = j
                        token_letter_n_grams.extend(self.find_letter_ngrams(tokens[i], j))
                if token_letter_n_grams:
                    letter_n_grams.append(token_letter_n_grams)

        return letter_n_grams

    @staticmethod
    def find_letter_ngrams(word, n):
        return [''.join(list(a)) for a in zip(*[word[i:] for i in range(n)])]
