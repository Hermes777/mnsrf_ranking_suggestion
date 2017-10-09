###############################################################################
# Author: Wasi Ahmad
# Project: Dual Embedding Space Model
# Date Created: 7/18/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Document(object):
    def __init__(self, content, dictionary):
        self.body = content.split()
        for i in range(len(self.body)):
            dictionary.add_word(self.body[i])
        self.is_clicked = False

    def set_clicked(self):
        self.is_clicked = True


class Query(object):
    def __init__(self, text, dictionary):
        self.query_terms = text.split()
        for i in range(len(self.query_terms)):
            dictionary.add_word(self.query_terms[i])
        self.rel_docs = []

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
                    query = Query(tokens[0], dictionary)
                    for i in range(2, len(tokens), 3):
                        doc = Document(tokens[i], dictionary)
                        if int(tokens[i + 1]) == 1:
                            doc.set_clicked()
                        query.add_rel_doc(doc)
                    queries.append(query)

        return queries
