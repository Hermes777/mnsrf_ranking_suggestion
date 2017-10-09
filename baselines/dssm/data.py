###############################################################################
# Author: Wasi Ahmad
# Project: Deep Semantic Similarity Model
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
        self.special_token = '#'

    def add_letter_trigram(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Document(object):
    def __init__(self, content):
        self.body = content.split()
        self.letter_trigrams = []
        self.is_clicked = False

    def update_doc_body(self, dictionary, is_test_instance):
        if is_test_instance:
            for i in range(len(self.body)):
                term = dictionary.special_token + self.body[i] + dictionary.special_token
                # create letter-trigrams
                for j in range(0, len(term) - 2):
                    if dictionary.contains(term[j:j + 3]):
                        self.letter_trigrams.append(term[j:j + 3])
        else:
            for i in range(len(self.body)):
                term = dictionary.special_token + self.body[i] + dictionary.special_token
                # create letter-trigrams
                for j in range(0, len(term) - 2):
                    self.letter_trigrams.append(term[j:j + 3])
                    dictionary.add_letter_trigram(term[j:j + 3])

    def set_clicked(self):
        self.is_clicked = True


class Query(object):
    def __init__(self, text):
        self.query_terms = text.split()
        self.letter_trigrams = []
        self.rel_docs = []

    def update_query_text(self, dictionary, is_test_query):
        if is_test_query:
            for i in range(len(self.query_terms)):
                term = dictionary.special_token + self.query_terms[i] + dictionary.special_token
                # create letter-trigrams
                for j in range(0, len(term) - 2):
                    if dictionary.contains(term[j:j + 3]):
                        self.letter_trigrams.append(term[j:j + 3])
        else:
            for i in range(len(self.query_terms)):
                term = dictionary.special_token + self.query_terms[i] + dictionary.special_token
                # create letter-trigrams
                for j in range(0, len(term) - 2):
                    self.letter_trigrams.append(term[j:j + 3])
                    dictionary.add_letter_trigram(term[j:j + 3])

    def add_rel_doc(self, doc):
        if isinstance(doc, Document):
            self.rel_docs.append(doc)
        else:
            print('Unknown document type!')


class Corpus(object):
    def __init__(self, path, filename, dictionary, is_test_corpus=False):
        self.data = self.parse(os.path.join(path, filename), dictionary, is_test_corpus)

    def parse(self, path, dictionary, is_test_corpus):
        """Parses the content of a file."""
        assert os.path.exists(path)

        queries = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.split('\t')
                    query = Query(tokens[0])
                    query.update_query_text(dictionary, is_test_corpus)
                    for i in range(2, len(tokens), 3):
                        doc = Document(tokens[i])
                        doc.update_doc_body(dictionary, is_test_corpus)
                        if int(tokens[i + 1]) == 1:
                            doc.set_clicked()
                        query.add_rel_doc(doc)
                    queries.append(query)

        return queries
