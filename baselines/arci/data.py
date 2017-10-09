###############################################################################
# Author: Wasi Ahmad
# Project: ARC-I: Convolutional Matching Model
# Date Created: 7/28/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # Create and store special tokens
        self.pad_token = '<pad>'
        self.start_token = '<sos>'
        self.end_token = '<eos>'
        self.unknown_token = '<unk>'
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1
        self.idx2word.append(self.start_token)
        self.word2idx[self.start_token] = len(self.idx2word) - 1
        self.idx2word.append(self.end_token)
        self.word2idx[self.end_token] = len(self.idx2word) - 1
        self.idx2word.append(self.unknown_token)
        self.word2idx[self.unknown_token] = len(self.idx2word) - 1

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
    def __init__(self, content):
        self.body = content.split()
        self.is_clicked = False

    def update_doc_body(self, dictionary, max_length, is_test_instance):
        if len(self.body) > max_length:
            self.body = self.body[:max_length]
        if is_test_instance:
            for i in range(len(self.body)):
                if not dictionary.contains(self.body[i]):
                    self.body[i] = dictionary.unknown_token
        else:
            for word in self.body:
                dictionary.add_word(word)

    def set_clicked(self):
        self.is_clicked = True

    def is_clicked(self):
        return self.is_clicked

    def get_doc_content(self):
        return self.body


class Query(object):
    def __init__(self, text):
        self.query_terms = text.split()
        self.rel_docs = []

    def update_query_text(self, dictionary, max_length, is_test_query):
        if len(self.query_terms) > max_length:
            self.query_terms = self.query_terms[:max_length]
        if is_test_query:
            for i in range(len(self.query_terms)):
                if not dictionary.contains(self.query_terms[i]):
                    self.query_terms[i] = dictionary.unknown_token
        else:
            for term in self.query_terms:
                dictionary.add_word(term)

    def add_rel_doc(self, doc):
        if isinstance(doc, Document):
            self.rel_docs.append(doc)
        else:
            print('Unknown document type!')

    def get_rel_docs(self):
        return self.rel_docs


class Corpus(object):
    def __init__(self, path, filename, dictionary, max_query_length, max_doc_length, is_test_corpus=False):
        self.data = self.parse(os.path.join(path, filename), dictionary, max_query_length, max_doc_length,
                               is_test_corpus)

    def parse(self, path, dictionary, max_query_length, max_doc_length, is_test_corpus):
        """Parses the content of a file."""
        assert os.path.exists(path)

        queries = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens = line.split('\t')
                    query = Query(tokens[0])
                    query.update_query_text(dictionary, max_query_length, is_test_corpus)
                    indices = list(range(2, len(tokens), 3))
                    for i in indices:
                        doc = Document(tokens[i])
                        doc.update_doc_body(dictionary, max_doc_length, is_test_corpus)
                        if int(tokens[i + 1]) == 1:
                            doc.set_clicked()
                        query.add_rel_doc(doc)
                    queries.append(query)

        return queries
