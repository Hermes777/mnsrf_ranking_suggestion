###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 6/20/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # Create and store three special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<SOS>'
        self.end_token = '<EOS>'
        self.unknown_token = '<UNKNOWN>'
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


class Session(object):
    def __init__(self):
        self.queries = []

    def form_session(self, queries, dictionary, max_length, is_test_instance=False):
        for query in queries:
            # terms = helper.tokenize_and_normalize(query)
            terms = query.split() + [dictionary.end_token]
            if len(terms) > (max_length + 1):
                continue
            self.queries.append(terms)

        if len(self.queries) > 2:
            for query in self.queries:
                if is_test_instance:
                    for i in range(len(query)):
                        if not dictionary.contains(query[i]):
                            query[i] = dictionary.unknown_token
                else:
                    for term in query:
                        dictionary.add_word(term)
            return len(self.queries)
        else:
            return -1

    def __len__(self):
        return len(self.queries)


class Corpus(object):
    def __init__(self, path, filename, dictionary, max_length, is_test_corpus=False):
        self.max_session_length = 0
        self.data = self.parse(os.path.join(path, filename), dictionary, max_length, is_test_corpus)

    def parse(self, path, dictionary, max_length, is_test_corpus):
        """Parses the content of a file."""
        assert os.path.exists(path)

        samples = {}
        with open(path, 'r') as f:
            for line in f:
                queries = line.strip().split(':::')
                session = Session()
                session_length = session.form_session(queries, dictionary, max_length, is_test_corpus)
                if session_length != -1:
                    if session_length in samples:
                        samples[session_length].append(session)
                    else:
                        samples[session_length] = [session]
                    if session_length > self.max_session_length:
                        self.max_session_length = session_length

        return samples

    def __len__(self):
        length = 0
        for key, value in self.data.items():
            length += len(value)
        return length
