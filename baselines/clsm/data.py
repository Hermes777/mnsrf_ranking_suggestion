###############################################################################
# Author: Wasi Ahmad
# Project: Convolutional Latent Semantic Model
# Date Created: 7/18/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os, util

args = util.get_args()


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # Create and store special tokens
        self.special_token = '#'
        self.pad_token = '<p>'
        self.start_end_token = '<s>'
        self.pad_letter_trigram = '###'
        self.idx2word.append(self.pad_letter_trigram)
        self.word2idx[self.pad_letter_trigram] = len(self.idx2word) - 1

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
        self.body_terms = ['#' + item + '#' for item in ['<s>'] + content.split() + ['<s>']]
        if len(self.body_terms) < (args.max_doc_length + 2):
            self.body_terms += ['#<p>#'] * ((args.max_doc_length + 2) - len(self.body_terms))
        else:
            self.body_terms = self.body_terms[0:args.max_doc_length + 2]
        self.letter_trigrams = []
        self.is_clicked = False

    def update_doc_body(self, dictionary, is_test_instance):
        assert len(self.body_terms) == args.max_doc_length + 2
        for i in range(len(self.body_terms)):
            # create letter-trigrams
            word = self.body_terms[i]
            letter_trigrams_for_words = []
            for j in range(0, len(word) - 2):
                if is_test_instance:
                    if dictionary.contains(word[j:j + 3]):
                        letter_trigrams_for_words.append(dictionary.word2idx[word[j:j + 3]])
                else:
                    dictionary.add_letter_trigram(word[j:j + 3])
                    letter_trigrams_for_words.append(dictionary.word2idx[word[j:j + 3]])
            self.letter_trigrams.append(letter_trigrams_for_words)
        assert len(self.letter_trigrams) == args.max_doc_length + 2

    def set_clicked(self):
        self.is_clicked = True


class Query(object):
    def __init__(self, text):
        self.query_terms = ['#' + item + '#' for item in ['<s>'] + text.split() + ['<s>']]
        if len(self.query_terms) < (args.max_query_length + 2):
            self.query_terms += ['#<p>#'] * ((args.max_query_length + 2) - len(self.query_terms))
        else:
            self.query_terms = self.query_terms[0:args.max_query_length + 2]
        self.letter_trigrams = []
        self.rel_docs = []

    def update_query_text(self, dictionary, is_test_query):
        assert len(self.query_terms) == args.max_query_length + 2
        for i in range(len(self.query_terms)):
            # create letter-trigrams
            word = self.query_terms[i]
            letter_trigrams_for_words = []
            for j in range(0, len(word) - 2):
                if is_test_query:
                    if dictionary.contains(word[j:j + 3]):
                        letter_trigrams_for_words.append(dictionary.word2idx[word[j:j + 3]])
                else:
                    dictionary.add_letter_trigram(word[j:j + 3])
                    letter_trigrams_for_words.append(dictionary.word2idx[word[j:j + 3]])
            self.letter_trigrams.append(letter_trigrams_for_words)
        assert len(self.letter_trigrams) == args.max_query_length + 2

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
