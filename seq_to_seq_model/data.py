###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 5/20/2017
#
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import os, helper


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


class Instance(object):
    def __init__(self):
        self.sentence1 = []
        self.sentence2 = []

    def add_sentence(self, sentence, sentence_no, dictionary, max_length, is_test_instance=False):
        if sentence_no == 1:
            # words = helper.tokenize_and_normalize(sentence)
            words = sentence.split() + [dictionary.end_token]
            if len(words) > (max_length + 1):
                return -1
        else:
            # words = [dictionary.start_token] + helper.tokenize_and_normalize(sentence) + [dictionary.end_token]
            words = [dictionary.start_token] + sentence.split() + [dictionary.end_token]
            if len(words) > (max_length + 2):
                return -1

        if is_test_instance:
            for i in range(len(words)):
                if not dictionary.contains(words[i]):
                    words[i] = dictionary.unknown_token
        else:
            for word in words:
                dictionary.add_word(word)

        if sentence_no == 1:
            self.sentence1 = words
        else:
            self.sentence2 = words

        return len(words)


class Corpus(object):
    def __init__(self, path, filename, dictionary, max_length, is_test_corpus=False):
        self.max_sent_length = 0
        self.data = self.parse(os.path.join(path, filename), dictionary, max_length, is_test_corpus)

    def parse(self, path, dictionary, max_length, is_test_corpus):
        """Parses the content of a file."""
        assert os.path.exists(path)

        samples = []
        # counter = 0
        with open(path, 'r') as f:
            for line in f:
                queries = line.strip().split(':::')
                for i in range(1, len(queries)):
                    instance = Instance()
                    length = instance.add_sentence(queries[i - 1], 1, dictionary, max_length, is_test_corpus)
                    if length == -1:
                        continue
                    elif length > self.max_sent_length:
                        self.max_sent_length = length

                    length = instance.add_sentence(queries[i], 2, dictionary, max_length, is_test_corpus)
                    if length == -1:
                        continue
                    elif length > self.max_sent_length:
                        self.max_sent_length = length

                    samples.append(instance)
                    # counter += 1

                # if counter >= 512 * 1:
                # break

        return samples
