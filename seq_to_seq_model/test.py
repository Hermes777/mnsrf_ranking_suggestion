###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 5/20/2017
#
# File Description: This script contains code for testing.
###############################################################################

import util, helper, os
from torch.autograd import Variable
import torch, queue, numpy, operator

from seq2seq import Sequence2Sequence

args = util.get_args()


class search_node:
    def __init__(self, id):
        self.params = None
        self.widx = id
        self.antecedents = []
        self.probs = []

    def set_params(self, parameters):
        self.params = parameters

    def set_score(self, score):
        self.score = score
        self.probs.append(score)

    def get_params(self):
        return self.params

    def get_word_id(self):
        return self.widx

    def add_antecedents(self, antecs):
        self.antecedents.extend(antecs)

    def get_antecedents(self):
        return self.antecedents

    def get_score(self):
        return self.score


def test(model, batch_sentence):
    if model.config.model == 'LSTM':
        encoder_hidden, encoder_cell = model.encoder.init_weights(batch_sentence.size(0))
        output, hidden = model.encoder(batch_sentence, (encoder_hidden, encoder_cell))
    else:
        encoder_hidden = model.encoder.init_weights(batch_sentence.size(0))
        output, hidden = model.encoder(batch_sentence, encoder_hidden)

    if model.config.bidirection:
        hidden = torch.mean(hidden[0], 0), torch.mean(hidden[1], 0)
        output = torch.div(torch.add(output[:, :, 0:model.config.nhid],
                                     output[:, :, model.config.nhid:2 * model.config.nhid]), 2)

    # Initialize hidden states of decoder with the last hidden states of the encoder
    decoder_hidden = hidden
    # decoder_context = Variable(torch.zeros(batch_sentence.size(0), 1, model.config.nhid))
    # if model.config.cuda:
    #     decoder_context = decoder_context.cuda()

    sos_token_index = model.dictionary.word2idx[model.dictionary.start_token]
    eos_token_index = model.dictionary.word2idx[model.dictionary.end_token]

    # First input of the decoder is the sentence start token
    decoder_input = Variable(torch.LongTensor([sos_token_index]))
    decoded_words = []
    # decoder_attentions = torch.zeros(model.config.max_length, batch_sentence.size(1))

    for di in range(model.config.max_length - 1):
        if model.config.cuda:
            decoder_input = decoder_input.cuda()
        # decoder_output, decoder_hidden, decoder_context, decoder_attention = model.decoder(decoder_input,
        #                                                                                    decoder_hidden,
        #                                                                                    decoder_context,
        #                                                                                    output)
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        # decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == eos_token_index:
            decoded_words.append(model.dictionary.end_token)
            break
        else:
            decoded_words.append(model.dictionary.idx2word[ni])
        decoder_input = Variable(torch.LongTensor([ni]))

    # return decoded_words, decoder_attentions[:di + 1]
    return decoded_words


def evaluate(model, dictionary, sentence):
    """Generates word sequence and their attentions"""
    source_words = sentence.split()
    input_tensor = helper.sentence_to_tensor(source_words, len(source_words), dictionary)
    input_sentence = Variable(input_tensor.view(1, - 1), volatile=True)
    if args.cuda:
        input_sentence = input_sentence.cuda()
    # output_words, attentions = test(model, input_sentence)
    output_words = test(model, input_sentence)
    # return output_words, attentions
    return output_words


def evaluate_and_show_attention(model, dictionary, input_sentence, filename):
    """Evaluates and shows attention given the input sentence"""
    output_words, attentions = evaluate(model, dictionary, input_sentence)
    print('input = ', input_sentence)
    print('output = ', ' '.join(output_words))
    helper.save_attention_plot(input_sentence, output_words, attentions, args.save_path + filename)


def do_beam_search(model, beam_size, batch_sentence):
    if model.config.model == 'LSTM':
        encoder_hidden, encoder_cell = model.encoder.init_weights(batch_sentence.size(0))
        output, hidden = model.encoder(batch_sentence, (encoder_hidden, encoder_cell))
    else:
        encoder_hidden = model.encoder.init_weights(batch_sentence.size(0))
        output, hidden = model.encoder(batch_sentence, encoder_hidden)

    if model.config.bidirection:
        hidden = torch.mean(hidden[0], 0), torch.mean(hidden[1], 0)
        output = torch.div(torch.add(output[:, :, 0:model.config.nhid],
                                     output[:, :, model.config.nhid:2 * model.config.nhid]), 2)

    sos_token_index = model.dictionary.word2idx[model.dictionary.start_token]
    eos_token_index = model.dictionary.word2idx[model.dictionary.end_token]

    # First input of the decoder is the sentence start token
    node = search_node(sos_token_index)
    node.set_params(hidden)
    node.set_score(1)
    qu = queue.Queue()
    qu.put(node)

    decoded_words = []
    while not qu.empty():
        input_node = qu.get()
        if input_node.get_word_id() == eos_token_index:
            decoded_words.append(input_node)
            continue

        decoder_input = Variable(torch.LongTensor([input_node.get_word_id()]))
        if model.config.cuda:
            decoder_input = decoder_input.cuda()

        decoder_hidden = input_node.get_params()
        decoder_output, decoder_hidden, decoder_attention = model.decoder(decoder_input, decoder_hidden, output)

        topv, topi = decoder_output.data.topk(beam_size)

        candidates = {}
        for i in range(topi.size(1)):
            node = search_node(topi[0][i])
            node.set_params(decoder_hidden)
            node.set_score(input_node.get_score() * numpy.exp(topv[0][i]))
            if input_node.get_word_id() != sos_token_index:
                node.add_antecedents(input_node.get_antecedents())
            node.add_antecedents([input_node.get_word_id()])
            candidates[node] = numpy.log(node.get_score()) / len(node.get_antecedents())

        if candidates:
            while not qu.empty():
                node = qu.get()
                candidates[node] = numpy.log(node.get_score()) / len(node.get_antecedents())

        sorted_dictionary = sorted(candidates.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(beam_size - len(decoded_words)):
            qu.put(sorted_dictionary[i][0])
        candidates.clear()

    return decoded_words


def run_beam_search(sentence1):
    source_words = sentence1.split()
    input_tensor = helper.sentence_to_tensor(source_words, len(source_words), dictionary)
    input_sentence = Variable(input_tensor.view(1, - 1), volatile=True)
    if args.cuda:
        input_sentence = input_sentence.cuda()
    outputs = do_beam_search(model, 5, input_sentence)
    return outputs


if __name__ == "__main__":
    # Load the saved pre-trained model
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, 'glove.840B.300d.q2q.txt')
    model = Sequence2Sequence(dictionary, embeddings_index, args)
    if args.cuda:
        model = model.cuda()
    # helper.load_model_states(model, os.path.join(args.save_path, 'model_loss_15.465827_epoch_48_model.pt'))
    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict')
    print('Model, embedding index and dictionary loaded.')

    # words = [dictionary.start_token] + helper.tokenize_and_normalize('football games in march') + [
    #     dictionary.end_token]
    # evaluate_and_show_attention(model, dictionary, words, 'attention3.png')

    '''
    seqs = run_beam_search('university of virginia')
    all_queries = {}
    for item in seqs:
        query = ''
        for ant in item.get_antecedents()[1:]:
            query = query + model.dictionary.idx2word[ant] + ' '
        query = query.strip()
        all_queries[query] = item.get_score()

    sorted_queries = sorted(all_queries.items(), key=operator.itemgetter(1), reverse=True)
    for item in sorted_queries:
        print(item)
    '''

    words = evaluate(model, dictionary, "walmart store")
    print(" ".join(words))
