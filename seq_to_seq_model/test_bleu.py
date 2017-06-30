###############################################################################
# Author: Wasi Ahmad
# Project: Context-aware Query Suggestion
# Date Created: 5/20/2017
#
# File Description: This script contains code for testing w.r.t. BLUE scores.
###############################################################################

import util, helper, data, multi_bleu
from torch.autograd import Variable
import torch

from seq2seq import Sequence2Sequence

args = util.get_args()


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

    sos_token_index = model.dictionary.word2idx[model.dictionary.start_token]
    eos_token_index = model.dictionary.word2idx[model.dictionary.end_token]

    # First input of the decoder is the sentence start token
    decoder_input = Variable(torch.LongTensor([sos_token_index]))
    decoded_words = []
    decoder_attentions = torch.zeros(model.config.max_length, batch_sentence.size(1))

    for di in range(model.config.max_length - 1):
        if model.config.cuda:
            decoder_input = decoder_input.cuda()
        decoder_output, decoder_hidden, decoder_attention = model.decoder(decoder_input, decoder_hidden,
                                                                          output)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == eos_token_index:
            decoded_words.append(model.dictionary.end_token)
            break
        else:
            decoded_words.append(model.dictionary.idx2word[ni])
        decoder_input = Variable(torch.LongTensor([ni]))

    return " ".join(decoded_words[:-1])


def evaluate(model, dictionary, source_words):
    """Generates word sequence and their attentions"""
    input_tensor = helper.sentence_to_tensor(source_words, len(source_words), dictionary)
    input_sentence = Variable(input_tensor.view(1, - 1), volatile=True)
    if args.cuda:
        input_sentence = input_sentence.cuda()
    output = test(model, input_sentence)
    return output


if __name__ == "__main__":
    # Load the saved pre-trained model
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings('../data/glove/', 'glove.840B.300d.q2q.txt')
    model = Sequence2Sequence(dictionary, embeddings_index, args)
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_without_dataparallel(model, 'model_loss_1.402306_epoch_12_model.pt')
    print('Model, embedding index and dictionary loaded.')

    # words = evaluate(model, dictionary, "university of virginia")
    # print(" ".join(words[0]))

    test_corpus = data.Corpus(args.data, 'session_test.txt', dictionary, args.max_length, is_test_corpus=True)
    print('Test set size = ', len(test_corpus.data))

    targets = []
    candidates = []
    for item in test_corpus.data:
        targets.append(evaluate(model, dictionary, item.sentence1))
        candidates.append(" ".join(item.sentence2))

    print("target size = ", len(targets))
    print("candidate size = ", len(candidates))
    multi_bleu.print_multi_bleu(targets, candidates)
