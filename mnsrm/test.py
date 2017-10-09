###############################################################################
# Author: Wasi Ahmad
# Project: Neural Session Relevance Model
# Date Created: 7/15/2017
#
# File Description: This script visualizes query and document representations.
###############################################################################

import torch, helper, util, numpy, os
from nsrm import Sequence2Sequence
from torch.autograd import Variable

args = util.get_args()


def visualize(model, sent, is_query=True):
    tokens = sent.split()
    batch = Variable(helper.sequence_to_tensors(tokens, len(tokens), model.dictionary), volatile=True)
    if model.config.cuda:
        batch = batch.cuda()

    embedded_sents = model.embedding(batch.unsqueeze(0))
    if is_query:
        if model.config.model == 'LSTM':
            encoder_hidden, encoder_cell = model.query_encoder.init_weights(embedded_sents.size(0))
            output, hidden = model.query_encoder(embedded_sents, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = model.query_encoder.init_weights(embedded_sents.size(0))
            output, hidden = model.query_encoder(embedded_sents, encoder_hidden)
    else:
        if model.config.model == 'LSTM':
            encoder_hidden, encoder_cell = model.document_encoder.init_weights(embedded_sents.size(0))
            output, hidden = model.document_encoder(embedded_sents, (encoder_hidden, encoder_cell))
        else:
            encoder_hidden = model.document_encoder.init_weights(embedded_sents.size(0))
            output, hidden = model.document_encoder(embedded_sents, encoder_hidden)

    output, idxs = torch.max(output, 1)
    idxs = idxs.data.cpu().numpy()
    argmaxs = [numpy.sum((idxs == k)) for k in range(len(tokens))]

    # visualize model
    import matplotlib.pyplot as plt
    x = range(len(tokens))
    y = [100.0 * n / numpy.sum(argmaxs) for n in argmaxs]
    fig = plt.figure()
    plt.xticks(x, tokens, rotation=45)
    plt.bar(x, y)
    plt.ylabel('%')
    plt.title('Visualisation of words importance')
    plt.show()
    fig.savefig('./figure.png')
    plt.close(fig)  # close the figure

    return output, idxs


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, 'glove.840B.300d.query.clicks.txt')
    model = Sequence2Sequence(dictionary, embeddings_index, args)
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict')
    print('Model, embedding index and dictionary loaded.')
    model.eval()
    visualize(model, 'starwood hotels resorts book hotels online')
