###############################################################################
# Author: Wasi Ahmad
# Project: Neural Session Relevance Model
# Date Created: 7/15/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import torch, helper, util, data, os
from torch.autograd import Variable
from nsrm import NSRM
from ranking_eval_functions import mean_average_precision, NDCG

args = util.get_args()


def test_loss(model, test_batches):
    num_batches = len(test_batches)
    test_loss = 0
    for batch_no in range(1, num_batches + 1):
        test_sessions, length, test_clicks, click_labels = helper.session_to_tensor(test_batches[batch_no - 1],
                                                                                    model.dictionary)
        if model.config.cuda:
            test_sessions = test_sessions.cuda()
            test_clicks = test_clicks.cuda()
            length = length.cuda()
            click_labels = click_labels.cuda()

        loss = model(test_sessions, length, test_clicks, click_labels)
        if loss.size(0) > 1:
            loss = torch.mean(loss)
        test_loss += loss.data[0]

    print('test loss - ', (test_loss / num_batches))


def compute_ranking_performance(model, test_batch, test_clicks, test_labels):
    # query level encoding
    embedded_queries = model.embedding(test_batch.view(-1, test_batch.size(-1)))
    if model.config.model == 'LSTM':
        encoder_hidden, encoder_cell = model.query_encoder.init_weights(embedded_queries.size(0))
        output, hidden = model.query_encoder(embedded_queries, (encoder_hidden, encoder_cell))
    else:
        encoder_hidden = model.query_encoder.init_weights(embedded_queries.size(0))
        output, hidden = model.query_encoder(embedded_queries, encoder_hidden)

    # encoded_queries = torch.max(output, 1)[0].squeeze(1)
    encoded_queries = output[:, -1, :].contiguous()
    encoded_queries = encoded_queries.view(*test_batch.size()[:-1], -1)

    # document level encoding
    embedded_clicks = model.embedding(test_clicks.view(-1, test_clicks.size(-1)))
    if model.config.model == 'LSTM':
        encoder_hidden, encoder_cell = model.document_encoder.init_weights(embedded_clicks.size(0))
        output, hidden = model.document_encoder(embedded_clicks, (encoder_hidden, encoder_cell))
    else:
        encoder_hidden = model.document_encoder.init_weights(embedded_clicks.size(0))
        output, hidden = model.document_encoder(embedded_clicks, encoder_hidden)

    # encoded_clicks = torch.max(output, 1)[0].squeeze(1)
    encoded_clicks = output[:, -1, :].contiguous()
    encoded_clicks = encoded_clicks.view(*test_clicks.size()[:-1], -1)

    # session level encoding
    sess_hidden = model.session_encoder.init_weights(encoded_queries.size(0))
    sess_output = Variable(torch.zeros(model.config.batch_size, 1, model.config.nhid_session))
    if model.config.cuda:
        sess_output = sess_output.cuda()
    ranking_performance, NDCG_at_1, NDCG_at_3, NDCG_at_10 = 0, 0, 0, 0
    for idx in range(encoded_queries.size(1)):
        combined_rep = torch.cat((sess_output.squeeze(), encoded_queries[:, idx, :]), 1)
        combined_rep = model.projection(combined_rep)
        combined_rep = combined_rep.unsqueeze(1).expand(*encoded_clicks[:, idx, :, :].size())
        click_score = torch.sum(torch.mul(combined_rep, encoded_clicks[:, idx, :, :]), 2).squeeze(2)
        ranking_performance += mean_average_precision(click_score, test_labels[:, idx, :])
        NDCG_at_1 += NDCG(click_score, test_labels[:, idx, :], 1)
        NDCG_at_3 += NDCG(click_score, test_labels[:, idx, :], 3)
        NDCG_at_10 += NDCG(click_score, test_labels[:, idx, :], 5)
        # update session state using query representations
        sess_output, sess_hidden = model.session_encoder(encoded_queries[:, idx, :].unsqueeze(1), sess_hidden)

    ranking_performance = ranking_performance / encoded_queries.size(1)
    NDCG_at_1 = NDCG_at_1 / encoded_queries.size(1)
    NDCG_at_3 = NDCG_at_3 / encoded_queries.size(1)
    NDCG_at_10 = NDCG_at_10 / encoded_queries.size(1)

    return ranking_performance, NDCG_at_1, NDCG_at_3, NDCG_at_10


def test_ranking(model, test_batches):
    num_batches = len(test_batches)
    map, ndcg_1, ndcg_3, ndcg_10 = 0, 0, 0, 0
    for batch_no in range(1, num_batches + 1):
        test_sessions, length, test_clicks, click_labels = helper.session_to_tensor(test_batches[batch_no - 1],
                                                                                    model.dictionary)
        if model.config.cuda:
            test_sessions = test_sessions.cuda()
            test_clicks = test_clicks.cuda()
            click_labels = click_labels.cuda()

        ret_val = compute_ranking_performance(model, test_sessions, test_clicks, click_labels)
        map += ret_val[0]
        ndcg_1 += ret_val[1]
        ndcg_3 += ret_val[2]
        ndcg_10 += ret_val[3]

    map = map / num_batches
    ndcg_1 = ndcg_1 / num_batches
    ndcg_3 = ndcg_3 / num_batches
    ndcg_10 = ndcg_10 / num_batches

    print('MAP - ', map)
    print('NDCG@1 - ', ndcg_1)
    print('NDCG@3 - ', ndcg_3)
    print('NDCG@10 - ', ndcg_10)


if __name__ == "__main__":
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, 'glove.6B.200d.query.clicks.txt')
    model = NSRM(dictionary, embeddings_index, args)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(cuda_visible_devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=cuda_visible_devices)
    if args.cuda:
        model = model.cuda()

    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict')
    print('Model, embedding index and dictionary loaded.')
    model.eval()

    test_corpus = data.Corpus(args.data + 'session_with_clicks_v5/', 'session_test.txt', dictionary,
                              args.max_query_length,
                              args.max_doc_length, is_test_corpus=True)
    print('Test set size = ', len(test_corpus))

    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    print('Number of test batches = ', len(test_batches))

    # test_loss(model, test_batches)
    test_ranking(model, test_batches)
