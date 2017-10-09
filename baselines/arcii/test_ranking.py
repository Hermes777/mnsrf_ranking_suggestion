###############################################################################
# Author: Wasi Ahmad
# Project: Match Tensor: a Deep Relevance Model for Search
# Date Created: 7/28/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import torch, helper, util, data, os
from cnn_match_model import CNN_ARC_II
from ranking_eval_functions import mean_average_precision, NDCG

args = util.get_args()


def test_ranking(model, test_batches):
    num_batches = len(test_batches)
    map, ndcg_1, ndcg_3, ndcg_10 = 0, 0, 0, 0
    for batch_no in range(1, num_batches + 1):
        test_queries, test_clicks, click_labels = helper.batch_to_tensor(test_batches[batch_no - 1], model.dictionary,
                                                                         model.config)
        if model.config.cuda:
            test_queries = test_queries.cuda()
            test_clicks = test_clicks.cuda()
            click_labels = click_labels.cuda()
        score = model(test_queries, test_clicks)
        map += mean_average_precision(score, click_labels)
        ndcg_1 += NDCG(score, click_labels, 1)
        ndcg_3 += NDCG(score, click_labels, 3)
        ndcg_10 += NDCG(score, click_labels, 5)

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
    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, 'glove.840B.300d.query.clicks.txt')
    model = CNN_ARC_II(dictionary, embeddings_index, args)

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
    print('Test set size = ', len(test_corpus.data))

    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    print('Number of test batches = ', len(test_batches))

    test_ranking(model, test_batches)
