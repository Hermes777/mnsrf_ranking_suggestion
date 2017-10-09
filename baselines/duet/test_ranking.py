###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import torch, helper, util, data, os
from duet import DUET
from ranking_eval_functions import mean_average_precision, NDCG

args = util.get_args()


def compute_ranking_performance(model, test_batch, test_clicks, test_labels):
    local_score = model.local_model(test_batch, test_clicks)
    distributed_score = model.distributed_model(test_batch, test_clicks)
    total_score = local_score + distributed_score

    MAP = mean_average_precision(total_score, test_labels)
    NDCG_at_1 = NDCG(total_score, test_labels, 1)
    NDCG_at_3 = NDCG(total_score, test_labels, 3)
    NDCG_at_10 = NDCG(total_score, test_labels, 5)

    return MAP, NDCG_at_1, NDCG_at_3, NDCG_at_10


def test_ranking(model, test_batches):
    num_batches = len(test_batches)
    map, ndcg_1, ndcg_3, ndcg_10 = 0, 0, 0, 0
    for batch_no in range(1, num_batches + 1):
        test_queries, test_docs, test_labels = helper.batch_to_tensor(test_batches[batch_no - 1], model.dictionary,
                                                                      model.config.max_query_length,
                                                                      model.config.max_doc_length)
        if model.config.cuda:
            test_queries = test_queries.cuda()
            test_docs = test_docs.cuda()
            test_labels = test_labels.cuda()

        ret_val = compute_ranking_performance(model, test_queries, test_docs, test_labels)
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
    dictionary = data.Dictionary(5)
    dictionary.load_dictionary(args.save_path, 'vocab.csv', 5000)
    model = DUET(dictionary, args)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(cuda_visible_devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=cuda_visible_devices)
    if args.cuda:
        model = model.cuda()

    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict')
    print('Model and dictionary loaded.')
    model.eval()

    test_corpus = data.Corpus(args.data, 'session_test.txt', dictionary)
    print('Test set size = ', len(test_corpus.data))

    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    print('Number of test batches = ', len(test_batches))

    test_ranking(model, test_batches)
