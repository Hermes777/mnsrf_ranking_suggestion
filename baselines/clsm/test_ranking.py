###############################################################################
# Author: Wasi Ahmad
# Project: Convolutional Latent Semantic Model
# Date Created: 7/18/2017
#
# File Description: This script evaluates test ranking performance.
###############################################################################

import torch, helper, util, data, os
from clsm import CLSM
from ranking_eval_functions import mean_average_precision, NDCG

args = util.get_args()


def compute_ranking_performance(model, test_batch, test_clicks, test_labels):
    # query encoding
    query_rep = model.convolution(test_batch.transpose(1, 2)).transpose(1, 2)
    latent_query_rep = torch.max(query_rep, 1)[0].squeeze()
    # document encoding
    doc_rep = model.convolution(test_clicks.view(-1, *test_clicks.size()[2:]).transpose(1, 2)).transpose(1, 2)
    latent_doc_rep = torch.max(doc_rep, 1)[0].squeeze().view(*test_clicks.size()[:2], -1)
    # compute loss
    latent_query_rep = latent_query_rep.unsqueeze(1).expand(*latent_doc_rep.size())
    cs_similarity = model.cosine_similarity(latent_query_rep, latent_doc_rep, 2)
    ranking_performance = mean_average_precision(cs_similarity, test_labels)
    NDCG_at_1 = NDCG(cs_similarity, test_labels, 1)
    NDCG_at_3 = NDCG(cs_similarity, test_labels, 3)
    NDCG_at_10 = NDCG(cs_similarity, test_labels, 5)
    return ranking_performance, NDCG_at_1, NDCG_at_3, NDCG_at_10


def test_ranking(model, test_batches):
    num_batches = len(test_batches)
    map, ndcg_1, ndcg_3, ndcg_10 = 0, 0, 0, 0
    for batch_no in range(1, num_batches + 1):
        test_queries, test_docs, test_labels = helper.batch_to_tensor(test_batches[batch_no - 1], len(model.dictionary))
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
    dictionary = helper.load_object(args.save_path + 'dictionary.p')
    model = CLSM(dictionary, args)

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_devices = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(cuda_visible_devices) > 1:
            model = torch.nn.DataParallel(model, device_ids=cuda_visible_devices)
    if args.cuda:
        model = model.cuda()

    helper.load_model_states_from_checkpoint(model, os.path.join(args.save_path, 'model_best.pth.tar'), 'state_dict')
    print('Model and dictionary loaded.')
    model.eval()

    test_corpus = data.Corpus(args.data, 'session_test.txt', dictionary, is_test_corpus=True)
    print('Test set size = ', len(test_corpus.data))

    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    print('Number of test batches = ', len(test_batches))

    test_ranking(model, test_batches)
