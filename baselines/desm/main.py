###############################################################################
# Author: Wasi Ahmad
# Project: Deep Semantic Similarity Model
# Date Created: 7/18/2017
#
# File Description: This script is the entry point of the entire pipeline.
###############################################################################

import util, helper, data, numpy

args = util.get_args()

###############################################################################
# Load data
###############################################################################

dictionary = data.Dictionary()
test_corpus = data.Corpus(args.data, 'session_test.txt', dictionary)
print('test set size = ', len(test_corpus.data))
print('vocabulary size = ', len(dictionary))

# embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, args.word_vectors_file)
# helper.save_word_embeddings(args.word_vectors_directory, 'glove.840B.300d.desm.txt', embeddings_index,
#                             dictionary.idx2word)

embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, 'glove.840B.300d.desm.txt')
print('Number of OOV words = ', len(dictionary) - len(embeddings_index))


# ###############################################################################
# # Test
# ###############################################################################

def compute_document_embedding(document_body):
    emb = numpy.zeros(300)
    total_found = 0
    for term in document_body:
        if term in embeddings_index:
            emb = numpy.add(numpy.array(embeddings_index[term]), emb)
            total_found += 1

    if total_found > 0:
        emb = emb / total_found
    return emb


def compare_query_with_document(query, document):
    doc_emb = compute_document_embedding(document)
    score = 0
    total_found = 0
    for query_term in query:
        if query_term in embeddings_index:
            term_emb = numpy.array(embeddings_index[query_term])
            score += numpy.dot(term_emb, doc_emb) / (numpy.linalg.norm(term_emb) * numpy.linalg.norm(doc_emb))
            total_found += 1

    if total_found > 0:
        score = score / total_found
    return score


def compute_average_precision(score, click_label):
    indices = numpy.array(score).argsort()[::-1]
    average_precision = 0
    num_rel = 0
    for i in range(len(indices)):
        if click_label[indices[i]] == 1:
            num_rel += 1
            average_precision += num_rel / (i + 1)
    return average_precision


def NDCG(logits, target, k):
    indices = numpy.array(logits).argsort()[::-1]
    DCG_ref = 0
    num_rel_docs = numpy.count_nonzero(target)
    for j in range(len(indices)):
        if j == k:
            break
        if target[indices[j]] == 1:
            DCG_ref += 1 / numpy.log2(j + 2)
    DCG_gt = 0
    for j in range(num_rel_docs):
        if j == k:
            break
        DCG_gt += 1 / numpy.log2(j + 2)

    return DCG_ref / DCG_gt


mean_average_precision = 0
ndcg_at_1 = 0
ndcg_at_3 = 0
ndcg_at_10 = 0
for query in test_corpus.data:
    score = []
    click_labels = []
    for document in query.rel_docs:
        score.append(compare_query_with_document(query.query_terms, document.body))
        click_labels.append(1 if document.is_clicked else 0)
    mean_average_precision += compute_average_precision(score, click_labels)
    ndcg_at_1 += NDCG(score, click_labels, 1)
    ndcg_at_3 += NDCG(score, click_labels, 3)
    ndcg_at_10 += NDCG(score, click_labels, 5)

print('mean average precision - ', mean_average_precision / len(test_corpus.data))
print('NDCG@1 - ', ndcg_at_1 / len(test_corpus.data))
print('NDCG@3 - ', ndcg_at_3 / len(test_corpus.data))
print('NDCG@10 - ', ndcg_at_10 / len(test_corpus.data))
