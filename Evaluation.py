import os
import numpy as np
from tqdm import tqdm
from dataset import read_bin_generator
from utils import Vocab


def coherence_score(test_bin_fname, vocab, aspect_word_list, table_fname='./data/inverse_mat.npy'):
    '''
    Measure the coherence score of each aspect word list based on document frequency.
    '''
    inverse_mat = make_inverse_table(test_bin_fname, vocab, table_fname)  # [vocab, doc]

    # Assert that number of selected words are all same.
    assert len(list(set([len(words) for words in aspect_word_list]))) == 1

    score = []
    for words in aspect_word_list:
        for idx1, word1 in enumerate(words):
            if idx1 == 0: continue
            for idx2, word2 in enumerate(words[:idx1]):
                word1_occ_docs, word2_occ_docs = inverse_mat[word1], inverse_mat[word2]
                cooccur_cnt = len(np.intersect1d(word1_occ_docs, word2_occ_docs))
                score.append(np.log((cooccur_cnt + 1) / len(word2_occ_docs)))
    return sum(score) / len(score)


def aspect_identification_with_labels():
    pass


def make_inverse_table(test_bin_fname, vocab, table_fname):
    """
    Make inverse lookup table for testset to calculate the coherence score.
    Return matrix shape is [vocab_size, test_document_size].
    """
    if os.path.exists(table_fname):
        return np.load(table_fname)
    bin_gen = read_bin_generator(test_bin_fname, single_pass=True)

    def read_txt_generator():
        while True:
            example = next(bin_gen)
            op_txt = example.features.feature['text'].bytes_list.value[0].decode()
            yield op_txt

    # [doc_num, vocab_size] matrix
    matrix = []
    txt_gen = read_txt_generator()
    for doc in txt_gen:
        ids = [vocab.word2id[word] if word in vocab.words else vocab.unk_tok_id for word in doc.strip().split()]
        if len(ids) <= 1: continue
        row = [0 for _ in range(len(vocab.words))]
        for id in ids: row[id] += 1
        matrix.append(row)
    matrix = np.asarray(matrix, dtype=np.int)
    inverse_matrix = np.transpose(matrix)
    print(np.shape(inverse_matrix))
    np.save(table_fname, inverse_matrix)






if __name__ == '__main__':
    test_fname = './data/datasets/restaurant/parsed_test.bin'
    vocab = Vocab('./data/vocab.txt')

    table = make_inverse_table(test_fname, vocab)


