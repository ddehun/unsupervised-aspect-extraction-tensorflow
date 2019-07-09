import os
import struct
import numpy as np
import tensorflow as tf
from tensorflow.core.example import example_pb2


def read_bin_generator(fname, single_pass=False):
    """
    Read binary file and yield data.
    If single_pass is True, Iterating for dataset is done only once.
    """
    while True:
        # Suppose there is only one binary
        reader = open(fname, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)
        if single_pass:
            return

def load_ckpt(ckpt_dir, sess, saver):
    ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = ckpt_state.model_checkpoint_path
    saver.restore(sess, ckpt_path)

def GPU_config():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


class Vocab:
    def __init__(self, vocab_fname):
        self.word2id, self.id2word, self.words = self.build_ids(vocab_fname)
        self.unk_tok_id = self.word2id['<UNK>']
        self.pad_tok_id = self.word2id['<PAD>']
        self.words.extend(['<UNK>', '<PAD>'])
        self.vocab_size = len(self.words)

    def build_ids(self, vocab_fname):
        w2i, i2w = dict(), dict()
        words = []
        with open(vocab_fname, 'r', encoding='utf8') as f:
            ls = [line.strip() for line in f.readlines()]
            for idx, line in enumerate(ls):
                words.append(line)
                w2i[line] = idx
                i2w[idx] = line
        return w2i, i2w, words


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


def make_inverse_table(test_bin_fname, vocab, table_fname):
    """
    Make inverse lookup table for testset to calculate the coherence score.
    Return matrix shape is [vocab_size, test_document_size].
    """
    if os.path.exists(table_fname):
        return np.load(table_fname)
    bin_gen = read_bin_generator(test_bin_fname, single_pass=True)

    def read_simple_txt_generator():
        while True:
            example = next(bin_gen)
            op_txt = example.features.feature['text'].bytes_list.value[0].decode()
            yield op_txt

    # [doc_num, vocab_size] matrix
    matrix = []
    txt_gen = read_simple_txt_generator()
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