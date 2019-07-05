import struct
import os
import numpy as np
import tensorflow as tf
from tensorflow.core.example import example_pb2


def read_bin_generator(fname, single_pass=False):
    """
    Read binary file and yield data.
    If single_pass is True, Iterating for dataset is done only once.
    """
    while True:
        with open(fname, 'rb') as f:
            len_bytes = f.read(8)
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, f.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)
        if single_pass:
            break

class Vocab:
    def __init__(self, vocab_fname):
        self.word2id, self.id2word = self.build_ids(vocab_fname)
        self.unk_tok_id = self.word2id['<UNK>']
        self.words = list(self.word2id.keys())

    def build_ids(self, vocab_fname):
        w2i, i2w = dict(), dict()
        with open(vocab_fname, 'r', encoding='utf8') as f:
            ls = [line.strip() for line in f.readlines()]
            for idx, line in enumerate(ls):
                w2i[line] = idx
                i2w[idx] = line
        return w2i, i2w


class Batcher:
    def __init__(self, vocab, hparams):
        self.hparams = hparams
        self.vocab = vocab
        self.train_gen = read_bin_generator(self.hparams.train_bin_fname)
        self.test_gen = read_bin_generator(self.hparams.test)

    def pack_one_sample(self, review_txt, label_txt):
        assert isinstance(review_txt, str) and isinstance(label_txt, str)
        review_toks = review_txt.split()
        review_ids = np.asarray([self.vocab.word2id[tok] if tok in self.vocab.words else self.vocab.unk_tok_id for tok in review_toks])
        if len(review_ids) > self.hparams.max_length: review_ids = review_ids[:self.hparams.max_length]
        labels = [int(label) for label in label_txt.strip().split()] if label_txt is not None else []
        labels = np.asarray(labels)
        return review_ids, labels

    def next_data(self):
        samples = []
        labels = []
        while len(samples) == self.hparams.batch_size:
            sample = next(self.read_txt_generator(self.hparams.mode == 'train'))
            packed_sample = self.pack_one_sample(sample[0], sample[1])
            samples.append(packed_sample[0])
            labels.append(packed_sample[1])
        batch = np.asarray(samples)
        yield batch, labels

    def read_txt_generator(self, is_train=True):
        while True:
            example = next(self.train_gen) if is_train else next(self.test_gen)
            op_txt = example.features['text'].bytes_list.value[0].decode()
            label_txt = None if 'label' not in example.features else example.features['label'].bytes_list.value[0].decode()
            if label_txt is None: assert is_train
            yield (op_txt, label_txt)

if __name__ == '__main__':
    train_fname = './data/datasets/restaurant/parsed_train.bin'
    test_fname = './data/datasets/restaurant/parsed_test.bin'

    gen = read_bin_generator(train_fname)
    test_gen = read_bin_generator(test_fname)
    print(next(gen))
    print(next(test_gen))
    vocab = Vocab('./data/vocab.txt')


