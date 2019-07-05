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
        reader = open(fname, 'rb')
        while True:
            len_bytes = reader.read(8)
            if not len_bytes: break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)
            if single_pass:
                break

class Vocab:
    def __init__(self, vocab_fname):
        self.word2id, self.id2word, self.words = self.build_ids(vocab_fname)
        self.unk_tok_id = self.word2id['<UNK>']
        self.pad_tok_id = self.word2id['<PAD>']

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


class Batcher:
    def __init__(self, vocab, hparams):
        self.hparams = hparams
        self.vocab = vocab
        self.train_gen = read_bin_generator(self.hparams.train_bin_fname)
        self.test_gen = read_bin_generator(self.hparams.test_bin_fname)

    def pack_one_sample(self, review_txt, label_txt):
        assert isinstance(review_txt, str) and (isinstance(label_txt, str) or label_txt is None)
        review_toks = review_txt.split()
        review_ids = [self.vocab.word2id[tok] if tok in self.vocab.words else self.vocab.unk_tok_id for tok in review_toks]
        if len(review_ids) > self.hparams.max_text_len: review_ids = review_ids[:self.hparams.max_text_len]

        review_pad = [1 for i in range(len(review_ids))]
        review_len = len(review_ids)

        while len(review_ids) != self.hparams.max_text_len:
            review_ids.append(self.vocab.pad_tok_id)
            review_pad.append(0)

        labels = [int(label) for label in label_txt.strip().split()] if label_txt is not None else []
        labels = np.asarray(labels)
        return review_ids, labels, review_pad, review_len

    def next_data(self):
        samples = []
        sample_pads = []
        sample_lens = []
        labels = []
        while len(samples) != self.hparams.batch_size:
            sample = next(self.read_txt_generator(self.hparams.mode == 'train'))
            ids_sample, label, pad_sample, len_sample = self.pack_one_sample(sample[0], sample[1])
            samples.append(ids_sample)
            labels.append(label)
            sample_pads.append([[1 if el == 1 else 0 for i in range(self.hparams.embed_dim)] for el in pad_sample])
            sample_lens.append(len_sample)

        batch = self.make_feeddict(samples, sample_pads, sample_lens)
        return batch, labels

    def make_feeddict(self, text, pad, lens):
        feed = {
            'enc_batch': np.asarray(text),
            'enc_pad': np.asarray(pad),
            'enc_len': np.asarray(lens)
        }
        return feed

    def read_txt_generator(self, is_train=True):
        while True:
            example = next(self.train_gen) if is_train else next(self.test_gen)
            op_txt = example.features.feature['text'].bytes_list.value[0].decode()
            label_txt = None if 'label' not in example.features.feature else \
            example.features.feature['label'].bytes_list.value[0].decode()
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


