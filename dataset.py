import numpy as np
from utils import Vocab, read_bin_generator
import random


class Batcher:
    def __init__(self, vocab, hparams, single_pass=False):
        self.hparams = hparams
        self.vocab = vocab
        self.train_gen = read_bin_generator(self.hparams.train_bin_fname, single_pass)
        self.negative_train_gen = read_bin_generator(self.hparams.train_bin_fname, single_pass)
        self.test_gen = read_bin_generator(self.hparams.test_bin_fname, single_pass)
        self.negative_test_gen = read_bin_generator(self.hparams.test_bin_fname, single_pass)

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

    def next_data(self, is_valid=False):
        samples = []
        sample_pads = []
        sample_lens = []
        labels = []
        negative_samples = []
        negative_sample_pads = []
        negative_sample_lens = []
        while len(samples) != self.hparams.batch_size:
            sample = next(self.read_txt_generator(self.hparams.mode == 'train' and not is_valid))
            if len(sample[0].split()) <= 3: continue
            ids_sample, label, pad_sample, len_sample = self.pack_one_sample(sample[0], sample[1])
            samples.append(ids_sample)
            labels.append(label)
            sample_pads.append([[1 if el == 1 else 0 for i in range(self.hparams.embed_dim)] for el in pad_sample])
            sample_lens.append(len_sample)

        if self.hparams.mode == 'train':
            while len(negative_samples) != self.hparams.negative_samples * self.hparams.batch_size:
                sample = next(self.read_negative_txt_generator(not is_valid))
                if len(sample[0].split()) <= 3: continue
                negative_ids_sample, _, negative_pad_sample, negative_len_sample = self.pack_one_sample(sample[0], sample[1])
                negative_samples.append(negative_ids_sample)
                negative_sample_pads.append([[1 if el == 1 else 0 for i in range(self.hparams.embed_dim)] for el in negative_pad_sample])
                negative_sample_lens.append(negative_len_sample)

        batch = self.make_feeddict(samples, sample_pads, sample_lens, negative_samples, negative_sample_pads, negative_sample_lens)
        return batch, labels

    def make_feeddict(self, text, pad, lens, neg_text, neg_pad, neg_lens):
        feed = {
            'enc_batch': np.asarray(text),
            'enc_pad': np.asarray(pad),
            'enc_len': np.asarray(lens),
            'neg_enc_batch': np.asarray(neg_text),
            'neg_enc_pad': np.asarray(neg_pad),
            'neg_enc_len': np.asarray(neg_lens),
        }
        return feed

    def read_txt_generator(self, read_train=True):
        while True:
            example = next(self.train_gen) if read_train else next(self.test_gen)
            op_txt = example.features.feature['text'].bytes_list.value[0].decode()
            label_txt = None if 'label' not in example.features.feature else \
            example.features.feature['label'].bytes_list.value[0].decode()
            if label_txt is None: assert read_train
            yield (op_txt, label_txt)

    def read_negative_txt_generator(self, read_train=True):
        assert self.hparams.mode == 'train'
        while True:
            # negative random sampling implementation strategy
            for _ in range(random.randint(10, 30)):
                example = next(self.negative_train_gen) if read_train else next(self.negative_test_gen)
            op_txt = example.features.feature['text'].bytes_list.value[0].decode()
            label_txt = None
            yield (op_txt, label_txt)




if __name__ == '__main__':
    train_fname = './data/datasets/restaurant/parsed_train.bin'
    test_fname = './data/datasets/restaurant/parsed_test.bin'

    gen = read_bin_generator(train_fname)
    test_gen = read_bin_generator(test_fname)
    print(next(gen))
    print(next(test_gen))
    vocab = Vocab('./data/vocab.txt')




