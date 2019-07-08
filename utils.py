import os
import tensorflow as tf


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

