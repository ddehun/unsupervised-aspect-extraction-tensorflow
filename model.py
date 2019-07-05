import os
import numpy as np
import tensorflow as tf


def build_embed_matrix(vocab, glove_path, custom_embed_path, dim):
    if os.path.exists(custom_embed_path):
        print("Already constructed matrix")
        matrix = np.load(custom_embed_path)
        return matrix
    else:
        matrix = []
        glove_dict = {}
        with open(glove_path, 'r', encoding='utf8') as f:
            ls = f.readlines()
            for line in ls:
                line = line.strip().split()
                assert len(line) == 1 + dim
                word, vec = line[0], [float(var) for var in line[1:]]
                glove_dict[word] = vec

        for token in vocab.words:
            if token in glove_dict: matrix.append(glove_dict[token])
            else:
                rand_var = np.random.normal(size=dim)
                matrix.append(rand_var)
        matrix = np.asarray(matrix)
        np.save(custom_embed_path, matrix)
        return matrix


class Model:
    def __init__(self, hparams, vocab):
        self.hparams = hparams
        self.vocab = vocab

    def build_graph(self):
        print("build graph start")
        self.add_placeholder()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        print("Add embed matrix start")
        self.add_embedding()

        print("Add model")
        self.add_model()

        print("Add train op")
        self.add_train_op()

        self.summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=10)

    def add_placeholder(self):
        self.text_input = tf.placeholder(tf.int32, [self.hparams.batch_size, self.hparams.max_text_len], name='enc_batch')
        self.text_pad_mask = tf.placeholder(tf.float64, [self.hparams.batch_size, self.hparams.max_text_len, self.hparams.embed_dim], name='enc_pad')
        self.text_len = tf.placeholder(tf.int32, [self.hparams.batch_size], name='enc_len')

    def add_embedding(self):
        matrix = build_embed_matrix(self.vocab, self.hparams.glove_matrix_fname, self.hparams.custom_embed_fname, self.hparams.embed_dim)
        self.embedding_matrix = tf.Variable(matrix, trainable=True)

        # [batch_size, max_len, embed_dim]
        self.embed_output = tf.nn.embedding_lookup(self.embedding_matrix, self.text_input)

        self.pad_embed = self.embed_output * self.text_pad_mask

    def add_model(self):
        pass

    def add_train_op(self):
        pass

    def run_train_step(self, batch, sess):
        batch = self.make_feeddict(batch)
        return sess.run([self.embed_output, self.res], feed_dict=batch)

    def make_feeddict(self, batch):
        feed = {
            self.text_input: batch['enc_batch'],
            self.text_pad_mask: batch['enc_pad'],
            self.text_len: batch['enc_len']
        }
        return feed

    def run_eval(self):
        pass