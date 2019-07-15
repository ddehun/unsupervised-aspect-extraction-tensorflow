import os
import numpy as np
import tensorflow as tf


def build_embed_matrix(vocab, glove_path, custom_embed_path, dim):
    if os.path.exists(custom_embed_path):
        print("Already constructed matrix")
        matrix = np.load(custom_embed_path).astype(np.float64)
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
        matrix = np.asarray(matrix).astype(np.float64)
        np.save(custom_embed_path, matrix)
        return matrix


class Model:
    def __init__(self, hparams, vocab):
        self.hparams = hparams
        self.vocab = vocab

    def build_graph(self):
        print("build graph start")
        self.add_placeholder()
        self.uniform_initializer = tf.random_uniform_initializer(-self.hparams.aspect_emb_scale,
                                                                 self.hparams.aspect_emb_scale, dtype=tf.float64, seed=777)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        print("Add embed matrix start")
        self.add_embedding()

        print("Add model")
        self.add_model()

        print("Add train op")
        self.add_train_op()
        self.show_aspect_near_words()

        self.summaries = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=10)

    def add_placeholder(self):
        self.text_input = tf.placeholder(tf.int32, [self.hparams.batch_size, self.hparams.max_text_len], name='enc_batch')
        self.text_pad_mask = tf.placeholder(tf.float64, [self.hparams.batch_size, self.hparams.max_text_len, self.hparams.embed_dim], name='enc_pad')
        self.text_len = tf.placeholder(tf.float64, [self.hparams.batch_size], name='enc_len')
        self.neg_text_input = tf.placeholder(tf.int32, [self.hparams.negative_samples*self.hparams.batch_size, self.hparams.max_text_len], name='neg_enc_batch')
        self.neg_text_pad_mask = tf.placeholder(tf.float64, [self.hparams.negative_samples*self.hparams.batch_size, self.hparams.max_text_len, self.hparams.embed_dim], name='neg_enc_pad')
        self.neg_text_len = tf.placeholder(tf.float64, [self.hparams.negative_samples*self.hparams.batch_size], name='neg_enc_len')
        self.near_k = tf.placeholder_with_default(1, (), name='top_k_nearsest_word')

    def add_embedding(self):
        with tf.variable_scope('embedding'):
            matrix = build_embed_matrix(self.vocab, self.hparams.glove_matrix_fname, self.hparams.custom_embed_fname, self.hparams.embed_dim)
            self.embedding_matrix = tf.Variable(matrix, trainable=True, dtype=tf.float64)

            # [batch_size, max_len, embed_dim]
            self.embed_output = tf.nn.embedding_lookup(self.embedding_matrix, self.text_input)

            # zero-padding for <PAD> token
            self.pad_embed_output = self.embed_output * self.text_pad_mask

            # [batch_size, embed_dim]
            self.embed_sum = tf.reduce_sum(self.pad_embed_output, axis=1)

            # [batch_size, embed_dim]
            self.embed_avg = self.embed_sum / tf.reshape(self.text_len, (-1, 1))

            # [aspect_num, embed_dim]
            self.aspect_matrix = tf.get_variable('aspect_emb', [self.hparams.aspect_num, self.hparams.embed_dim],
                                                 dtype=tf.float64, initializer=self.uniform_initializer)

            # Below is for negative sample word embedding
            # [negative_sample_size * batch_size, max_len, embed_dim]
            self.neg_embed_output = tf.nn.embedding_lookup(self.embedding_matrix, self.neg_text_input)

            # zero-padding for <PAD> token
            self.neg_pad_embed_output = self.neg_embed_output * self.neg_text_pad_mask

            # [negative_sample_size * batch_size, embed_dim]
            self.neg_embed_sum = tf.reduce_sum(self.neg_pad_embed_output, axis=1)

            # [negative_sample_size * batch_size, embed_dim]
            self.neg_embed_avg = self.neg_embed_sum / tf.reshape(self.neg_text_len, (-1, 1))

    def add_model(self):
        with tf.variable_scope('encoding'):
            self.sentence_encoding()

        with tf.variable_scope('reconstruction'):
            self.reconstruct()
            self.add_penalize_term()

    def sentence_encoding(self):
        """
        Sentence encoding using self-attention mechanism with word embedding
        """

        # Matrix to calculate attention score between global context(avg. of word embedding for sentence) and each word
        matrix1 = tf.get_variable('m1', [self.hparams.embed_dim, self.hparams.embed_dim], dtype=tf.float64, initializer=self.uniform_initializer)

        # [batch_size, word_dim, 1]
        tmp1 = tf.expand_dims(tf.transpose(tf.matmul(matrix1, self.embed_avg, transpose_b=True)),2)

        # [batch_size, max_len]
        score = tf.squeeze(tf.matmul(self.pad_embed_output, tmp1),axis=2)

        attn_dist = tf.nn.softmax(score)

        # Masking not to attend for <PAD> token.
        attn_dist *= tf.reduce_mean(self.text_pad_mask, 2)  # Set attention weight of <PAD> token to be zero.
        self.attn_dist = attn_dist / tf.reshape(tf.reduce_sum(attn_dist, 1), [-1, 1])  # Re-normalize the attention distribution.

        attn_dist += 1e-12

        # Weighted sum of word embedding with attention distribution.
        self.sent_repr = tf.reduce_sum(self.pad_embed_output * tf.expand_dims(self.attn_dist, axis=2), 1)

    def reconstruct(self):
        weight = tf.get_variable('w1', [self.hparams.embed_dim, self.hparams.aspect_num], dtype=tf.float64, initializer=self.uniform_initializer)
        bias = tf.get_variable('b1', [self.hparams.aspect_num], dtype=tf.float64, initializer=self.uniform_initializer)

        # [batch_size, aspect_num]
        self.aspect_prob = tf.nn.softmax(tf.matmul(self.sent_repr, weight) + bias)

        self.aspect_prob += 1e-12

        # [batch_size, embed_dim]
        self.reconstruct_sent = tf.matmul(self.aspect_prob, self.aspect_matrix)

    def add_penalize_term(self):
        """
        Penalize the aspect matrix to avoid redundant aspects.
        """
        # Normalized aspect embedding
        normalized_aspect_matrix = self.aspect_matrix / tf.expand_dims(tf.norm(self.aspect_matrix + 1e-12, axis=1), axis=1)

        TT_T = tf.matmul(normalized_aspect_matrix, tf.transpose(normalized_aspect_matrix, [1, 0]))
        I = tf.eye(self.hparams.aspect_num, dtype=tf.float64)

        self.penalty_term = tf.square(tf.norm(TT_T - I + 1e-12, axis=[-2, -1], ord='fro')) + 1e-12
        self.penalty_term = tf.where(tf.is_nan(self.penalty_term), tf.constant(1.0, dtype=tf.float64), self.penalty_term)
        tf.summary.scalar('penalty', self.penalty_term)

    def calc_recons_loss(self):
        """
        Reconstruction loss calculation by hinge loss with negative sample embedding.
        """
        # shape [batch_size, 1]
        positive_loss = tf.diag_part(tf.matmul(self.reconstruct_sent, self.sent_repr, transpose_b=True))

        positive_loss = tf.reshape(tf.tile(tf.expand_dims(positive_loss, 1), [1, self.hparams.negative_samples]), (-1,1))

        tiled_recons = tf.reshape(tf.tile(self.reconstruct_sent, [1, self.hparams.negative_samples]), [-1, self.hparams.embed_dim])

        negative_loss = tf.diag_part(tf.matmul(tiled_recons, self.neg_embed_avg, transpose_b=True))

        hinge_loss = tf.reduce_sum(tf.maximum(tf.constant(0.0, dtype=tf.float64), 1. - positive_loss + negative_loss)) + 1e-12
        return hinge_loss

    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.lr)

        self.reconstruction_loss = self.calc_recons_loss()
        tf.summary.scalar('recons_loss', self.reconstruction_loss)

        self.loss = self.reconstruction_loss + self.hparams.penalty_weight * self.penalty_term
        tf.summary.scalar('loss', self.loss)
        tvars = tf.trainable_variables()

        gradients = tf.gradients(
            self.loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        grad, grad_norm = tf.clip_by_global_norm(gradients, self.hparams.max_grad_norm)

        tf.summary.scalar('global_norm', grad_norm)

        self.op = optimizer.apply_gradients(zip(grad, tvars), global_step=self.global_step)

    def run_train_step(self, batch, sess):
        batch = self.make_feeddict(batch)
        return sess.run([self.op, self.loss, self.global_step, self.summaries], feed_dict=batch)

    def run_eval_step(self, batch, sess):
        batch = self.make_feeddict(batch)
        return sess.run([self.loss, self.global_step, self.summaries], feed_dict=batch)

    def get_nearest_words(self, sess, k):
        near_ids = sess.run(self.top_ids, feed_dict={self.near_k: k})
        near_words = {idx: [self.vocab.id2word[id_] for id_ in word_ids] for idx, word_ids in enumerate(near_ids)}
        return near_ids, near_words

    def make_feeddict(self, batch):
        feed = {
            self.text_input: batch['enc_batch'],
            self.text_pad_mask: batch['enc_pad'],
            self.text_len: batch['enc_len'],
            self.neg_text_input: batch['neg_enc_batch'],
            self.neg_text_pad_mask: batch['neg_enc_pad'],
            self.neg_text_len: batch['neg_enc_len'],
        }
        return feed

    def show_aspect_near_words(self):
        norm_aspect_emb = tf.nn.l2_normalize(self.aspect_matrix)
        norm_word_emb = tf.nn.l2_normalize(self.embedding_matrix)

        sim_matrix = tf.matmul(norm_aspect_emb, norm_word_emb, transpose_b=True)
        top_val, self.top_ids = tf.nn.top_k(sim_matrix, self.near_k)

