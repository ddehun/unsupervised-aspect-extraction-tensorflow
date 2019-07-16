import os
from pprint import pprint
import argparse
import tensorflow as tf
from tqdm import trange
from dataset import Vocab, Batcher
from model import Model
from utils import load_ckpt, GPU_config, coherence_score

parser = argparse.ArgumentParser()

# Path
parser.add_argument('--train_bin_fname', type=str, default='./data/datasets/restaurant/parsed_train.bin')
parser.add_argument('--test_bin_fname', type=str, default='./data/datasets/restaurant/parsed_test.bin')
parser.add_argument('--vocab_fname', type=str, default='./data/vocab.txt')
parser.add_argument('--model_path', type=str, default='./model/ckpt/')
parser.add_argument('--train_logdir', type=str, default='./model/logdir/train/')
parser.add_argument('--valid_logdir', type=str, default='./model/logdir/valid/')
parser.add_argument('--glove_matrix_fname', type=str, default='./data/glove.6B.200d.txt')
parser.add_argument('--custom_embed_fname', type=str, default='./data/emb_matrx.npy')
parser.add_argument('--nearword_fname', type=str, default='./data/near_words.txt')

# Experiments setting
parser.add_argument('--mode', type=str, default='train', choices=['train','test'])
parser.add_argument('--vocab_size', type=int, default=8000)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_step', type=int, default=30000)
parser.add_argument('--embed_dim', type=int, default=200)
parser.add_argument('--near_K', type=int, default=5)
parser.add_argument('--min_word_cnt', type=int, default=10)

# Neural net model related
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_text_len', type=int, default=20)
parser.add_argument('--aspect_num', type=int, default=15)
parser.add_argument('--aspect_emb_scale', type=float, default=0.5)
parser.add_argument('--penalty_weight', type=float, default=1.0)
parser.add_argument('--negative_samples', type=int, default=20)
parser.add_argument('--max_grad_norm', type=float, default=10.0)
args = parser.parse_args()


def main():
    print("Hello!")
    voca = Vocab(args.vocab_fname)
    model = Model(args, voca)
    batcher = Batcher(voca, args)

    with tf.Session(config=GPU_config()) as sess:
        model.build_graph()

        if args.mode == 'train':
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(args.train_logdir): os.makedirs(args.train_logdir)
            if not os.path.exists(args.valid_logdir): os.makedirs(args.valid_logdir)
            train_writer, valid_writer = tf.summary.FileWriter(args.train_logdir, sess.graph), tf.summary.FileWriter(args.valid_logdir, sess.graph)

            t = trange(args.max_step, leave=True)
            for i in t:
                sample, label = batcher.next_data()
                _, loss, step, summaries = model.run_train_step(sample, sess)
                t.set_description('Train loss: {}'.format(round(loss, 3)))
                train_writer.add_summary(summaries, step)

                if step % 5e3 == 0:
                    model.saver.save(sess, args.model_path, step)

                if step % 5 == 0:
                    valid_sample, valid_label = batcher.next_data(is_valid=True)
                    loss, step, summaries = model.run_eval_step(valid_sample, sess)
                    valid_writer.add_summary(summaries, step)
                    t.set_description('Valid loss: {}'.format(round(loss, 3)))

                if step % 100 == 0:
                    near_ids, near_words = model.get_nearest_words(sess, args.near_K)
                    pprint(near_words)
                    score = coherence_score(args.test_bin_fname, voca, near_ids)
                    summary = tf.Summary()
                    summary.value.add(tag='coherence_score_{}k'.format(args.near_K), simple_value=score)
                    valid_writer.add_summary(summary, step)

        else:
            load_ckpt(args.model_path, sess, model.saver)
            near_words_dict = {i:[] for i in range(args.aspect_num)}
            for k in range(5,50,5):
                near_ids, near_words = model.get_nearest_words(sess, k)
                score = coherence_score(args.test_bin_fname, voca, near_ids)
                print(k, score)
                for asp_idx in near_words:
                    for word in near_words[asp_idx]:
                        if word not in near_words_dict[asp_idx]:
                            near_words_dict[asp_idx].append(word)

            with open(args.nearword_fname, 'w') as f:
                for idx in range(len(list(near_words_dict.keys()))):
                    print(near_words_dict[idx])
                    f.write(str(idx) + '   ')
                    f.write(' '.join(near_words_dict[idx][:5]))
                    f.write('\n')


if __name__ == '__main__':
    main()