import argparse
import numpy as np
import tensorflow as tf
from dataset import Vocab, Batcher
from model import Model


parser = argparse.ArgumentParser()

# Path
parser.add_argument('--train_bin_fname', type=str, default='./data/datasets/restaurant/parsed_train.bin')
parser.add_argument('--test_bin_fname', type=str, default='./data/datasets/restaurant/parsed_test.bin')
parser.add_argument('--vocab_fname', type=str, default='./data/vocab.txt')
parser.add_argument('--model_path', type=str, default='./model')
parser.add_argument('--glove_matrix_fname', type=str, default='./data/glove.6B.200d.txt')
parser.add_argument('--custom_embed_fname', type=str, default='./data/emb_matrx.npy')

# Experiments setting
parser.add_argument('--mode', type=str, default='train', choices=['train','test'])
parser.add_argument('--vocab_size', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=150)
parser.add_argument('--embed_dim', type=int, default=200)

# Neural net model related
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--max_text_len', type=int, default=30)
parser.add_argument('--aspect_num', type=int, default=15)
args = parser.parse_args()

def main():
    print("Hello!")
    voca = Vocab(args.vocab_fname)
    model = Model(args, voca)
    batcher = Batcher(voca, args)

    with tf.Session() as sess:
        model.build_graph()
        sess.run(tf.global_variables_initializer())
        sample, label = batcher.next_data()

        res = model.run_train_step(sample, sess)
        print(np.shape(res[0]))
        print(np.shape(res[1]))

    if args.mode == 'train':
        pass

    else:
        pass



if __name__ == '__main__':
    main()