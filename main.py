import argparse
from dataset import Vocab, Batcher
from model import Model


parser = argparse.ArgumentParser()

# Path
parser.add_argument('--train_bin_fname', type=str, default='./data/datasets/restaurant/parsed_train.bin')
parser.add_argument('--test_bin_fname', type=str, default='./data/datasets/restaurant/parsed_test.bin')
parser.add_argument('--vocab_fname', type=str, default='./data/vocab.txt')
parser.add_argument('--model_path', type=str, default='./model')
# Experiments setting
parser.add_argument('--mode', type=str, default='train', choices=['train','test'])
parser.add_argument('--vocab_size', type=int, default=30000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=150)

# Neural net model related
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--max_length', type=int, default=30)
args = parser.parse_args()

def main():
    voca = Vocab(args.vocab_fname)
    model = Model()
    pass

def train():
    pass

def test():
    pass


if __name__ == '__main__':
    main()