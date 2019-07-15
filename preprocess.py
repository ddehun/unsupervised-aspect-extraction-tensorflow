import re
import os
import struct
import argparse
import collections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.core.example import example_pb2
from utils import is_num


parser = argparse.ArgumentParser()

# Path
parser.add_argument('--dataset', type=str, choices=['restaurant', 'beer'], default='restaurant')
parser.add_argument('--vocab_fname', type=str, default='./data/vocab.txt')

# Experiments setting
parser.add_argument('--mode', type=str, default='train', choices=['train','test'])
parser.add_argument('--vocab_size', type=int, default=8000)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_step', type=int, default=65000)
parser.add_argument('--embed_dim', type=int, default=200)
parser.add_argument('--near_K', type=int, default=5)
parser.add_argument('--min_word_cnt', type=int, default=10)

args = parser.parse_args()

"""
Preprocessing script file
"""


def tokenize_sent(sent, lemtzr, stopword):
    tokens = tokenize(sent.strip().lower())
    tokens = re.sub(r'[^A-Za-z0-9]+',' ', ' '.join(tokens))
    tokens = [tok for tok in tokens.split() if tok not in stopword]
    tokens = [tok if not is_num(tok) else '<NUM>' for tok in tokens]
    tokens = [lemtzr.lemmatize(tok) for tok in tokens]
    return tokens


def tokenize_train_file(fname):
    """
    Tokenize the raw train data(unlabeled).
    """
    split_fname = fname.split('/')
    new_fname = '/'.join([el if idx != len(split_fname) - 1 else 'parsed_' + el for idx, el in enumerate(split_fname)])
    if os.path.exists(new_fname): return new_fname

    with open(fname, 'r', encoding='utf8') as f:
        ls = f.readlines()

    parsed_data = []
    lemtzr = WordNetLemmatizer()
    stopword = stopwords.words('english')

    for line in ls:
        tokens = tokenize_sent(line, lemtzr, stopword)
        parsed_data.append(tokens)

    save_file(parsed_data, new_fname)
    return new_fname


def tokenize_labeled_test_file(fname, label_fname):
    """
    Tokenize the raw test data (labelled).
    """
    split_fname = fname.split('/')
    new_fname = '/'.join([el if idx != len(split_fname) - 1 else 'parsed_' + el for idx, el in enumerate(split_fname)])
    label_map_fname = '/'.join([el if idx != len(split_fname) - 1 else 'label_map.txt' for idx, el in enumerate(split_fname)])
    if os.path.exists(new_fname): return label_map_fname, new_fname

    with open(fname, 'r', encoding='utf8') as f1, open(label_fname, 'r', encoding='utf8') as f2:
        ls1, ls2 = f1.readlines(), f2.readlines()

    parsed_data = []
    lemtzr = WordNetLemmatizer()
    stopword = stopwords.words('english')

    for line in ls1:
        tokens = tokenize_sent(line, lemtzr, stopword)
        parsed_data.append(tokens)

    assert len(ls1) == len(ls2) == len(parsed_data)

    new_parsed, new_ls2 = [], []
    for parsed, label in zip(parsed_data, ls2):
        if 'Positive' in label or 'Neutral' in label:
            continue
        new_parsed.append(parsed)
        new_ls2.append(label)

    assert len(new_parsed) == len(new_ls2)
    parsed_data, ls2 = new_parsed, new_ls2

    label_text = list(set([tok for line in ls2 for tok in line.strip().split()]))

    label_map = dict()
    print("Label for this dataset with assigned index is as follows.")
    for idx, label in enumerate(label_text):
        print('{}: {}'.format(label, idx))
        label_map[label] = idx
    with open(label_map_fname, 'w') as f:
        for key,val in label_map.items():
            f.write("{}  {} ||| ".format(key, val))

    for idx, data in enumerate(parsed_data):
        labels = ls2[idx].strip().split()
        assert all([label in list(label_map.keys()) for label in labels])
        parsed_data[idx].insert(0, '|||')
        for label in labels:
            parsed_data[idx].insert(0, str(label_map[label]))

    save_file(parsed_data, new_fname)
    return label_map_fname, new_fname


def build_vocab(parsed_train_fname, vocab_file, vocab_size=30000):
    """
    Build vocab based on frequency of each word in train set.
    Save vocab file and return vocab list.
    """
    if os.path.exists(vocab_file):
        with open(vocab_file, 'r', encoding='utf8') as f:
            ls = f.readlines()
            assert len(ls) == vocab_size
        vocab = [line.strip() for line in ls]
        return vocab

    with open(parsed_train_fname, 'r', encoding='utf8') as f:
        ls = f.readlines()
        tokens = [tok for line in ls for tok in line.strip().split()]
    counts = dict(collections.Counter(tokens))

    import operator
    vocab = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    print("TOTAL VOCAB SIZE: {}".format(len(vocab)))

    for idx,tok in enumerate(vocab):
        if tok[1] <= 10:
            print("WORDS MORE THAN 10: {}".format(idx))
            break
    vocab = [tok[0] for tok in vocab][:idx]

    vocab.append('<UNK>')
    vocab.append('<PAD>')

    assert all([isinstance(tok, str) for tok in vocab])
    with open(vocab_file, 'w') as f:
        f.write('\n'.join([tok for tok in vocab]))
    return vocab


def save_file(data, new_fname):
    """
    Change the "raw_fname" into parsed fname, then save "data" into parsed fname.
    """
    assert isinstance(data, list)

    with open(new_fname, 'w') as f: f.write('\n'.join([" ".join(one_sample) for one_sample in data]))


def tokenize(sent):
    assert isinstance(sent, str)
    return word_tokenize(sent)


def make_binary_dataset(fname, is_label=False):
    """
    Make a binary data file for learning.
    """
    binary_fname = fname.replace('.txt', '.bin')
    if os.path.exists(binary_fname): return

    with open(fname, 'r', encoding='utf8') as f:
        ls = f.readlines()
        data = [line.strip() for line in ls]

    assert all(['|||' in dat for dat in data]) if is_label else all(['|||'  not in dat for dat in data])

    with open(binary_fname, 'wb') as f:
        for line in data:
            if is_label:
                split_line = line.split('|||')
                assert len(split_line) == 2
                label, text = split_line[0].strip(), split_line[1].strip()
            else:
                text = line
            example = example_pb2.Example()
            example.features.feature['text'].bytes_list.value.extend([text.encode()])
            if is_label:
                example.features.feature['label'].bytes_list.value.extend([label.encode()])
            example_str = example.SerializeToString()
            str_len = len(example_str)
            f.write(struct.pack('q', str_len))
            f.write(struct.pack('%ds' % str_len, example_str))
    return


def main():
    train_fname = './data/datasets/{}/train.txt'.format(args.dataset)
    test_fname = './data/datasets/{}/test.txt'.format(args.dataset)
    test_label_fname = './data/datasets/{}/test_label.txt'.format(args.dataset)
    vocab_fname = './data/vocab.txt'
    vocab_size = args.vocab_size

    parsed_train_fname = tokenize_train_file(train_fname)
    label_map, parsed_test_fname = tokenize_labeled_test_file(test_fname, test_label_fname)
    build_vocab(parsed_train_fname, vocab_fname, vocab_size=vocab_size)

    make_binary_dataset(parsed_train_fname, False)
    make_binary_dataset(parsed_test_fname, True)


if __name__ == '__main__':
    main()