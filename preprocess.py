import os, struct
import collections
from nltk.tokenize import word_tokenize
from tensorflow.core.example import example_pb2
import tensorflow as tf


"""
Preprocessing script file
"""


def tokenize_train_file(fname):
    """
    Tokenize the raw train data(unlabeled).
    """
    split_fname = fname.split('/')
    new_fname = '/'.join([el if idx != len(split_fname) - 1 else 'parsed_' + el for idx, el in enumerate(split_fname)])
    if os.path.exists(new_fname): return new_fname

    with open(fname, 'r', encoding='utf8') as f:
        ls = f.readlines()
    parsed_data = [[tok.lower() for tok in tokenize(line.strip())] for line in ls]

    save_file(parsed_data, new_fname)
    return new_fname


def tokenize_labeled_test_file(fname, label_fname):
    """
    Tokenize the raw test data (labelled).
    """
    split_fname = fname.split('/')
    new_fname = '/'.join([el if idx != len(split_fname) - 1 else 'parsed_' + el for idx, el in enumerate(split_fname)])
    if os.path.exists(new_fname): return new_fname

    with open(fname, 'r', encoding='utf8') as f1, open(label_fname, 'r', encoding='utf8') as f2:
        ls1, ls2 = f1.readlines(), f2.readlines()

    parsed_data = [[tok.lower() for tok in tokenize(line.strip())] for line in ls1]

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

    for idx, data in enumerate(parsed_data):
        labels = ls2[idx].strip().split()
        assert all([label in list(label_map.keys()) for label in labels])
        parsed_data[idx].insert(0, '|||')
        for label in labels:
            parsed_data[idx].insert(0, str(label_map[label]))


    save_file(parsed_data, new_fname)
    return label_map, new_fname


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
    counts = collections.Counter(tokens)
    vocab = sorted(counts, key=lambda x: -counts[x])[:vocab_size-1]
    vocab.append('<UNK>')
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

    assert all(['|||'  in dat for dat in data]) if is_label else all(['|||'  not in dat for dat in data])

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
    train_fname = './data/datasets/restaurant/train.txt'
    test_fname = './data/datasets/restaurant/test.txt'
    test_label_fname = './data/datasets/restaurant/test_label.txt'
    vocab_fname = './data/vocab.txt'

    parsed_train_fname = tokenize_train_file(train_fname)
    parsed_test_fname = tokenize_labeled_test_file(test_fname, test_label_fname)
    build_vocab(parsed_train_fname, vocab_fname)

    make_binary_dataset(parsed_train_fname, False)
    make_binary_dataset(parsed_test_fname, True)


if __name__ == '__main__':
    main()