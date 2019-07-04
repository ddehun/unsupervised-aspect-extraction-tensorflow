import os
import collections
from nltk.tokenize import word_tokenize


"""
Preprocessing script file
"""


def tokenize_train_file(fname):
    """
    Tokenize the raw train data(unlabeled).
    """
    with open(fname, 'r', encoding='utf8') as f:
        ls = f.readlines()
    parsed_data = [[tok.lower() for tok in tokenize(line.strip())] for line in ls]

    new_fname = save_file(parsed_data, fname)
    return new_fname


def tokenize_labeled_test_file(fname, label_fname):
    """
    Tokenize the raw test data (labelled).
    """
    with open(fname, 'r', encoding='utf8') as f1, open(label_fname, 'r', encoding='utf8') as f2:
        ls1, ls2 = f1.readlines(), f2.readlines()

    parsed_data = [[tok.lower() for tok in tokenize(line.strip())] for line in ls1]

    assert len(ls1) == len(ls2) == len(parsed_data)

    label_text = list(set([tok.strip() for tok in ls2]))
    label_map = dict()
    print("Label for this dataset with assigned index is as follows.")
    for idx, label in enumerate(label_text):
        print('{}: {}'.format(label, idx))
        label_map[label] = idx

    for idx, data in enumerate(parsed_data):
        assert ls2[idx].strip() in list(label_map.keys())
        parsed_data[idx].insert(0, str(label_map[ls2[idx].strip()]))

    new_fname = save_file(parsed_data, fname)
    return label_map, new_fname


def build_vocab(parsed_train_fname, vocab_file, vocab_size=30000):
    """
    Build vocab based on frequency of each word in train set.
    Save vocab file and return vocab list.
    """
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


def save_file(data, raw_fname):
    '''
    Change the "raw_fname" into parsed fname, then save "data" into parsed fname.
    '''
    assert isinstance(data, list)
    split_fname = raw_fname.split('/')
    new_fname = '/'.join([el if idx != len(split_fname) - 1 else 'parsed_' + el for idx, el in enumerate(split_fname)])
    with open(new_fname, 'w') as f: f.write('\n'.join([" ".join(one_sample) for one_sample in data]))
    return new_fname


def tokenize(sent):
    assert isinstance(sent, str)
    return word_tokenize(sent)


if __name__ == '__main__':
    train_fname = './data/datasets/restaurant/train.txt'
    test_fname = './data/datasets/restaurant/test.txt'
    test_label_fname = './data/datasets/restaurant/test_label.txt'

    tokenize_train_file(train_fname)
    tokenize_labeled_test_file(test_fname, test_label_fname)