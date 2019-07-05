import struct
import os
import numpy as np
import tensorflow as tf
from tensorflow.core.example import example_pb2


def read_bin_generator(fname, single_pass=False):
    """
    Read binary file and yield data.
    If single_pass is True, Iterating for dataset is done only once.
    """
    while True:
        with open(fname, 'rb') as f:
            len_bytes = f.read(8)
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, f.read(str_len))[0]
            yield example_pb2.Example.FromString(example_str)
        if single_pass:
            break

def parser



if __name__ == '__main__':
    train_fname = './data/datasets/restaurant/parsed_train.bin'
    test_fname = './data/datasets/restaurant/parsed_test.bin'

    gen = read_bin_generator(train_fname)
    test_gen = read_bin_generator(test_fname)
    print(next(gen))
    print(next(test_gen))

