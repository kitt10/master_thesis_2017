#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    examples.mnist.create_dataset_mnist
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script forms a MNIST dataset for kitt_lib framework (after downloading it from the Yan LeCun's page)

    @arg name_appendix   : appendix to the destination name
"""

from kitt_monkey import print_message
from argparse import ArgumentParser
from shelve import open as open_shelve
from gzip import open as open_gzip
from cPickle import load as load_cpickle
from numpy import reshape


def parse_arguments():
    parser = ArgumentParser(description='Creates a MNIST dataset for kitt_lib.')
    parser.add_argument('-ns', '--n_samples', type=int, default=-1,
                        help='Number of training samples per class.')
    parser.add_argument('-na', '--name_appendix', type=str, default='',
                        help='Dataset filename appendix')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    destination = 'dataset_mnist'+args.name_appendix+'.ds'

    print_message(message='Loading YannLecun\'s MNIST data...')
    with open_gzip('../../../data/data_mnist/mnist.pkl.gz', 'rb') as f:
        data_train, data_val, data_test = load_cpickle(f)

    dataset = open_shelve(destination, 'c')
    class_counter = dict()
    if args.n_samples == -1:
        print_message(message='Got MNIST dataset: '+str(len(data_train[0]))+' : '+str(len(data_val[0]))+' : '+str(
            len(data_test[0]))+', saving...')
        dataset['x'] = [reshape(x, (784, 1)) for x in data_train[0]]
        dataset['y'] = data_train[1]
    else:
        print_message(message='Got MNIST dataset: '+str(args.n_samples*10)+' : '+str(len(data_val[0]))+' : '+str(
            len(data_test[0]))+', saving...')
        tmp_x = list()
        tmp_y = list()
        for x, y in zip(data_train[0], data_train[1]):
            if y not in class_counter.keys():
                class_counter[y] = 0
            else:
                class_counter[y] += 1
            if class_counter[y] < args.n_samples:
                tmp_x.append(reshape(x, (784, 1)))
                tmp_y.append(y)
        dataset['x'] = tmp_x
        dataset['y'] = tmp_y

    dataset['x_val'] = [reshape(x, (784, 1)) for x in data_val[0]]
    dataset['y_val'] = data_val[1]
    dataset['x_test'] = [reshape(x, (784, 1)) for x in data_test[0]]
    dataset['y_test'] = data_test[1]
    dataset.close()
    print_message(message='Dataset dumped as '+destination)
