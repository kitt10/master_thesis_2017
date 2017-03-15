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
import numpy as np

def parse_arguments():
    parser = ArgumentParser(description='Creates a MNIST dataset for kitt_lib.')
    parser.add_argument('-na', '--name_appendix', type=str, default='',
                        help='Dataset filename appendix')
    return parser.parse_args()


def load_data_wrapper(data_src):
    with open_gzip(data_src, 'rb') as f:
        tr_d, va_d, te_d = load_cpickle(f)

    the_x = dict()
    the_y = dict()

    the_x['training'] = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    the_y['training'] = tr_d[1]
    the_x['validation'] = [np.reshape(x, (784, 1)) for x in va_d[0]]
    the_y['validation'] = va_d[1]
    the_x['testing'] = [np.reshape(x, (784, 1)) for x in te_d[0]]
    the_y['testing'] = te_d[1]
    return the_x, the_y

if __name__ == '__main__':
    args = parse_arguments()
    destination = 'dataset_mnist'+args.name_appendix+'.ds'

    print_message(message='Loading YannLecun\'s MNIST data...')
    with open_gzip('../../../data/data_mnist/mnist.pkl.gz', 'rb') as f:
        data_train, data_val, data_test = load_cpickle(f)
    
    print_message(message='Got MNIST dataset: '+str(len(data_train[0]))+' : '+str(len(data_val[0]))+' : '+str(len(data_test[0]))+', saving...')
    dataset = open_shelve(destination, 'c')
    dataset['x'] = [np.reshape(x, (784, 1)) for x in data_train[0]]
    dataset['y'] = data_train[1]
    dataset['x_val'] = [np.reshape(x, (784, 1)) for x in data_val[0]]
    dataset['y_val'] = data_val[1]
    dataset['x_test'] = [np.reshape(x, (784, 1)) for x in data_test[0]]
    dataset['y_test'] = data_test[1]
    dataset.close()
    print_message(message='Dataset dumped as '+destination)
