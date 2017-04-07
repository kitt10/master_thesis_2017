#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    examples.train.create_dataset_train
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates a TRAIN classification problem dataset for kitt_lib framework.

    @arg n_samples          : number of samples per class
    @arg data_split         : training : validation : testing data split
    @arg name_appendix      : appendix to the destination name
"""

from kitt_monkey import print_message, print_param
from argparse import ArgumentParser
from sys import stderr
from random import choice, uniform
from shelve import open as open_shelve
from numpy import array, arange


def parse_arguments():  
    parser = ArgumentParser(description='Creates an TRAIN dataset for kitt_lib.')
    parser.add_argument('-ns', '--n_samples', type=int, default=1000,
                        help='Number of samples per class')
    parser.add_argument('-ds', '--data_split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        choices=list(arange(start=0.0, stop=1.0, step=0.01)),
                        help='Training : Validation : Testing data split')
    parser.add_argument('-na', '--name_appendix', type=str, default='',
                        help='Dataset filename')
    args_tmp = parser.parse_args()

    ''' Check args '''
    if abs(sum(args_tmp.data_split) - 1) > 1e-5:
        stderr.write('Error: data_split args must give 1.0 together (e.g. 0.8 0.1 0.1).\n')
        exit()
    else:
        return args_tmp

if __name__ == '__main__':
    args = parse_arguments()
    split_bounds = (args.n_samples*args.data_split[0], args.n_samples*(args.data_split[0]+args.data_split[1]))
    destination = 'dataset_train'+args.name_appendix+'.ds'

    print_message(message='Generating and splitting TRAIN data...')
    data = {'x': list(), 'y': list(), 'x_val': list(), 'y_val': list(), 'x_test': list(), 'y_test': list()}
    for ni in range(args.n_samples):
        if ni%3 == 0:
            x_east = [0, 0, 1, 1, 0, 1, 0]
            x_west = [0, 1, 1, 1, 1, 0, 0]
        elif ni%3 == 1:
            x_east = [0, 0, 0, 1, 1, 0, 0]
            x_west = [1, 0, 1, 1, 1, 0, 0]
        else:
            x_east = [0, 0, 0, 1, 0, 1, 1]
            x_west = [1, 1, 1, 0, 1, 1, 1]

        ''' train/val/test split '''
        if ni < split_bounds[0]:
            data['x'].append(array(x_east, ndmin=2).T)
            data['x'].append(array(x_west, ndmin=2).T)
            data['y'].append('east')
            data['y'].append('west')
        elif split_bounds[0] <= ni < split_bounds[1]:
            data['x_val'].append(array(x_east, ndmin=2).T)
            data['x_val'].append(array(x_west, ndmin=2).T)
            data['y_val'].append('east')
            data['y_val'].append('west')
        else:
            data['x_test'].append(array(x_east, ndmin=2).T)
            data['x_test'].append(array(x_west, ndmin=2).T)
            data['y_test'].append('east')
            data['y_test'].append('west')
    
    print_message(message='Got TRAIN dataset: '+str(len(data['x']))+' : '+str(len(data['x_val']))+' : '+str(len(data['x_test']))+', saving...')
    dataset = open_shelve(destination, 'c')
    for key, value in data.items():
        dataset[key] = value
    dataset.close()
    print_message(message='Dataset dumped as '+destination)
