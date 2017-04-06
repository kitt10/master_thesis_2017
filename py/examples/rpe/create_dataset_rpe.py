#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    examples.rpe.create_dataset_rpe
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates a RULE-PLUS-EXCEPTION (AB+abcd) dataset for kitt_lib framework.

    @arg n_samples          : number of samples per class
    @arg data_split         : training : validation : testing data split
    @arg name_appendix      : appendix to the destination name
"""

from kitt_monkey import print_message
from argparse import ArgumentParser
from sys import stderr
from random import choice
from shelve import open as open_shelve
from numpy import array, arange


def parse_arguments():  
    parser = ArgumentParser(description='Creates an XOR dataset for kitt_lib.')
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
    destination = 'dataset_rpe'+args.name_appendix+'.ds'

    print_message(message='Generating and splitting RPE data...')
    data = {'x': list(), 'y': list(), 'x_val': list(), 'y_val': list(), 'x_test': list(), 'y_test': list()}
    for ni in range(args.n_samples):
        a = choice((0, 1))
        b = choice((0, 1))
        c = choice((0, 1))
        d = choice((0, 1))
        if a == 1 and b == 1:
            y = 1
        elif 1 not in (a, b, c, d):
            y = 1
        else:
            y = 0

        ''' train/val/test split '''
        if ni < split_bounds[0]:
            data['x'].append(array([a, b, c, d], ndmin=2).T)
            data['y'].append(y)
        elif split_bounds[0] <= ni < split_bounds[1]:
            data['x_val'].append(array([a, b, c, d], ndmin=2).T)
            data['y_val'].append(y)
        else:
            data['x_test'].append(array([a, b, c, d], ndmin=2).T)
            data['y_test'].append(y)
    
    print_message(message='Got RPE dataset: '+str(len(data['x']))+' : '+str(len(data['x_val']))+' : '+str(len(data['x_test']))+', saving...')
    dataset = open_shelve(destination, 'c')
    for key, value in data.items():
        dataset[key] = value
    dataset.close()
    print_message(message='Dataset dumped as '+destination)
