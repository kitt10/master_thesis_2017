#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    examples.xor.create_dataset_xor
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates an XOR dataset for kitt_lib framework.

    @arg class_dist         : distance between the classes
    @arg n_samples          : number of samples per class
    @arg data_split         : training : validation : testing data split
    @arg name_appendix      : appendix to the destination name
"""

from kitt_monkey import print_message
from argparse import ArgumentParser
from sys import stderr
from random import choice, uniform
from math import sin, cos, pi
from shelve import open as open_shelve
from numpy import array, arange


def parse_arguments():  
    parser = ArgumentParser(description='Creates an XOR dataset for kitt_lib.')
    parser.add_argument('-cd', '--class_dist', type=float, default=0.01,
                        help='Distance between classes')
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
    destination = 'dataset_xor'+args.name_appendix+'.ds'

    print_message(message='Generating and splitting XOR data...')
    data = {'x': list(), 'y': list(), 'x_val': list(), 'y_val': list(), 'x_test': list(), 'y_test': list()}
    for ni in range(args.n_samples):
        x0 = uniform(-0.5, 0.5)                                         # x-coordinate for sample of class 0
        y0 = uniform(-0.5+args.class_dist, 0.5-args.class_dist)         # y-coordinate for sample of class 0
        x1 = uniform(-0.5, 0.5)                                         # x-coordinate for sample of class 1
        y1 = choice([uniform(-1.0, -0.5), uniform(0.5, 1.0)])           # y-coordinate for sample of class 1

        ''' rotate points in space, 45deg '''
        x0_r = x0 * cos(pi / 4) - y0 * sin(pi / 4)
        y0_r = y0 * cos(pi / 4) + x0 * sin(pi / 4)
        x1_r = x1 * cos(pi / 4) - y1 * sin(pi / 4)
        y1_r = y1 * cos(pi / 4) + x1 * sin(pi / 4)

        ''' train/val/test split '''
        if ni < split_bounds[0]:
            data['x'].append(array([x0_r, y0_r], ndmin=2).T)
            data['x'].append(array([x1_r, y1_r], ndmin=2).T)
            data['y'].append(0.0)
            data['y'].append(1.0)
        elif split_bounds[0] <= ni < split_bounds[1]:
            data['x_val'].append(array([x0_r, y0_r], ndmin=2).T)
            data['x_val'].append(array([x1_r, y1_r], ndmin=2).T)
            data['y_val'].append(0.0)
            data['y_val'].append(1.0)
        else:
            data['x_test'].append(array([x0_r, y0_r], ndmin=2).T)
            data['x_test'].append(array([x1_r, y1_r], ndmin=2).T)
            data['y_test'].append(0.0)
            data['y_test'].append(1.0)
    
    print_message(message='Got XOR dataset: '+str(len(data['x']))+' : '+str(len(data['x_val']))+' : '+str(len(data['x_test']))+', saving...')
    dataset = open_shelve(destination, 'c')
    for key, value in data.items():
        dataset[key] = value
    dataset['class_dist'] = args.class_dist
    dataset.close()
    print_message(message='Dataset dumped as '+destination)