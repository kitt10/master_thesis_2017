#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    examples.xor.create_dataset_speech
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates a SPEECH dataset for kitt_lib framework.

    @arg n_records          : number of records
    @arg name_appendix      : appendix to the destination name
"""

from kitt_monkey import print_message, print_param
from argparse import ArgumentParser
from sys import stderr
from os.path import basename
from re import search as re_search
from shelve import open as open_shelve
from numpy import array, arange


def parse_arguments():  
    parser = ArgumentParser(description='Creates an XOR dataset for kitt_lib.')
    parser.add_argument('-ffn', '--feature_filename', type=str, default='../../../data/data_speech/log_fb_en_25_10_ham_norm.hdf5',
                        help='Path to the file with features')
    parser.add_argument('-afn', '--alignment_filename', type=str, default='../../../data/data_speech/data_aligned_phones.txt',
                        help='Path to the file with alignments')                        
    parser.add_argument('-bs', '--border_size', type=int, default=0,
                        help='Strict?')
    parser.add_argument('-cs', '--context_size', type=int, default=0,
                        help='Use context?')
    parser.add_argument('-nr', '--n_records', type=int, default=1000,
                        help='Number of records')
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

def read_alignments(htk_time2frames=100000, comment_start=('#', '#!'), encoding='cp1250'):
    with open(args.alignment_filename, 'r') as f:
        for line in f:
            if len(line) == 0 or line.startswith(comment_start):
                continue
            if line[0] == '"':
                name = basename(line.strip().strip('"')).replace('.lab', '.wav')
            elif line[0] != '.':
                line = line.decode('string_escape').decode(encoding).strip()
                #                     cas        cas       slovo      rest
                match = re_search(r'([-\d.]*)\s*([-\d.]*)\s*([^\s]+)\s*(.*?)$', line.strip())
                
                if match:
                    t_in = int(match.group(1).strip())/htk_time2frames
                    t_out = int(match.group(2).strip())/htk_time2frames
                    phonem = match.group(3).strip()  
                    
                    if phonem not in phonems:
                        phonems.append(phonem)
                    if t_in >= t_out:
                        continue
                    if name not in mlf.keys():
                        mlf[name] = list()

                    for i in range(t_out-t_in):
                        if i < args.border_size or i >= (t_out-t_in-args.border_size):
                            mlf[name].append('%%'+phonem)       # TODO - is this needed to store?
                        else:
                            mlf[name].append(phonem)

def read_features():
    print_message(message='Reading features...')
    split_bounds = (args.n_records*args.data_split[0], args.n_records*(args.data_split[0]+args.data_split[1]))

def get_speech_data():
    print_message(message='Reading alignments...')
    read_alignments()
    print_param(description='Number of loaded records', param_str=str(len(mlf.keys())))
    print_param(description='Number of found phonems', param_str=str(len(phonems)))
    print_param(description='Found phonems', param_str=str(phonems))
    #read_features()

if __name__ == '__main__':
    args = parse_arguments()
    destination = 'dataset_speech'+args.name_appendix+'.ds'

    print_message(message='Processing SPEECH data...')
    print_param(description='Path to features', param_str=args.feature_filename)
    print_param(description='Path to alignments', param_str=args.alignment_filename)
    print_param(description='Border size (strictness)', param_str=str(args.border_size))
    print_param(description='Context size', param_str=str(args.context_size))
    print_param(description='Number of records', param_str=str(args.n_records))
    print_param(description='Data split (train/val/test)', param_str=str(args.data_split))
    print_param(description='Dataset destination file name', param_str=destination)
    
    data = {'x': list(), 'y': list(), 'x_val': list(), 'y_val': list(), 'x_test': list(), 'y_test': list()}
    mlf = dict()
    phonems = list()
    get_speech_data()

    print_message(message='Got SPEECH dataset: '+str(len(data['x']))+' : '+str(len(data['x_val']))+' : '+str(len(data['x_test']))+', saving...')
    dataset = open_shelve(destination, 'c')
    for key, value in data.items():
        dataset[key] = value
    dataset['features'] = args.feature_filename
    dataset['alignments'] = args.alignment_filename
    dataset['border_size'] = args.border_size
    dataset['context_size'] = args.context_size
    dataset.close()
    print_message(message='Dataset dumped as '+destination)
