#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    examples.xor.create_dataset_speech
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates a SPEECH dataset for kitt_lib framework.

    @arg feature_filename   : Path to the file with features
    @arg alignment_filename : Path to the file with alignments
    @arg border_size        : Strictness for splitting individual phonems
    @arg context_size       : size of phonem context (influences sample length)
    @arg n_filters          : number of mel filters to be used
    @arg n_records          : number of records
    @arg name_appendix      : appendix to the destination name
"""

from kitt_monkey import print_message, print_param
from argparse import ArgumentParser
from sys import stderr, maxint
from os.path import basename
from re import search as re_search
from h5py import File as h5py_file
from random import shuffle
from shelve import open as open_shelve
from numpy import array, arange


def parse_arguments():  
    parser = ArgumentParser(description='Creates a SPEECH dataset for kitt_lib.')
    parser.add_argument('-ffn', '--feature_filename', type=str, default='../../../data/data_speech/log_fb_en_25_10_ham_norm.hdf5',
                        help='Path to the file with features')
    parser.add_argument('-afn', '--alignment_filename', type=str, default='../../../data/data_speech/data_aligned_phones.txt',
                        help='Path to the file with alignments')                        
    parser.add_argument('-bs', '--border_size', type=int, default=0,
                        help='Strict?')
    parser.add_argument('-cs', '--context_size', type=int, default=5,
                        help='Use context?')
    parser.add_argument('-nf', '--n_filters', type=int, default=40,
                        help='Number of MEL filters to be used')
    parser.add_argument('-nr', '--n_records', type=int, default=15000,
                        help='Number of records')
    parser.add_argument('-ns', '--n_samples', type=int, default=maxint,
                        help='Number of samples per class (phonem)')
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
                key = basename(line.strip().strip('"')).replace('.lab', '.wav')
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
                    if key not in mlf.keys():
                        mlf[key] = list()

                    for i in range(t_out-t_in):
                        if i >= args.border_size and i < (t_out-t_in-args.border_size):
                            mlf[key].append(phonem)
                        else:
                            mlf[key].append('%%')

def read_features():
    with h5py_file(args.feature_filename, 'r') as f:
        for file_key in f.keys():
            features[file_key] = f[file_key].value

def append_data(key, data_group):
    last_idx = len(features[key])-1
    for i_phonem, phonem in enumerate(mlf[key]):
        if phonem == '%%':
            continue

        data['y'+data_group].append(phonem)
        sample = list()
        for i_context in range(1+2*args.context_size):
            idx = i_phonem-args.context_size+i_context
            if idx < 0:
                sample.extend(features[key][0][:args.n_filters])
            elif idx > last_idx:
                sample.extend(features[key][last_idx][:args.n_filters])
            else:
                sample.extend(features[key][idx][:args.n_filters])

        data['x'+data_group].append(array(sample, ndmin=2).T)
        data['record_keys'+data_group].append(key)

def split_data():
    split_bounds = (args.n_records*args.data_split[0], args.n_records*(args.data_split[0]+args.data_split[1]))
    mlf_keys = mlf.keys()
    shuffle(mlf_keys) 
    for i_key, key in enumerate(mlf_keys):
        if i_key == args.n_records:
            break
        if i_key < split_bounds[0]:
            append_data(key=key, data_group='')
        elif split_bounds[0] <= i_key < split_bounds[1]:
            append_data(key=key, data_group='_val')
        else:
            append_data(key=key, data_group='_test')
            
def get_speech_data():
    print_message(message='Reading alignments...')
    read_alignments()
    print_param(description='Number of loaded records (alignments)', param_str=str(len(mlf.keys())))
    print_param(description='Number of alignment frames', param_str=str(len(mlf[mlf.keys()[0]])))
    print_param(description='Number of found phonems', param_str=str(len(phonems)))
    print_param(description='Found phonems', param_str=str(phonems))
    
    print_message(message='Reading features...')
    read_features()
    print_param(description='Number of loaded records (features)', param_str=str(len(features.keys())))
    print_param(description='Number of feature frames', param_str=str(len(features[features.keys()[0]])))
    
    print_message(message='Splitting data...')
    split_data()
    print_param(description='Number of training samples', param_str=str(len(data['x'])))
    print_param(description='Number of validation samples', param_str=str(len(data['x_val'])))
    print_param(description='Number of testing samples', param_str=str(len(data['x_test'])))
    print_param(description='Problem dimension', param_str=str(data['x'][0].shape[0]))
    print_param(description='Number of classes', param_str=str(len(phonems)))

if __name__ == '__main__':
    args = parse_arguments()
    destination = 'dataset_speech_bs'+str(args.border_size)
    destination += '_cs'+str(args.context_size)+'_nf'+str(args.n_filters)
    destination += '_ds'+str(args.data_split[0]*10)+str(args.data_split[1]*10)+str(args.data_split[2]*10)
    if args.n_samples != maxint:
        destination += '_ns'+str(args.n_samples)
    destination += '_nr'+str(args.n_records)+args.name_appendix+'.ds'

    print_message(message='Processing SPEECH data...')
    print_param(description='Path to features', param_str=args.feature_filename)
    print_param(description='Path to alignments', param_str=args.alignment_filename)
    print_param(description='Border size (strictness)', param_str=str(args.border_size))
    print_param(description='Context size', param_str=str(args.context_size))
    print_param(description='Number of MEL filters', param_str=str(args.n_filters))
    print_param(description='Number of records', param_str=str(args.n_records))
    print_param(description='Data split (train/val/test)', param_str=str(args.data_split))
    print_param(description='Dataset destination file name', param_str=destination)
    
    mlf = dict()
    phonems = list()
    features = dict()
    data = {'x': list(), 'y': list(), 'x_val': list(), 'y_val': list(), 'x_test': list(), 'y_test': list(), 
            'record_keys': list(), 'record_keys_val': list(), 'record_keys_test': list()}
    get_speech_data()

    print_message(message='Saving dataset...')
    dataset = open_shelve(destination, 'c', protocol=2)
    for key, value in data.items():
        dataset[key] = value
    dataset['features'] = args.feature_filename
    dataset['alignments'] = args.alignment_filename
    dataset['border_size'] = str(args.border_size)
    dataset['context_size'] = str(args.context_size)
    dataset['n_filters'] = str(args.n_filters)
    dataset['n_records'] = str(args.n_records)
    dataset.close()
    print_message(message='Dataset dumped as '+destination)
