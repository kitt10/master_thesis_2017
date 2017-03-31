#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    examples.xor.create_dataset_speech
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This script creates a SPEECH dataset for kitt_lib framework.

    @:param feature_filename   : Path to the file with features
    @:param alignment_filename : Path to the file with alignments
    @:param border_size        : Strictness for splitting individual phonemes
    @:param context_size       : size of phonem context (influences sample length)
    @:param n_filters          : number of mel filters to be used
    @:param n_samples          : number of samples
    @:param max_rest           : maximum number of other phonemes than selected
    @:param n_records          : number of records
    @:param phonemes           : phonemes as classes
    @:param data_split         : data split ratio
    @:param name_appendix      : appendix to the destination name
"""

from kitt_monkey import print_message, print_param
from argparse import ArgumentParser
from sys import stderr
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
    parser.add_argument('-bs', '--border_size', type=int, default=2,
                        help='Strict?')
    parser.add_argument('-ph', '--phonemes', type=str, default=[], nargs='+',
                        help='Create a dataset for selected phones only')
    parser.add_argument('-cs', '--context_size', type=int, default=5,
                        help='Use context?')
    parser.add_argument('-nf', '--n_filters', type=int, default=40,
                        help='Number of MEL filters to be used')
    parser.add_argument('-nr', '--n_records', type=int, default=15000,
                        help='Number of records')
    parser.add_argument('-ns', '--n_samples', type=int, required=True,
                        help='Number of samples per class (phonem)')
    parser.add_argument('-mr', '--max_rest', type=int, default=1e6,
                        help='Max of rest phonemes (others than selected)')
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

                    if args.phonemes and phonem not in args.phonemes:
                        phonem = '%'

                    if phonem not in samples.keys():
                        samples[phonem] = list()

                    if t_in >= t_out:
                        continue
                    if key not in mlf.keys():
                        mlf[key] = list()

                    for i in range(t_out-t_in):
                        if args.border_size <= i < (t_out-t_in-args.border_size):
                            mlf[key].append(phonem)
                        else:
                            mlf[key].append('%%')


def read_features():
    with h5py_file(args.feature_filename, 'r') as f:
        for file_key in f.keys():
            features[file_key] = f[file_key].value


def add_samples():
    mlf_keys = mlf.keys()
    shuffle(mlf_keys)
    for i_key, key in enumerate(mlf_keys):
        if i_key == args.n_records:
            break

        last_idx = len(features[key])-1
        for i_phonem, phonem in enumerate(mlf[key]):
            if phonem == '%%' or (len(samples[phonem]) == args.n_samples and phonem != '%') \
                    or (phonem == '%' and len(samples['%']) == args.max_rest):
                continue

            sample = list()
            for i_context in range(1+2*args.context_size):
                idx = i_phonem-args.context_size+i_context
                if idx < 0:
                    sample.extend(features[key][0][:args.n_filters])
                elif idx > last_idx:
                    sample.extend(features[key][last_idx][:args.n_filters])
                else:
                    sample.extend(features[key][idx][:args.n_filters])

            samples[phonem].append(sample)


def split_data():
    for phonem, sample_list in samples.iteritems():
        split_bounds = (len(sample_list)*args.data_split[0], len(sample_list)*(args.data_split[0]+args.data_split[1]))
        for i_sample, sample in enumerate(sample_list):
            if i_sample < split_bounds[0] or (phonem == '%' and len(samples['%']) < args.data_split[0]*args.max_rest):
                append_sample(sample=sample, target=phonem, data_group='')          # add to training set
            elif split_bounds[0] <= i_sample < split_bounds[1] and phonem != '%':
                append_sample(sample=sample, target=phonem, data_group='_val')      # add to validation set
            else:
                append_sample(sample=sample, target=phonem, data_group='_test')     # add to testing set


def append_sample(sample, target, data_group):
    data['y'+data_group].append(target)
    data['x'+data_group].append(array(sample, ndmin=2).T)
    # data['record_keys'+data_group].append(key)


def get_speech_data():
    print_message(message='Reading alignments...')
    read_alignments()
    print_param(description='Number of loaded records (alignments)', param_str=str(len(mlf.keys())))
    print_param(description='Number of alignment frames', param_str=str(len(mlf[mlf.keys()[0]])))
    print_param(description='Number of found phonemes', param_str=str(len(samples)))
    print_param(description='Found phonemes', param_str=str(sorted(samples.keys())))
    
    print_message(message='Reading features...')
    read_features()
    print_param(description='Number of loaded records (features)', param_str=str(len(features.keys())))
    print_param(description='Number of feature frames', param_str=str(len(features[features.keys()[0]])))

    print_message(message='Adding samples...')
    add_samples()

    print_message(message='Splitting data...')
    split_data()
    print_param(description='Number of training samples', param_str=str(len(data['x'])))
    print_param(description='Number of validation samples', param_str=str(len(data['x_val'])))
    print_param(description='Number of testing samples', param_str=str(len(data['x_test'])))
    print_param(description='Problem dimension', param_str=str(data['x'][0].shape[0]))
    print_param(description='Number of classes', param_str=str(len(samples)))

    print_message(message='Number of samples per class:')
    for phonem in sorted(samples.keys()):
        print_param(description=phonem, param_str=str(len(samples[phonem])))

if __name__ == '__main__':
    args = parse_arguments()
    destination = 'dataset_speech_bs'+str(args.border_size)
    destination += '_cs'+str(args.context_size)+'_nf'+str(args.n_filters)
    destination += '_ds'+str(int(args.data_split[0]*10))+str(int(args.data_split[1]*10))+str(int(args.data_split[2]*10))
    destination += '_ns'+str(args.n_samples)+'_nr'+str(args.n_records)
    if args.phonemes:
        destination += '_'+str(args.phonemes).replace(',','+').replace(' ', '').replace('\'', '')[1:-1]
    destination += args.name_appendix+'.ds'

    print_message(message='Processing SPEECH data...')
    print_param(description='Path to features', param_str=args.feature_filename)
    print_param(description='Path to alignments', param_str=args.alignment_filename)
    print_param(description='Border size (strictness)', param_str=str(args.border_size))
    print_param(description='Context size', param_str=str(args.context_size))
    print_param(description='Number of MEL filters', param_str=str(args.n_filters))
    print_param(description='Number of records', param_str=str(args.n_records))
    print_param(description='Number of samples', param_str=str(args.n_samples))
    print_param(description='Maximum number of other phonemes', param_str=str(args.max_rest))
    print_param(description='Phonemes as classes', param_str=str(args.phonemes) if args.phonemes else 'all')
    print_param(description='Data split (train/val/test)', param_str=str(args.data_split))
    print_param(description='Dataset destination file name', param_str=destination)
    
    mlf = dict()
    features = dict()
    samples = dict()
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
