#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.test_xor
    ~~~~~~~~~~~~~~~~

    Testing the XOR data example on kitt_lib.
"""

from kitt_net import FeedForwardNet
from kitt_monkey import print_message, print_param
from kitt_stats import PruningAnalyzer
from argparse import ArgumentParser
from shelve import open as open_shelve
from numpy import sum as np_sum
from cPickle import load as load_cpickle


def parse_arguments():  
    parser = ArgumentParser(description='Run experiments and plot results for XOR dataset.')
    parser.add_argument('-g', '--generate', type=bool, default=False,
                        help='Generate new stats or load the dumped?')
    parser.add_argument('-no', '--n_obs', type=int, default=10,
                        help='Number of experiment observations')
    parser.add_argument('-hs', '--hidden_structure', type=int, default=[50], nargs='+',
                        help='Neural network structure')
    parser.add_argument('-ra', '--req_acc', type=float, default=1.0,
                        help='Required classificationa accuracy')
    parser.add_argument('-lev', '--levels', type=int, default=(75, 50, 35, 20, 10, 5, 1, 0), nargs='+',
                        help='Pruning percentile levels')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print_message(message='EXAMPLE: XOR dataset')
    print_param(description='Number of experiment observations', param_str=str(args.n_obs))
    print_param(description='Initial number of hidden neurons', param_str=str(args.hidden_structure))
    print_param(description='Required accuracy', param_str=str(args.req_acc))

    params_str = '_hs'+str(args.hidden_structure)+'_ra'+str(args.req_acc).replace('.', '')+'_no'+str(args.n_obs)
    if args.generate:
        stats_data = list()
        for i_obs in range(1, args.n_obs+1):
            print_message(message='XOR experiment, observation '+str(i_obs)+'/'+str(args.n_obs))
            net = FeedForwardNet(hidden=args.hidden_structure, tf_name='Sigmoid')
            dataset = open_shelve('../examples/xor/dataset_xor.ds', 'c')
            net.fit(x=dataset['x'], y=dataset['y'], x_val=dataset['x_val'], y_val=dataset['y_val'], learning_rate=0.4,
                    n_epoch=50, req_acc=1.0)
            res = net.evaluate(x=dataset['x_test'], y=dataset['y_test'])
            print_message(message='Evaluation on test data after training:')
            print_param(description='Accuracy', param_str=str(res[1]))
            print_param(description='Error', param_str=str(res[0]))
            if net.learning.stats['t_acc'][-1] < 0.9:
                print 'Skipping observation'
                continue
            net.prune(req_acc=args.req_acc, req_err=0.05, n_epoch=50, levels=args.levels)
            res = net.evaluate(x=dataset['x_test'], y=dataset['y_test'])
            print_message(message='Evaluation on test data after pruning:')
            print_param(description='Accuracy', param_str=str(res[1]))
            print_param(description='Error', param_str=str(res[0]))
            stats_data.append(net.opt['pruning'].stats)
            net.dump('../examples/xor/net_xor'+params_str+'_obs'+str(i_obs)+'_pruned.net')
            dataset.close()

        analyzer = PruningAnalyzer(stats_data=stats_data)
        #analyzer.analyze()
        analyzer.dump_stats(file_name='../examples/xor/experiment_xor'+params_str+'.stats')
    else:
        analyzer = PruningAnalyzer(stats_data=[])
        analyzer.load_stats(file_name='../examples/xor/experiment_xor'+params_str+'.stats')

    #analyzer.analyze()
    #analyzer.plot_pruning_process(req_acc=args.req_acc)
    #analyzer.plot_pruning_result_pie()
    '''
    for obs in range(10):
        with open('../examples/xor/net_xor_hs[50]_ra10_no10_obs'+str(obs+1)+'_pruned.net', 'r') as f:
            net_pack = load_cpickle(f)
        print net_pack['w_is']
        print '----------'
    '''
    exit()
    analyzer.new_stats_data['n_synapses_layers'] = list()
    for i_obs in range(10):
        if i_obs == 8:
            continue
        try:
            net = FeedForwardNet(hidden=args.hidden_structure, tf_name='Sigmoid')
            net.load('../examples/xor/net_xor'+params_str+'_obs'+str(i_obs)+'_pruned.net')
            analyzer.new_stats_data['n_synapses_layers'].append([np_sum(w_i) for w_i in net.w_is])
            print analyzer.new_stats_data['n_synapses_layers']
        except:
            pass
    analyzer.plot_pruning_results()