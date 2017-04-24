#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.test_speech
    ~~~~~~~~~~~~~~~~~~~

    Testing the SPEECH data example on kitt_lib.
"""

from kitt_net import FeedForwardNet
from kitt_monkey import print_message, print_param
from kitt_stats import PruningAnalyzer, FeatureAnalyzer
from argparse import ArgumentParser
from shelve import open as open_shelve


def parse_arguments():  
    parser = ArgumentParser(description='Run experiments and plot results for XOR dataset.')
    parser.add_argument('-g', '--generate', type=bool, default=False,
                        help='Generate new stats or load the dumped?')
    parser.add_argument('-no', '--n_obs', type=int, default=1,
                        help='Number of experiment observations')
    parser.add_argument('-hs', '--hidden_structure', type=int, default=[100], nargs='+',
                        help='Neural network structure')
    parser.add_argument('-ra', '--req_acc', type=float, default=0.6,
                        help='Required classificationa accuracy')
    parser.add_argument('-lev', '--levels', type=int, default=(75, 50, 35, 20, 10, 5, 1, 0), nargs='+',
                        help='Pruning percentile levels')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print_message(message='EXAMPLE: SPEECH dataset')
    print_param(description='Number of experiment observations', param_str=str(args.n_obs))
    print_param(description='Initial number of hidden neurons', param_str=str(args.hidden_structure))
    print_param(description='Required accuracy', param_str=str(args.req_acc))

    params_str = '_hs'+str(args.hidden_structure)+'_ra'+str(args.req_acc).replace('.', '')+'_no'+str(args.n_obs)
    if args.generate:
        stats_data = list()
        for i_obs in range(1, args.n_obs+1):
            print_message(message='SPEECH experiment, observation '+str(i_obs)+'/'+str(args.n_obs))
            net = FeedForwardNet(hidden=args.hidden_structure, tf_name='Sigmoid')
            dataset = open_shelve('../examples/speech/dataset_speech_bs2_cs5_nf40_ds811_nr200.ds', 'c')
            net.fit(x=dataset['x'], y=dataset['y'], x_val=dataset['x_val'], y_val=dataset['y_val'], learning_rate=0.07, n_epoch=50, req_acc=0.7, batch_size=10)
            res = net.evaluate(x=dataset['x_test'], y=dataset['y_test'])
            print_message(message='Evaluation on test data after training:')
            print_param(description='Accuracy', param_str=str(res[1]))
            print_param(description='Error', param_str=str(res[0]))
            if net.learning.stats['t_acc'][-1] < args.req_acc:
                print 'Skipping observation'
                continue
            net.prune(req_acc=args.req_acc, req_err=0.05, n_epoch=5, levels=args.levels)
            #res = net.evaluate(x=dataset['x_test'], y=dataset['y_test'])
            #print_message(message='Evaluation on test data after pruning:')
            #print_param(description='Accuracy', param_str=str(res[1]))
            #print_param(description='Error', param_str=str(res[0]))
            stats_data.append(net.opt['pruning'].stats)
            net.dump('../examples/speech/net_speech'+params_str+'_obs'+str(i_obs)+'_pruned.net')
            dataset.close()

        analyzer = PruningAnalyzer(stats_data=stats_data)
        analyzer.analyze()
        analyzer.dump_stats(file_name='../examples/speech/experiment_speech'+params_str+'.stats')
    else:
        #analyzer = PruningAnalyzer(stats_data=[])
        #analyzer.load_stats(file_name='../examples/speech/experiment_speech'+params_str+'.stats')
        pass

    #analyzer.plot_pruning_process(req_acc=args.req_acc, pruning_steps=range(20)+range(113, 117))

    net = FeedForwardNet(hidden=args.hidden_structure, tf_name='Sigmoid')
    # net.load('../examples/mnist/net_mnist_hs[20]_ra05_no1_obs1_pruned.net')
    net.load('../examples/speech/net_simple_cs10_pruned.net')
    net.n_features_init = 840
    net.opt['feature_energy'].find_paths()
    net.opt['feature_energy'].compute_energies()

    f_analyzer = FeatureAnalyzer(net=net)
    f_analyzer.plot_feature_energy()