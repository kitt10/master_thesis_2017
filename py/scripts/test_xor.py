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
from shelve import open as open_shelve

if __name__ == '__main__':
    n_obs = 2
    n_hidden = 50
    req_acc = 0.99
    pruning_levels = (75, 50, 35, 20, 10, 5, 1, 0)

    print_message(message='EXAMPLE: XOR dataset')
    print_param(description='Number of experiment observations', param_str=str(n_obs))
    print_param(description='Initial number of hidden neurons', param_str=str(n_hidden))
    print_param(description='Required accuracy', param_str=str(req_acc))
    print_param(description='Pruning levels', param_str=str(pruning_levels))

    stats = list()
    for i_obs in range(1, n_obs+1):
        print_message(message='XOR experiment, observation '+str(i_obs)+'/'+str(n_obs))
        net = FeedForwardNet(hidden=[n_hidden], tf_name='Sigmoid')
        dataset = open_shelve('../examples/xor/dataset_xor.ds', 'c')
        net.fit(x=dataset['x'], y=dataset['y'], x_val=dataset['x_val'], y_val=dataset['y_val'], learning_rate=0.3, n_epoch=10)
        res = net.evaluate(x=dataset['x_test'], y=dataset['y_test'])
        print_message(message='Evaluation on test data after training:')
        print_param(description='Accuracy', param_str=str(res[1]))
        print_param(description='Error', param_str=str(res[0]))

        net.prune(req_acc=req_acc, req_err=0.05, n_epoch=50, levels=pruning_levels)
        res = net.evaluate(x=dataset['x_test'], y=dataset['y_test'])
        print_message(message='Evaluation on test data after pruning:')
        print_param(description='Accuracy', param_str=str(res[1]))
        print_param(description='Error', param_str=str(res[0]))
        stats.append(net.opt['pruning'].stats)
        dataset.close()

    analyzer = PruningAnalyzer(stats=stats)
    stats_file_name = 'experiment_xor_nh'+str(n_hidden)+'_ra'+str(req_acc).replace('.', '')+'_no'+str(n_obs)+'.stats'
    analyzer.dump_stats(file_name=stats_file_name)