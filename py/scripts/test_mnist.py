#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.test_mnist
    ~~~~~~~~~~~~~~~~~~

    Testing the MNIST data example on kitt_lib.
"""

from kitt_net import FeedForwardNet
from kitt_monkey import print_message, print_param
from kitt_stats import PruningAnalyzer, FeatureAnalyzer
from argparse import ArgumentParser
from shelve import open as open_shelve
from numpy import array
from sklearn.metrics import confusion_matrix, accuracy_score

def parse_arguments():  
    parser = ArgumentParser(description='Run experiments and plot results for XOR dataset.')
    parser.add_argument('-g', '--generate', type=bool, default=False,
                        help='Generate new stats or load the dumped?')
    parser.add_argument('-no', '--n_obs', type=int, default=1,
                        help='Number of experiment observations')
    parser.add_argument('-hs', '--hidden_structure', type=int, default=[20], nargs='+',
                        help='Neural network structure')
    parser.add_argument('-ra', '--req_acc', type=float, default=0.9,
                        help='Required classificationa accuracy')
    parser.add_argument('-lev', '--levels', type=int, default=(75, 50, 35, 20, 10, 5, 1, 0), nargs='+',
                        help='Pruning percentile levels')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print_message(message='EXAMPLE: MNIST dataset')
    print_param(description='Number of experiment observations', param_str=str(args.n_obs))
    print_param(description='Initial number of hidden neurons', param_str=str(args.hidden_structure))
    print_param(description='Required accuracy', param_str=str(args.req_acc))

    params_str = '_hs'+str(args.hidden_structure)+'_ra'+str(args.req_acc).replace('.', '')+'_no'+str(args.n_obs)
    if args.generate:
        stats_data = list()
        for i_obs in range(1, args.n_obs+1):
            print_message(message='MNIST experiment, observation '+str(i_obs)+'/'+str(args.n_obs))
            net = FeedForwardNet(hidden=args.hidden_structure, tf_name='Sigmoid')
            dataset = open_shelve('../examples/mnist/dataset_mnist.ds', 'c')
            net.fit(x=dataset['x'], y=dataset['y'], x_val=dataset['x_val'], y_val=dataset['y_val'], learning_rate=0.3, n_epoch=10, req_acc=0.92, batch_size=10)
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
            net.dump('../examples/mnist/net_mnist'+params_str+'_obs'+str(i_obs)+'_pruned.net')
            dataset.close()

        analyzer = PruningAnalyzer(stats_data=stats_data)
        analyzer.analyze()
        analyzer.dump_stats(file_name='../examples/mnist/experiment_mnist'+params_str+'.stats')
    else:
        #analyzer = PruningAnalyzer(stats_data=[])
        #analyzer.load_stats(file_name='../examples/mnist/experiment_mnist'+params_str+'.stats')
        pass

    #analyzer.plot_pruning_process(req_acc=args.req_acc)

    net = FeedForwardNet(hidden=args.hidden_structure, tf_name='Sigmoid')
    net.load('../examples/mnist/net_mnist_hs[20]_ra09_no1_obs1_pruned.net')
    #net.compute_feature_energy()

    #f_analyzer = FeatureAnalyzer(net=net)
    #f_analyzer.plot_feature_energy()

    dataset = open_shelve('../examples/mnist/dataset_mnist.ds', 'c')
    n_to_test = 1000
    x_test = list()
    for sample in dataset['x_test'][:n_to_test]:
        x_test.append([x for x_i, x in enumerate(sample) if x_i in [f[1] for f in net.used_features]])
    res = net.evaluate(x=array(x_test), y=dataset['y_test'][:n_to_test])
    print_message(message='Evaluation on test data after pruning:')
    print_param(description='Accuracy', param_str=str(res[1]))
    print_param(description='Error', param_str=str(res[0]))
    print_message(message='Confusion matrix for test data after pruning:')
    predictions = [net.predict(x)[0][0] for x in x_test]
    print confusion_matrix(y_true=dataset['y_test'][:n_to_test], y_pred=predictions, labels=net.labels)
    print_param(description='Scikit-learn accuracy score', param_str=str(accuracy_score(y_true = dataset['y_test'][:n_to_test], y_pred=predictions)))
    
    net.init_tailoring()
    net.opt['tailoring'].add_neurons(class_label=9, h=3)
    net.learning.net.w = net.w
    net.learning.net.w_is = net.w_is
    net.learning.net.b = net.b
    net.learning.net.b_is = net.b_is
    net.retrain()
    predictions = [net.predict(x)[0][0] for x in x_test]
    print confusion_matrix(y_true=dataset['y_test'][:n_to_test], y_pred=predictions, labels=net.labels)
    print_param(description='Scikit-learn accuracy score', param_str=str(accuracy_score(y_true = dataset['y_test'][:n_to_test], y_pred=predictions)))
    dataset.close()
    
