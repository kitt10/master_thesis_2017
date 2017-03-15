#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    scripts.test_mnist
    ~~~~~~~~~~~~~~~~~~

    Testing the MNIST data example on kitt_lib.
"""

from kitt_net import FeedForwardNet
from kitt_monkey import print_message, print_param
from shelve import open as open_shelve

if __name__ == '__main__':
    print_message(message='EXAMPLE: MNIST dataset')

    net = FeedForwardNet(hidden=[15], tf_name='Sigmoid')
    dataset = open_shelve('../examples/mnist/dataset_mnist.ds', 'c')
    net.fit(x=dataset['x'], y=dataset['y'], x_val=dataset['x_val'], y_val=dataset['y_val'], learning_rate=0.3, n_epoch=5, batch_size=20)
    res = net.evaluate(x=dataset['x_test'], y=dataset['y_test'])
    print_message(message='Evaluation on test data after training:')
    print_param(description='Accuracy:', param_str=str(res[1]))
    print_param(description='Error:', param_str=str(res[0]))

    net.prune(req_acc=0.9, req_err=0.05, n_epoch=5)
    res = net.evaluate(x=dataset['x_test'], y=dataset['y_test'])
    print_message(message='Evaluation on test data after pruning:')
    print_param(description='Accuracy:', param_str=str(res[1]))
    print_param(description='Error:', param_str=str(res[0]))
    dataset.close()