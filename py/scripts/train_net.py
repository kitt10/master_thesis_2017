#!/usr/bin/env python
# -*- coding: utf-8 -*-

from kitt_net import FeedForwardNet
from shelve import open as open_shelve
import numpy as np
np.set_printoptions(threshold=np.nan)


if __name__ == '__main__':
    net = FeedForwardNet(hidden=[50], tf_name='Sigmoid')
    dataset = open_shelve('../examples/speech/dataset_speech_5000.ds')
    net.fit(x=dataset['x'], y=dataset['y'], x_val=dataset['x_val'], y_val=dataset['y_val'],
            learning_rate=0.01, n_epoch=400, req_acc=1.0, batch_size=10)
    print 'Net structure to be dumped:', net.structure, '| Number of synapses:', net.count_synapses()
    net.dump('../examples/speech/net_speech_5000.net')
    net.prune(req_acc=0.6, req_err=0.05, n_epoch=5, levels=(75, 50, 30, 20, 10, 7, 5, 3, 2, 1, 0))
    print 'Net structure to be dumped:', net.structure, '| Number of synapses:', net.count_synapses()
    net.dump('../examples/speech/net_speech_5000_pruned.net')
    dataset.close()