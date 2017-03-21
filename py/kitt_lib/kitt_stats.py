# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_stats
    ~~~~~~~~~~~~~~~~~~~
    Elaboration of all statistics and plotting.
"""

from cPickle import dump as dump_cpickle

class PruningAnalyzer(object):

    def __init__(self, stats):
        self.stats = stats
        self.n_obs = len(self.stats)

    def dump_stats(self, file_name):
        with open(file_name, 'w') as f:
            dump_pickle(self.stats, f)

    def plot_pruning_stats(self):
        pass