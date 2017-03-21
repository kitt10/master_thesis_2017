# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_stats
    ~~~~~~~~~~~~~~~~~~~
    Elaboration of all statistics and plotting.
"""

from kitt_monkey import print_message
from cPickle import dump as dump_cpickle, load as load_cpickle
from numpy import mean, std, var
from matplotlib import pyplot as plt, rcParams as mpl_params, patches as mpl_patches

mpl_params['axes.labelsize'] = 18
mpl_params['xtick.labelsize'] = 15
mpl_params['ytick.labelsize'] = 15
mpl_params['legend.fontsize'] = 18

class PruningAnalyzer(object):

    def __init__(self, stats_data):
        self.stats_data = stats_data
        for obs in range(len(self.stats_data)):
            self.stats_data[obs]['n_synapses_total'] = [sum(w_b) for w_b in self.stats_data[obs]['n_synapses']]
        self.means = dict()
        self.vars = dict()
        self.stds = dict()
        self.n_obs = len(self.stats_data)
        self.pruning_steps = range(max([len(obs['acc']) for obs in self.stats_data]))

    def analyze(self):
        for key in self.stats_data[0].keys():
            for obs in self.stats_data:
                while len(obs[key]) < self.pruning_steps[-1]:
                    obs[key].append(obs[key][-1])
            
            print key, [obs[key] for obs in self.stats_data]
            if key == 'structure' or key == 'n_synapses':
                self.means[key] = [mean([obs[key][i_layer] for obs in self.stats_data], axis=0) for i_layer in range(len(self.stats_data[0][key]))]
                self.stds[key] = [std([obs[key][i_layer] for obs in self.stats_data], axis=0) for i_layer in range(len(self.stats_data[0][key]))]
                self.vars[key] = [var([obs[key][i_layer] for obs in self.stats_data], axis=0) for i_layer in range(len(self.stats_data[0][key]))]
            else:
                self.means[key] = mean([obs[key] for obs in self.stats_data], axis=0)
                self.stds[key] = std([obs[key] for obs in self.stats_data], axis=0)
                self.vars[key] = var([obs[key] for obs in self.stats_data], axis=0)

    def dump_stats(self, file_name):
        stats_pack = {'data': self.stats_data, 'means': self.means, 'stds': self.stds, 'vars': self.vars, 'n_obs': self.n_obs, 'pruning_steps': self.pruning_steps}
        with open(file_name, 'w') as f:
            dump_cpickle(stats_pack, f)
        print_message(message='Experiment statistics dumped as '+file_name)
    
    def load_stats(self, file_name):
        print_message(message='Loading pruning process statistics from '+file_name)
        with open(file_name, 'r') as f:
            stats_pack = load_cpickle(f)
        self.stats_data = stats_pack['data']
        self.means = stats_pack['means']
        self.stds = stats_pack['stds']
        self.vars = stats_pack['vars']
        self.n_obs = stats_pack['n_obs']
        self.pruning_steps = stats_pack['pruning_steps']

    def plot_pruning_stats(self, show_fig=True, savefig_name=None):
        _, ax1 = plt.subplots()
        ax1.errorbar(x=self.pruning_steps, y=self.means['acc'], yerr=self.stds['acc'], color='darkgreen')
        ax1.set_xlabel('pruning step')
        ax1.set_ylabel('classification accuracy', color='darkgreen')
        ax1.set_ylim([0, 1.1])
        for tl in ax1.get_yticklabels():
            tl.set_color('darkgreen')

        ax2 = ax1.twinx()
        ax2.errorbar(x=self.pruning_steps, y=self.means['n_synapses_total'], yerr=self.stds['n_synapses_total'], color='maroon')
        ax2.set_ylabel('number of synapses', color='maroon')
        ax2.set_ylim([0, max(self.means['n_synapses_total'])])
        for tl in ax2.get_yticklabels():
            tl.set_color('maroon')

        blue_patch = mpl_patches.Patch(color='darkblue', label='Net structure')
        plt.legend([blue_patch], [p.get_label() for p in [blue_patch]], loc='center right')

        for step in self.pruning_steps:
            plt.annotate(str(self.means['structure'][step]).replace(' ', ''), xy=(step, self.means['n_synapses_total'][step]+1),
                        horizontalalignment='center', verticalalignment='center', fontsize=13, color='darkblue')
        plt.grid()
        plt.xlim([-1, len(self.pruning_steps)+1])
        if savefig_name:
            plt.savefig(savefig_name, bbox_inches='tight', pad_inches=0.1)
        if show_fig:
            plt.show()