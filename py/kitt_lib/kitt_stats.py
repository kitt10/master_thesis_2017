# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_stats
    ~~~~~~~~~~~~~~~~~~~
    Elaboration of all statistics and plotting.
"""

from kitt_monkey import print_message
from cPickle import dump as dump_cpickle, load as load_cpickle
from numpy import mean, std, var, unique, array
from matplotlib import pyplot as plt, rcParams as mpl_params, patches as mpl_patches

mpl_params['axes.labelsize'] = 32
mpl_params['xtick.labelsize'] = 32
mpl_params['ytick.labelsize'] = 25
mpl_params['legend.fontsize'] = 30


class PruningAnalyzer(object):

    def __init__(self, stats_data):
        self.stats_data = stats_data
        for obs in range(len(self.stats_data)):
            #self.stats_data[obs]['n_synapses_total'] = [sum(w_b) for w_b in self.stats_data[obs]['n_synapses']]
            self.stats_data[obs]['n_synapses_total'] = [w_b[0] for w_b in self.stats_data[obs]['n_synapses']]
        self.means = dict()
        self.vars = dict()
        self.stds = dict()
        self.n_obs = len(self.stats_data)
        self.new_stats_data = dict()
        try:
            self.pruning_steps = range(max([len(obs['n_synapses']) for obs in self.stats_data]))
        except ValueError:
            self.pruning_steps = None

    def analyze(self):
        self.stats_data = [self.stats_data[7]]
        for key in self.stats_data[0].keys():
            for i_obs in range(len(self.stats_data)):
                tmp = self.stats_data[i_obs][key][-1]
                while len(self.stats_data[i_obs][key]) <= self.pruning_steps[-1]:
                    self.stats_data[i_obs][key].append(tmp)
            
            if key == 'structure' or key == 'n_synapses':
                self.means[key] = [mean([obs[key][i_layer] for obs in self.stats_data], axis=0) for i_layer in range(len(self.stats_data[0][key]))]
                self.stds[key] = [std([obs[key][i_layer] for obs in self.stats_data], axis=0) for i_layer in range(len(self.stats_data[0][key]))]
                self.vars[key] = [var([obs[key][i_layer] for obs in self.stats_data], axis=0) for i_layer in range(len(self.stats_data[0][key]))]
            elif key == 'th':
                continue
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

    def plot_pruning_process(self, req_acc=1.0, show_fig=True, savefig_name=None, pruning_steps=None):
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        #ax1.errorbar(x=self.pruning_steps, y=self.means['acc'], yerr=self.stds['acc'], color='darkgreen')
        if pruning_steps:
            self.pruning_steps = pruning_steps
        for i_step, step in enumerate(self.pruning_steps):
            #alpha = 0.8 if self.means['retrained'][step] > 0.5 else 0.2
            color1 = 'maroon' if self.means['retrained'][step] > 0.5 else '#e5cccc'
            color2 = 'darkgreen' if self.means['retrained'][step] > 0.5 else '#cce0cc'
            ax1.bar(i_step-0.3, width=0.3, height=self.means['n_synapses_total'][step], color=color1, alpha=1.0)
            ax2.bar(i_step, width=0.3, height=self.means['acc'][step], color=color2, alpha=1.0)
        
        ax1.set_xlabel('pruning step')
        ax1.set_ylabel('number of synapses', color='maroon')
        ax1.set_ylim([0, self.means['n_synapses_total'][0]+0.1*self.means['n_synapses_total'][0]])
        for tl in ax1.get_yticklabels():
            tl.set_color('maroon')

        #ax2.errorbar(x=self.pruning_steps, y=self.means['n_synapses_total'], yerr=self.stds['n_synapses_total'], color='maroon')
        ax2.set_ylabel('classification accuracy', color='darkgreen')
        dashed_line = ax2.plot([-1, self.pruning_steps[-1]+1], [req_acc]*2, 'g--', label='required accuracy')[0]
        ax2.set_ylim([0, 1.1])
        for tl in ax2.get_yticklabels():
            tl.set_color('darkgreen')
        
        blue_patch = mpl_patches.Patch(color='darkblue', label='net structure')
        plt.legend([blue_patch, dashed_line], [p.get_label() for p in [blue_patch, dashed_line]], loc='upper right')

        for i_step, step in enumerate(self.pruning_steps):
            if self.means['retrained'][step] > 0.5:
                plt.annotate(str([int(n) for n in self.means['structure'][step].tolist()]).replace(',', '-').replace(' ', '')[1:-1], xy=(i_step-0.1, 0.5),
                            horizontalalignment='center', verticalalignment='center', fontsize=25, color='white', rotation=75, backgroundcolor='darkblue')
            #alpha = 0.8 if self.means['retrained'][step] > 0.5 else 0.2
            color1 = 'maroon' if self.means['retrained'][step] > 0.5 else '#e5cccc'
            plt.annotate(int(self.means['n_synapses_total'][step]), xy=(i_step-0.3, self.means['n_synapses_total'][step]/self.means['n_synapses_total'][0]+0.03),
                        horizontalalignment='center', verticalalignment='center', fontsize=25, color=color1, alpha=1.0)

        plt.grid()
        plt.xlim([-1, len(self.pruning_steps)])
        plt.xticks(range(len(self.pruning_steps)), self.pruning_steps)
        if savefig_name:
            plt.savefig(savefig_name, bbox_inches='tight', pad_inches=0.1)
        if show_fig:
            plt.show()

    def plot_pruning_result_pie(self):
        labels = 'solution A (2-2-2)', 'solution B (2-3-2)', 'other'
        n_222 = 0
        ns_222 = list()
        n_232 = 0
        ns_232 = list()
        n_other = 0
        ns_other = list()
        for obs in self.stats_data:
            ind = max([i for i, r in enumerate(obs['retrained']) if r])
            if obs['structure'][ind] == [2, 2, 2]:
                n_222 += 1
                ns_222.append(obs['n_synapses'][ind][0])
            elif obs['structure'][ind] == [2, 3, 2]:
                n_232 += 1
                ns_232.append(obs['n_synapses'][ind][0])
            else:
                n_other += 1
                ns_other.append(obs['n_synapses'][ind][0])
        sizes = [n_222, n_232, n_other]
        colors = ['darkblue', 'darkgreen', 'maroon']
        explode = (0, 0, 0.1)
        patches, _, texts = plt.pie(sizes, explode=explode, colors=colors, shadow=True, startangle=140,
                            autopct = '%1.0f%%')
        texts[0].set_fontsize(35)
        texts[1].set_fontsize(35)
        texts[2].set_fontsize(35)
        plt.legend(patches, labels, loc="lower right")
        plt.axis('equal')
        plt.show()

        medianprops = dict(linestyle='-', linewidth=3, color='black')
        meanpointprops = dict(marker='D', markeredgecolor='black',
                              markerfacecolor='black')
        boxes = plt.boxplot([ns_222, ns_232, ns_other], widths=[0.75]*3, positions=range(3), patch_artist=True, medianprops=medianprops,
                            meanprops=meanpointprops, showmeans=True)
        xlabs = plt.gca().get_xticklabels()
        for box, xlab, color in zip(boxes['boxes'], xlabs, colors):
            box.set_facecolor(color)
            box.set_color(color)
            xlab.set_color(color)

        plt.xlim([-0.5, 2.5])
        plt.xticks(range(3), labels, rotation=-15)
        plt.xlabel('structure after pruning')
        plt.ylim([4, 15])
        plt.ylabel('number of synapses')
        plt.grid()
        plt.show()

        
    def plot_pruning_results(self, show_fig=True, savefig_name=None, max_synapses_in_layer=0):
        init_structure = self.means['structure'][0]
        init_synapses_l = [l1*l2 for l1, l2 in zip(init_structure[:-1], init_structure[1:])]
        
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        # input layer
        plt.bar(-0.3, width=0.6, height=max(init_synapses_l)+0.1*max(init_synapses_l), color='whitesmoke', alpha=0.3)
        ax1.bar(-0.25, width=0.5, height=init_structure[0], color='darkblue', alpha=0.3)
        ax1.boxplot([d[0] for d in self.means['structure']], widths=[0.5], positions=[0])

        # hidden layers
        for i_layer in range(len(init_structure[1:-1])):
            plt.bar(i_layer+1-0.45, width=0.9, height=max(init_synapses_l)+0.1*max(init_synapses_l), color='whitesmoke', alpha=0.3)
            ax2.bar(i_layer+1-0.45, width=0.4, height=init_synapses_l[i_layer+1], color='maroon', alpha=0.3)
            ax2.boxplot([d[i_layer+1] for d in self.new_stats_data['n_synapses_layers']], widths=[0.4], positions=[i_layer+1-0.25])
            ax1.bar(i_layer+1+0.05, width=0.4, height=init_structure[i_layer+1], color='darkblue', alpha=0.3)
            ax1.boxplot([d[i_layer+1] for d in self.means['structure']], widths=[0.4], positions=[i_layer+1+0.25])
            
        # output layer
        plt.bar(len(init_structure)-1-0.45, width=0.9, height=max(init_synapses_l)+0.1*max(init_synapses_l), color='whitesmoke', alpha=0.5)
        ax2.bar(len(init_structure)-1-0.45, width=0.4, height=init_synapses_l[0], color='maroon', alpha=0.3)
        ax2.boxplot([d[-1] for d in self.new_stats_data['n_synapses_layers']], widths=[0.4], positions=[len(init_structure)-1-0.25])
        ax1.bar(len(init_structure)-1+0.05, width=0.4, height=init_structure[-1], color='darkblue', alpha=0.3)
        ax1.boxplot([d[-1] for d in self.means['structure']], widths=[0.4], positions=[len(init_structure)-1+0.25])

        ax1.set_xlabel('network layers')
        ax1.set_ylabel('number of neurons', color='darkblue')
        ax1.set_ylim([0, max(self.means['structure'][0])+0.1*max(self.means['structure'][0])])
        for tl in ax1.get_yticklabels():
            tl.set_color('darkblue')

        #ax2.errorbar(x=self.pruning_steps, y=self.means['n_synapses_total'], yerr=self.stds['n_synapses_total'], color='maroon')
        ax2.set_ylabel('number of synapses', color='maroon')
        ax2.set_ylim([0, max(init_synapses_l)+0.1*max(init_synapses_l)])
        for tl in ax2.get_yticklabels():
            tl.set_color('maroon')
        plt.xlim([-1, len(init_structure)])
        plt.xticks(range(len(init_structure)), ['I']+['H'+str(k+1) for k in range(len(init_structure[1:-1]))]+['O'])
        plt.grid()
        plt.show()


class FeatureAnalyzer(object):

    def __init__(self, net):
        self.net = net
        self.fe = self.net.opt['feature_energy']
    
    def plot_feature_energy(self):
        '''
        mat = list()
        fe_features = [f_i for f_i in sorted(self.fe.energies.keys()) if abs(self.fe.energies[f_i]['total'])>30]
        fe_classes = self.net.labels+['total']
        for label in fe_classes:
            mat.append([self.fe.energies[f_i][label] for f_i in fe_features])
        plt.imshow(mat, interpolation='nearest', aspect='auto')
        plt.yticks(range(len(fe_classes)), fe_classes)
        plt.xticks(range(len(fe_features)), fe_features, rotation=90)
        plt.ylabel('classes')
        plt.xlabel('features')
        plt.colorbar()
        plt.grid()
        plt.show()
        
        for digit in range(10)+['total']:
            mat = list()
            for row in range(28):
                mat.append([self.fe.energies[f_i][digit] for f_i in sorted(self.fe.energies.keys())[row*28:row*28+28]])
            
            plt.imshow(mat, interpolation='nearest', aspect='auto', vmin=-10, vmax=10, cmap='gray')
            plt.title(digit)
            plt.colorbar()
            plt.grid()
            plt.show()
        '''

        mat = list()
        counter = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
        for row_i in range(28):
            col = list()
            for f_i in sorted(self.fe.energies.keys())[row_i*28:row_i*28+28]:
                classes_found = list()
                for digit in range(10):
                    if abs(self.fe.energies[f_i][digit]) > 0:
                        classes_found.append(digit)
                if len(classes_found) == 1:
                    col.append(classes_found[0])
                elif len(classes_found) == 0:
                    col.append(-2)
                else:
                    col.append(-1)
                counter[len(classes_found)] +=1
                
            mat.append(col)
        print counter
        cmap = plt.get_cmap('jet', 12)
        cmap.set_under('gray')
        plt.imshow(mat, interpolation='none', aspect='auto', vmin=-2, vmax=10, cmap=cmap)
        colorbar = plt.colorbar()
        colorbar.set_ticks([i+0.5 for i in range(-2, 10)])
        colorbar.set_ticklabels(['no digit', 'more than one']+['only '+str(i) for i in range(10)])
        plt.grid()
        plt.show()

        hidden = dict()
        for h in range(20):
            hidden[h] = list()
        for f_i in range(self.fe.n_features):
            for c_i, label in enumerate(self.net.labels):
                for path in self.fe.paths[f_i]:
                    if path[-1][0] == c_i:
                        hidden[path[0][0]].append(label)

        for h in range(20):
            print h, len(hidden[h]), unique(hidden[h])
        print '-------------------'
        
        tmp = dict()
        mat = list()
        for label in self.net.labels:
            tmp[label] = list()
            mat.append(list())
            for h in range(20):
                if label in hidden[h]:
                    tmp[label].append(h)
                    mat[-1].append(1)
                else:
                    mat[-1].append(0)
            print label, tmp[label]
        plt.imshow(array(mat).T, interpolation='none', aspect='auto', vmin=0, vmax=1, cmap='gray')
        plt.xticks(range(10))
        plt.xlabel('classes')
        plt.yticks(range(20))
        plt.ylabel('hidden neurons')
        plt.colorbar()
        plt.grid()
        plt.show()