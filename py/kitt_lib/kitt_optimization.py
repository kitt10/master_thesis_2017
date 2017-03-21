# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_optimization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Optimization methods.
"""

from kitt_monkey import print_pruning_started, print_pruning_step, print_pruning_finished
from numpy import array, percentile, where, hstack, logical_and, inf, delete, sum as np_sum
from time import time

class Pruning(object):

    def __init__(self, kw):
        self.net = kw['self']
        self.kw = kw
        self.vars = {'pruned': False, 'step': 0, 'level_ind': 0, 'level': self.kw['levels'][0], 'net_tmp': self.net.copy_()}
        self.stats = {'structure': list(), 'n_synapses': list(), 'n_to_cut': [0], 'retrained': [True], 
                      'step_time': [0.0], 'acc': [self.net.learning.stats['t_acc'][-1]], 'err': [self.net.learning.stats['t_err'][-1]]}
        self.vars['net_tmp'].learning.kw['req_acc'] = self.kw['req_acc']
        self.vars['net_tmp'].learning.kw['req_err'] = self.kw['req_err']
        self.vars['net_tmp'].learning.kw['c_stable'] = self.kw['c_stable']
        self.vars['net_tmp'].learning.kw['n_epoch'] = self.kw['n_epoch']
        self.vars['net_tmp'].learning.kw['strict_termination'] = self.kw['strict_termination_learning']
        self.vars['net_tmp'].learning.kw['verbose'] = self.kw['verbose_learning']
        self.prune_()

    def prune_(self):
        print_pruning_started(net=self.net, kw=self.kw, vars=self.vars, stats=self.stats)
        while not self.vars['pruned']:
            t0 = time()
            self.vars['step'] += 1
            self.vars['net_tmp'].set_params_(from_net=self.net)
            self.cut_()
            self.shrink_()
            
            if self.vars['net_tmp'].learning.retrainable_(stats=self.stats):
                self.stats['retrained'].append(True)
                self.net.set_params_(from_net=self.vars['net_tmp'])
            else:
                self.stats['retrained'].append(False)
                if self.stats['n_to_cut'][-1] == 1 or self.vars['level'] == 0:
                    self.vars['pruned'] = True
                else:
                    self.vars['level_ind'] += 1
                    self.vars['level'] = self.kw['levels'][self.vars['level_ind']]
            
            self.stats['step_time'].append(time()-t0)
            if self.kw['verbose']:
                print_pruning_step(stats=self.stats, vars=self.vars)
        print_pruning_finished(net=self.net, kw=self.kw, stats=self.stats)
    
    def cut_(self):
        changes_ = [abs(w-w0) for w, w0 in zip(self.vars['net_tmp'].w, self.vars['net_tmp'].w_init)]
        changes_active_ = list()
        for ch_mat, w_is_mat in zip(changes_, self.vars['net_tmp'].w_is):
            for ch_, w_is in zip(hstack(ch_mat), hstack(w_is_mat)):
                if w_is:
                    changes_active_.append(ch_)

        if self.vars['level'] > 0:
            th_ = percentile(a=changes_active_, q=self.vars['level'])
        else:
            th_ = min(changes_active_)
        where_ = [logical_and(ch_<=th_, w_is!=0) for ch_, w_is in zip(changes_, self.vars['net_tmp'].w_is)]
        for w_is, wh_ in zip(self.vars['net_tmp'].w_is, where_):
            w_is[wh_] = 0.0
        self.stats['n_to_cut'].append(sum([np_sum(wh_) for wh_ in where_]))

    def shrink_(self):
        # Delete neurons with no inputs
        for l_i in range(len(self.vars['net_tmp'].structure)-2):
            neurons_to_delete = where(~self.vars['net_tmp'].w_is[l_i].any(axis=1))[0]
            self.vars['net_tmp'].w_is[l_i] = delete(self.vars['net_tmp'].w_is[l_i], neurons_to_delete, axis=0)
            self.vars['net_tmp'].w[l_i] = delete(self.vars['net_tmp'].w[l_i], neurons_to_delete, axis=0)
            self.vars['net_tmp'].w_init[l_i] = delete(self.vars['net_tmp'].w_init[l_i], neurons_to_delete, axis=0)
            self.vars['net_tmp'].b[l_i] = delete(self.vars['net_tmp'].b[l_i], neurons_to_delete, axis=0)
            self.vars['net_tmp'].b_is[l_i] = delete(self.vars['net_tmp'].b_is[l_i], neurons_to_delete, axis=0)
            self.vars['net_tmp'].b_init[l_i] = delete(self.vars['net_tmp'].b_init[l_i], neurons_to_delete, axis=0)
            self.vars['net_tmp'].w_is[l_i+1] = delete(self.vars['net_tmp'].w_is[l_i+1], neurons_to_delete, axis=1)
            self.vars['net_tmp'].w[l_i+1] = delete(self.vars['net_tmp'].w[l_i+1], neurons_to_delete, axis=1)
            self.vars['net_tmp'].w_init[l_i+1] = delete(self.vars['net_tmp'].w_init[l_i+1], neurons_to_delete, axis=1)

        # Delete neurons with no outputs
        f_ind_to_delete = where(~self.vars['net_tmp'].w_is[0].any(axis=0))[0]
        self.vars['net_tmp'].used_features = [(i_f, f) for i_f, (old_i_f, f) in enumerate(self.vars['net_tmp'].used_features) if i_f not in f_ind_to_delete]
        
        tmp = list()
        tmp_y = list()
        for x, y in self.net.t_data:
            tmp.append(array([x[i_f] for (i_f, f) in self.vars['net_tmp'].used_features]))
            tmp_y.append(y.copy())
        self.vars['net_tmp'].t_data = zip(tmp, tmp_y)

        tmp = list()
        tmp_y = list()
        for x, y in self.net.v_data:
            tmp.append(array([x[i_f] for (i_f, f) in self.vars['net_tmp'].used_features]))
            tmp_y.append(y.copy())
        self.vars['net_tmp'].v_data = zip(tmp, tmp_y)

        self.vars['net_tmp'].w_is[0] = delete(self.vars['net_tmp'].w_is[0], f_ind_to_delete, axis=1)
        self.vars['net_tmp'].w[0] = delete(self.vars['net_tmp'].w[0], f_ind_to_delete, axis=1)
        self.vars['net_tmp'].w_init[0] = delete(self.vars['net_tmp'].w_init[0], f_ind_to_delete, axis=1)
        for l_i in range(1, len(self.vars['net_tmp'].structure)-1):
            neurons_to_delete = where(~self.vars['net_tmp'].w_is[l_i].any(axis=0))[0]
            self.vars['net_tmp'].w_is[l_i] = delete(self.vars['net_tmp'].w_is[l_i], neurons_to_delete, axis=1)
            self.vars['net_tmp'].w[l_i] = delete(self.vars['net_tmp'].w[l_i], neurons_to_delete, axis=1)
            self.vars['net_tmp'].w_init[l_i] = delete(self.vars['net_tmp'].w_init[l_i], neurons_to_delete, axis=1)
            self.vars['net_tmp'].w_is[l_i-1] = delete(self.vars['net_tmp'].w_is[l_i-1], neurons_to_delete, axis=0)
            self.vars['net_tmp'].w[l_i-1] = delete(self.vars['net_tmp'].w[l_i-1], neurons_to_delete, axis=0)
            self.vars['net_tmp'].w_init[l_i-1] = delete(self.vars['net_tmp'].w_init[l_i-1], neurons_to_delete, axis=0)
            self.vars['net_tmp'].b[l_i-1] = delete(self.vars['net_tmp'].b[l_i-1], neurons_to_delete, axis=0)
            self.vars['net_tmp'].b_is[l_i-1] = delete(self.vars['net_tmp'].b_is[l_i-1], neurons_to_delete, axis=0)
            self.vars['net_tmp'].b_init[l_i-1] = delete(self.vars['net_tmp'].b_init[l_i-1], neurons_to_delete, axis=0)
            
        # update net structure
        self.vars['net_tmp'].structure = [self.vars['net_tmp'].w[0].shape[1]]+[w.shape[0] for w in self.vars['net_tmp'].w]