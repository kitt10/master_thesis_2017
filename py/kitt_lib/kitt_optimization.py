# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_optimization
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Optimization methods.
"""

from kitt_monkey import print_pruning_started, print_pruning_step, print_pruning_finished
from numpy import array, percentile, where, hstack, logical_and, inf, delete, nonzero, concatenate, zeros, unique, \
    newaxis, argmax, sum as np_sum
from numpy.random import uniform
from sklearn.metrics import confusion_matrix
from shelve import open as open_shelve
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
        if self.kw['measure'] == 'kitt':
            changes_ = [abs(w-w0) for w, w0 in zip(self.vars['net_tmp'].w, self.vars['net_tmp'].w_init)]
        elif self.kw['measure'] == 'karnin':
            changes_ = [zeros(shape=w_i.shape) for w_i in self.vars['net_tmp'].w]
            for ep_i in range(self.vars['net_tmp'].dw_i):
                for l_i in range(len(changes_)):
                    changes_[l_i] += self.vars['net_tmp'].dw_container[l_i][ep_i]**2 * \
                                     (self.vars['net_tmp'].w[l_i]/(self.vars['net_tmp'].learning.kw['learning_rate']*(self.vars['net_tmp'].w[l_i]-self.vars['net_tmp'].w_init[l_i])))

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

            if self.kw['measure'] == 'karnin':
                for ep_i in range(self.vars['net_tmp'].dw_i):
                    self.vars['net_tmp'].dw_container[l_i][ep_i] = delete(self.vars['net_tmp'].dw_container[l_i][ep_i], neurons_to_delete, axis=0)
                    self.vars['net_tmp'].dw_container[l_i+1][ep_i] = delete(self.vars['net_tmp'].dw_container[l_i+1][ep_i], neurons_to_delete, axis=1)

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
        if self.kw['measure'] == 'karnin':
            for ep_i in range(self.vars['net_tmp'].dw_i):
                self.vars['net_tmp'].dw_container[0][ep_i] = delete(self.vars['net_tmp'].dw_container[0][ep_i], f_ind_to_delete, axis=1)

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

            if self.kw['measure'] == 'karnin':
                for ep_i in range(self.vars['net_tmp'].dw_i):
                    self.vars['net_tmp'].dw_container[l_i][ep_i] = delete(self.vars['net_tmp'].dw_container[l_i][ep_i], neurons_to_delete, axis=1)
                    self.vars['net_tmp'].dw_container[l_i-1][ep_i] = delete(self.vars['net_tmp'].dw_container[l_i-1][ep_i], neurons_to_delete, axis=0)

        # update net structure
        self.vars['net_tmp'].structure = [self.vars['net_tmp'].w[0].shape[1]]+[w.shape[0] for w in self.vars['net_tmp'].w]


class FeatureEnergy(object):

    def __init__(self, net):
        self.net = net
        self.paths = dict()
        self.energies = dict()

    def find_paths(self):
        f = 0
        for f_i in range(self.net.n_features_init):
            self.paths[f_i] = list()
            if f_i in [uf[1] for uf in self.net.used_features]:
                ks = nonzero(self.net.w_is[0][:, f])[0]
                for k in ks:
                    qs = nonzero(self.net.w_is[1][:, k])[0]
                    for q in qs:
                        self.paths[f_i].append(list())
                        self.paths[f_i][-1].append((k, self.net.w[0][k, f], self.net.b[0][k][0]))
                        self.paths[f_i][-1].append((q, self.net.w[1][q, k], self.net.b[1][q][0]))
                f += 1

    def compute_energies(self):
        for f_i in range(self.net.n_features_init):
            self.energies[f_i] = dict()
            for c_i, label in enumerate(self.net.labels):
                self.energies[f_i][label] = 0.0
                for path in self.paths[f_i]:
                    if c_i == path[-1][0]:
                        self.energies[f_i][label] += float(path[0][1])/abs(path[0][2]) * float(path[1][1])/abs(path[1][2])
            self.energies[f_i]['total'] = sum([e for e in self.energies[f_i].values()])


class Tailoring(object):
    
    def __init__(self, net):
        self.net = net

    def add_neurons(self, class_labels, h=1):
        m_i_s = [self.net.labels.index(label) for label in class_labels]

        for h_i in range(h):
            self.net.w[0] = concatenate((self.net.w[0], array([uniform() for w_i in range(self.net.w[0].shape[1])], ndmin=2)), axis=0)
            self.net.w_init[0] = concatenate((self.net.w_init[0], array(self.net.w[0][-1,:], ndmin=2)), axis=0)
            self.net.w_is[0] = concatenate((self.net.w_is[0], array([1 for w_i in range(self.net.w_is[0].shape[1])], ndmin=2)), axis=0)
            self.net.b[0] = concatenate((self.net.b[0], array([uniform()], ndmin=2)), axis=0)
            self.net.b_init[0] = concatenate((self.net.b_init[0], array(self.net.b[0][-1], ndmin=2)), axis=0)
            self.net.b_is[0] = concatenate((self.net.b_is[0], array([1], ndmin=2)), axis=0)
            
            self.net.w[1] = concatenate((self.net.w[1], array([uniform() for w_i in range(self.net.w[1].shape[0])], ndmin=2).T), axis=1)
            self.net.w_init[1] = concatenate((self.net.w_init[1], array(self.net.w[1][:,-1], ndmin=2).T), axis=1)
            self.net.w_is[1] = concatenate((self.net.w_is[1], array([0 for w_i in range(self.net.w_is[1].shape[0])], ndmin=2).T), axis=1)
            for m_i in m_i_s:
                self.net.w_is[1][m_i, -1] = 1

    def disconnect_class(self, class_labels):
        for lab in class_labels:
            self.net.w_is[1][self.net.labels.index(lab), :] = 0
            self.net.w[1][self.net.labels.index(lab), :] = 0
            self.net.b_is[1][self.net.labels.index(lab)] = 0
            self.net.b[1][self.net.labels.index(lab)] = 0

    def insert_net(self, net, mapping):
        tmp_1 = self.net.w[1].copy()
        z_1 = zeros(shape=(self.net.w[1].shape[0], net.structure[1]))
        for lab, lab_i in mapping:
            z_1[self.net.labels.index(lab), :] = net.w[1][lab_i]
            self.net.b[1][self.net.labels.index(lab)] = net.b[1][lab_i]
        self.net.w[1] = concatenate((tmp_1, z_1), axis=1)

        used_features_new = [(i, ff) for i, ff in enumerate(sorted(unique([f[1] for f in self.net.used_features]+[f[1] for f in net.used_features])))]
        features = [f[1] for f in used_features_new]
        w_0 = zeros(shape=(self.net.structure[1]+net.structure[1], len(used_features_new)))
        b_0 = zeros(shape=(self.net.structure[1]+net.structure[1], 1))
        for row in range(self.net.w[0].shape[0]):
            for col in range(self.net.w[0].shape[1]):
                w_0[row, features.index(self.net.used_features[col][1])] = self.net.w[0][row, col]
            b_0[row] = self.net.b[0][row]

        for row in range(net.w[0].shape[0]):
            for col in range(net.w[0].shape[1]):
                w_0[self.net.w[0].shape[0]+row, features.index(net.used_features[col][1])] = net.w[0][row, col]
            b_0[net.w[0].shape[0]+row] = net.b[0][row]

        self.net.used_features = used_features_new
        self.net.w[0] = w_0
        self.net.b[0] = b_0

    def analyse_cm(self, x_val, y_val, th=0.15):
        y_pred = [self.net.predict(x)[0][0] for x, y in self.net.prepare_data(x=x_val, y=y_val)]
        cm = confusion_matrix(y_true=y_val, y_pred=y_pred)
        cm = cm.astype('float')/cm.sum(axis=1)[:, newaxis]
        '''
        cm_an = dict()
        for i in range(cm.shape[0]):
            cm_an[self.net.labels[i]] = sorted([(self.net.labels[j], round(cm[i, j], 2)) for j in range(cm.shape[1])],
                                               key=lambda x: x[1], reverse=True)
        '''
        groups = list()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > th:
                    for rn in groups:
                        if self.net.labels[i] in rn and self.net.labels[j] not in rn:
                            rn.append(self.net.labels[j])
                            break
                        elif self.net.labels[j] in rn and self.net.labels[i] not in rn:
                            rn.append(self.net.labels[i])
                            break
                        elif self.net.labels[j] in rn and self.net.labels[i] in rn:
                            break
                    else:
                        groups.append([self.net.labels[i], self.net.labels[j]])
        return groups

    def create_dataset(self, ph_keep, a_path):
        na = str(ph_keep)[1:-1].replace(' ', '').replace('\'', '').replace(',', '+')
        dataset_new = open_shelve(a_path+'dataset_'+na+'.ds')
        tmp_x = list()
        tmp_y = list()
        for x, y in self.net.t_data:
            if y in ph_keep:
                tmp_y.append(y)
                tmp_x.append(x)
        dataset_new['x'] = tmp_x
        dataset_new['y'] = tmp_y

        tmp_x = list()
        tmp_y = list()
        for x, y in self.net.v_data:
            if y in ph_keep:
                tmp_y.append(y)
                tmp_x.append(x)
        dataset_new['x_val'] = tmp_x
        dataset_new['y_val'] = tmp_y
        dataset_new.close()

    def train_subnet(self, ph_keep, a_path):
        na = str(ph_keep)[1:-1].replace(' ', '').replace('\'', '').replace(',', '+')
        #net = FeedForwardNet(hidden=[10], tf_name='Sigmoid')
        dataset = open_shelve(a_path+'dataset_'+na+'.ds')
        #net.fit(x=dataset['x'], y=dataset['y'], x_val=dataset['x_val'], y_val=dataset['y_val'], learning_rate=0.07, n_epoch=200, req_acc=1.0, batch_size=10)
        dataset.close()
        #net.dump(a_path+'net_'+na+'.net')
        #return net

    def map_output(self, class_labels, net):
        self.net.subnets.append((net, None))
        for label in class_labels:
            self.net.mapping[label] = argmax(self.net.subnets[-1][1])