# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_learning
    ~~~~~~~~~~~~~~~~~~~~~~
    Learning methods.
"""

from kitt_monkey import print_learning_started, print_and_check_epoch, print_learning_finished
from numpy import zeros, dot, inf, multiply
from numpy.random import shuffle
from time import time


class Backpropagation(object):
    
    def __init__(self, kw):
        self.net = kw['self']
        self.kw = kw
        self.stats = {'t_err': list(), 't_acc': list(), 'v_err': list(), 'v_acc': list(), 'ep_time': list(), 
                      't_err_best': inf, 'v_err_best': inf, 'c_stable': 0}
    
    def learn_(self):
        print_learning_started(self.kw)
        for self.stats['i_epoch'] in xrange(1, self.kw['n_epoch']+1):
            self.net.dw_i += 1                                                      # Karnin
            self.net.saliency = [zeros(w.shape) for w in self.net.w]                # OBD
            shuffle(self.net.t_data)
            t0 = time()
            for mini_batch in [self.net.t_data[k:k+self.kw['batch_size']] for k in xrange(0, len(self.net.t_data), self.kw['batch_size'])]:
                self.update_mini_batch(mini_batch)
            self.stats['ep_time'].append(time()-t0)
            self.net.evaluate_(data=self.net.t_data, acc_buf=self.stats['t_acc'], err_buf=self.stats['t_err'])
            if self.net.v_data is not None:
                self.net.evaluate_(data=self.net.v_data, acc_buf=self.stats['v_acc'], err_buf=self.stats['v_err'])
            if print_and_check_epoch(self.stats, self.kw):
                break
        else:
            print_learning_finished(why='Given number of epochs performed.', t=sum(self.stats['ep_time']),
                                    verbose=self.kw['verbose'])
    
    def update_mini_batch(self, mini_batch):
        nabla_b = [zeros(b_i.shape) for b_i in self.net.b]
        nabla_w = [zeros(w_i.shape) for w_i in self.net.w]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate_error(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.net.w = multiply([w-self.kw['learning_rate']*nw for w, nw in zip(self.net.w, nabla_w)], self.net.w_is)
        self.net.b = multiply([b-self.kw['learning_rate']*nb for b, nb in zip(self.net.b, nabla_b)], self.net.b_is)

        for l, dw in enumerate(nabla_w):
            if self.net.dw_i > len(self.net.dw_container[l]):
                self.net.dw_container[l].append(zeros(self.net.w[l].shape))
            self.net.dw_container[l][self.net.dw_i-1] += dw

    def backpropagate_error(self, x, y):
        nabla_b = [zeros(b_i.shape) for b_i in self.net.b]
        nabla_w = [zeros(w_i.shape) for w_i in self.net.w]

        a_ = [x]                    # list to store all the activations, layer by layer
        z_ = list()                 # list to store all the z vectors, layer by layer
        for b, w in zip(self.net.b, self.net.w):
            z_.append(dot(w, a_[-1])+b)
            a_.append(self.net.tf.fire(z_[-1]))

        delta = (a_[-1]-y)*self.net.tf.prime(z_[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = dot(delta, a_[-2].transpose())

        if self.kw['nd_der']:
            delta2 = (y-a_[-1])*self.net.tf.prime2(z_[-1])+self.net.tf.prime(z_[-1])**2
            self.net.saliency[-1] += dot(delta2, (a_[-2]**2).transpose())

        for i_layer in xrange(2, len(self.net.structure)): 
            delta = dot(self.net.w[-i_layer+1].transpose(), delta) * self.net.tf.prime(z_[-i_layer])
            nabla_b[-i_layer] = delta
            nabla_w[-i_layer] = dot(delta, a_[-i_layer-1].transpose())

            if self.kw['nd_der']:
                delta2 = (-1)*dot(self.net.w[-i_layer+1].transpose(), delta2)*self.net.tf.prime2(z_[-i_layer])+self.net.tf.prime(z_[-i_layer])**2
                self.net.saliency[-i_layer] += dot(delta2, (a_[-i_layer-1]**2).transpose())

        return nabla_b, nabla_w

    def retrainable_(self, stats, retrain):
        if retrain:
            self.stats = {'t_err': list(), 't_acc': list(), 'v_err': list(), 'v_acc': list(), 'ep_time': list(),
                          't_err_best': inf, 'v_err_best': inf, 'c_stable': 0}
            try:
                self.learn_()
                stats['acc'].append(self.stats['t_acc'][-1])
                stats['err'].append(self.stats['t_err'][-1])
            except ValueError:
                stats['acc'].append(0.0)
                stats['err'].append(1.0)
                return False
            return self.stats['t_acc'][-1] >= self.kw['req_acc']                        # do retraining
        else:
            return self.net.evaluate_(data=self.net.t_data)[1] >= self.kw['req_acc']    # online pruning (no retraining)