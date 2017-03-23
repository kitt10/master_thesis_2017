# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_net
    ~~~~~~~~~~~~~~~~~
    Artificial neural network.
"""

import kitt_tf
from kitt_learning import Backpropagation
from kitt_optimization import Pruning, FeatureEnergy
from kitt_monkey import print_initialized, print_message
from numpy.random import standard_normal
from numpy import array, dot, ones, unique, argmax, inf, copy, sum as np_sum
from cPickle import dump as dump_cpickle, load as load_cpickle

class FeedForwardNet(object):
    
    def __init__(self, hidden, tf_name='Tanh'):
        self.structure = hidden                             # network hidden structure [h1, h2, ..., hk]
        self.tf_name = tf_name
        self.tf = getattr(kitt_tf, self.tf_name)()          # transfer function
        self.w = self.b = None                              # weights and biases
        self.w_init = self.b_init = None                    # initial weights and biases
        self.w_is = self.b_is = None                        # existence of weights and biases
        self.labels = list()                                # list of sorted class labels
        self.label_sign = dict()                            # and corresponding vectorized signatures
        self.t_data = None                                  # training data for this net
        self.v_data = None                                  # validation data for this net
        self.learning = None                                # net learning algorithm
        self.opt = dict()                                   # optimization method instances
        self.used_features = None                           # features still interesting after pruning

    def init_(self, n, x, y, x_val, y_val):
        self.labels = sorted(unique(y))
        for index, label in enumerate(self.labels):
            self.label_sign[label] = self.tf.neg*ones(shape=(len(self.labels), 1))
            self.label_sign[label][index][0] = self.tf.pos

        self.structure = [n]+self.structure+[len(self.labels)]             # overall network structure [n, h1, h2, ..., hk, m]
        self.w = [standard_normal(size=(i, j)) for j, i in zip(self.structure[:-1], self.structure[1:])]
        self.b = [standard_normal(size=(i, 1)) for i in self.structure[1:]]
        self.w_is = [ones(shape=w_i.shape) for w_i in self.w]
        self.b_is = [ones(shape=b_i.shape) for b_i in self.b]
        self.w_init = [copy(w) for w in self.w]
        self.b_init = [copy(b) for b in self.b]
        self.t_data = zip(x, array([self.label_sign[y_i] for y_i in y]))
        if x_val is not None and y_val is not None:
            self.v_data = zip(x_val, array([self.label_sign[y_i] for y_i in y_val]))
        self.used_features = zip(range(self.structure[0]), range(self.structure[0]))
        print_initialized(net=self)

    def forward(self, a):
        for b, w in zip(self.b, self.w):
            a = self.tf.fire(z=(dot(w, a) + b))
        return a
    
    def predict(self, x):
        return [(self.labels[i[0]], i[1][0]) for i in sorted(enumerate(self.forward(a=x)), key=lambda x:x[1], reverse=True)]

    def fit(self, x, y, x_val=None, y_val=None, learning_rate=0.03, batch_size=1, n_epoch=int(1e10), c_stable=inf, req_acc=inf, req_err=-inf, strict_termination=False, verbose=True):
        self.init_(n=len(x[0]), x=x, y=y, x_val=x_val, y_val=y_val)
        self.learning = Backpropagation(locals())
        self.learning.learn_()
    
    def evaluate(self, x, y):
        return self.evaluate_(data=zip(x, array([self.label_sign[y_i] for y_i in y])))
    
    def evaluate_(self, data, acc_buf=None, err_buf=None):
        n_correct = 0
        err = 0
        for x_i, y_i in data:
            u_i = self.forward(a=x_i)
            if argmax(u_i) == argmax(y_i):
                n_correct += 1
            err += sum([(u_ii-y_ii)*(u_ii-y_ii) for u_ii, y_ii in zip(u_i, y_i)])[0]
        err /= 2*len(data)*len(data[0][1])
        if acc_buf is not None:
            acc_buf.append(float(n_correct)/len(data))
        if err_buf is not None:
            err_buf.append(err)
        return err, float(n_correct)/len(data)

    def prune(self, req_acc=1.0, req_err=0.0, n_epoch=100, c_stable=10, levels=(50, 35, 10, 0), strict_termination_learning=True, verbose=True, verbose_learning=False):
        self.opt['pruning'] = Pruning(locals())

    def count_synapses(self):
        return (int(sum([np_sum(w_i) for w_i in self.w_is])), int(sum([np_sum(b_i) for b_i in self.b_is])))

    def copy_(self):
        new_net = FeedForwardNet(hidden=[], tf_name=self.tf_name)
        new_net.set_params_(from_net=self)
        new_net.labels = self.labels[:]
        new_net.label_sign = self.label_sign.copy()
        new_net.learning = Backpropagation(kw={'self': new_net, 'learning_rate': self.learning.kw['learning_rate'], 'batch_size': self.learning.kw['batch_size']})
        return new_net
    
    def set_params_(self, from_net):
        self.w = [w.copy() for w in from_net.w]
        self.b = [b.copy() for b in from_net.b]
        self.w_is = [w_is.copy() for w_is in from_net.w_is]
        self.b_is = [b_is.copy() for b_is in from_net.b_is]
        self.w_init = [w_init.copy() for w_init in from_net.w_init]
        self.b_init = [b_init.copy() for b_init in from_net.b_init]
        self.structure = from_net.structure[:]
        self.t_data = zip(array([x[0].copy() for x in from_net.t_data]), array([x[1].copy() for x in from_net.t_data]))
        self.v_data = zip(array([x[0].copy() for x in from_net.v_data]), array([x[1].copy() for x in from_net.v_data]))
        self.used_features = from_net.used_features[:]

    def dump(self, net_file_name):
        net_pack = {'w': self.w, 'b': self.b, 'w_is': self.w_is, 'b_is': self.b_is, 'w_init': self.w_init, 'b_init': self.b_init,
                    'structure': self.structure, 'tf': self.tf_name, 'labels': self.labels, 'features': self.used_features}
        with open(net_file_name, 'w') as f:
            dump_cpickle(net_pack, f)
        print_message(message='Net dumped as '+net_file_name)
    
    def load(self, net_file_name):
        with open(net_file_name, 'r') as f:
            net_pack = load_cpickle(f)
        self.w_is = net_pack['w_is']

    def compute_feature_energy(self):
        self.opt['feature_energy'] = FeatureEnergy(locals())
