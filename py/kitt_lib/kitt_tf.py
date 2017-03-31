# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_tf
    ~~~~~~~~~~~~~~~~
    Transfer functions.
"""

from numpy import tanh, exp as exp


class Tanh(object):
    
    def __init__(self):
        self.pos = 1
        self.neg = -1
        
    def fire(self, z):
        return tanh(z)

    def prime(self, z):
        return 1-(tanh(z)*tanh(z))


class Sigmoid(object):
    
    def __init__(self):
        self.pos = 1
        self.neg = 0

    def fire(self, z):
        return 1.0/(1.0+exp(-z))

    def prime(self, z):
        return self.fire(z)*(1-self.fire(z))