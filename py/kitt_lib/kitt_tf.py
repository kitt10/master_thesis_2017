# -*- coding: utf-8 -*-

"""
    kitt_lib.kitt_tf
    ~~~~~~~~~~~~~~~~
    Transfer functions.
"""

from numpy import tanh, sinh, cosh, exp


class Tanh(object):
    
    def __init__(self):
        self.pos = 1
        self.neg = -1
        
    def fire(self, z):
        return tanh(z)

    def prime(self, z):
        return 1-(tanh(z)*tanh(z))

    def prime2(self, z):
        return -8*sinh(z)/(3*cosh(z)+cosh(3*z))


class Sigmoid(object):
    
    def __init__(self):
        self.pos = 1
        self.neg = 0

    def fire(self, z):
        return 1.0/(1.0+exp(-z))

    def prime(self, z):
        return self.fire(z)*(1-self.fire(z))

    def prime2(self, z):
        return (exp(z)*(exp(z)-1))/(exp(z)+1)**3