# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:50:54 2018

@author: gaoxy
"""

import math
import random
import pickle

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

random.seed(0)

def rand(a, b):
    return (b - a)*random.random() + a

def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))

def dsigmoid(y):
    return y*(1 - y)

class Unit:
    def __init__(self, length):
        self.weight = [rand(-0.2, 0.2) for i in range(length)]
        self.change = [0.0] * length
        self.threshold = rand(-0.2, 0.2)
    
    def calc(self, sample):
        self.sample = sample[:]
        partsum = sum([i * j for i, j in zip(self.sample, self.weight)]) - self.threshold
        self.output = sigmoid(partsum)
        return self.output
    
    def update(self, diff, rate=0.5, factor=0.1):
        change = [rate * x * diff + factor * c for x, c in zip(self.sample, self.change)]
        self.weight = [w + c for w, c in zip(self.weight, change)]
        self.change = [x * diff for x in self.sample]
        
    def get_weight(self):
        return self.weight[:]
    
    def set_weight(self, weight):
        self.weight = weight[:]
        
class Layer:
    def __init__(self, input_length, output_length):
        self.units = [Unit(input_length) for i in range(output_length)]
        self.output = [0.0] * output_length
        self.ilen = input_length
    
    def calc(self, sample):
        self.output = [unit.calc(sample) for unit in self.units]
        return self.output[:]
    
    def update(self, diffs, rate=0.5, factor=0.1):
        for diff, unit in zip(diffs, self.units):
            unit.update(diff, rate, factor)
    
    def get_error(self, deltas):
        def _error(deltas, j):
            return sum([delta * unit.weight[j] for delta, unit in zip(deltas, self.units)])
        return [_error(deltas, j) for j in range(self.ilen)]
    
    def get_weights(self):
        weights = {}
        for key, unit in enumerate(self.units):
            weights[key] = unit.get_weight()
        return weights
    
    def set_weights(self, weights):
        for key, unit in enumerate(self.units):
            unit.set_weight(weights[key])


class BPNN:
    def __init__(self, ni, nh, no):
        self.ni = ni + 1
        self.nh = nh
        self.no = no
        self.hlayer = Layer(self.ni, self.nh)
        self.olayer = Layer(self.nh, self.no)
        
    def calc(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')
        self.ai = inputs[:] + [1.0]
        self.ah = self.hlayer.calc(self.ai)
        self.ao = self.olayer.calc(self.ah)
        return self.ao[:]
    
    def update(self, targets, rate, factor):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')
        output_deltas = [dsigmoid(ao) * (target - ao) for target, ao in zip(targets, self.ao)]
        hidden_deltas = [dsigmoid(ah) * error for ah, error in zip(self.ah, self.olayer.get_error(output_deltas))]
        self.olayer.update(output_deltas, rate, factor)
        self.hlayer.update(hidden_deltas, rate, factor)
        return sum([0.5 * (t - o)**2 for t, o in zip(targets, self.ao)])
    
    def train(self, data, label, iterations=1000, N=0.5, M=0.1):
        for i in range(iterations):
            error = 0.0
            for x, y in zip(data, label):
                self.calc(x)
                error = error + self.update(y, N, M)
            if i % 100 == 0:
                print('error %-.10f' % error)
    
    def save_weights(self, path):
        weights = {
                "olayer":self.olayer.get_weights(),
                "hlayer":self.hlayer.get_weights()
                }
        with open(path, "wb") as f:
            pickle.dump(weights, f)
            
    def load_weights(self, path):
        with open(path, "rb") as f:
            weights = pickle.load(f)
            self.olayer.set_weights(weights["olayer"])
            self.hlayer.set_weights(weights["hlayer"])
    
    def test(self, data, label):
        label_pred = []
        min = 0
        max = 1
        for x, y in zip(data, label):
            pred = self.calc(x)[0]
            if y == 1:
                if pred < min:
                    min = pred
            else:
                if pred > max:
                    max = pred
            if (pred > 0.99):
                label_pred.append(1)
            else:
                label_pred.append(0)
        print('min: ', min)
        print('max: ', max)
        acc = accuracy_score(label, label_pred)
        print('Test Dataset accuracy: ', acc)
        roc = roc_auc_score(label, label_pred)
        print('Test Dataset AUC: ', roc)
        f1 = f1_score(label, label_pred)
        print('Test Dataset f1_score: ', f1)
        print(classification_report(label, label_pred))
        print()
        print('Cofusion Matrix: ')
        print(confusion_matrix(label, label_pred))
        print()
    
def demo():
    # Teach network XOR function
    X = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]
    Y = [[0], [1], [1], [0]]
    # create a network with two input, two hidden, and one output nodes
    n = BPNN(2, 2, 1)
    # train it with some patterns
    n.train(X, Y)
    # test it
    n.save_weights("demo.weights")
    n.test(X, Y)

if __name__=='__main__':
    demo()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        