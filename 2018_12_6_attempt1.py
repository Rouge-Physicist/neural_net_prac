# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:48:32 2018

@author: oliver
"""

import numpy as np 
import scipy as sp 
import sklearn as sk
import sympy as sym

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoid_deriv(x):
    output = sigmoid(x)*(1-sigmoid(x))
    return output

def gradvec_Calc(weights,bias):
    grad_vec = 0
    return grad_vec

class Neuralnet(): 
    
    def __init__(self,node_nums,inpts,outpts,learning_rate):
        self.nodes_in_layer = node_nums
        self.input = inpts 
        self.outpts = outpts
        self.learnrate = learning_rate
        self.weights = []
        self.bias = []
        for n,x in enumerate(node_nums):
            self.bias.append(np.random.uniform(0,1))
            if n == 0 :
                self.weights.append(np.random.uniform(0,1,(x,len(self.input[0]))))
            else :
                self.weights.append(np.random.uniform(0,1,(node_nums[n],node_nums[n-1])))

        self.weights.append(np.random.uniform(0, 1, (len(self.outpts[0]),node_nums[-1])))
        self.bias.append(0)
        
    def feedforward(self):
        costfunc = []
        for k,z in enumerate(self.input):
            self.layers = []
            self.layer_deriv = []
            for x in self.nodes_in_layer:
                self.layers.append(np.zeros((x,1)))
                self.layer_deriv.append(np.zeros((x, 1)))
            num = len(self.outpts)
            self.layers.append(np.zeros((num,1)))
            self.layer_deriv.append(np.zeros((num, 1)))
            for n,x in enumerate(self.weights):
                if n == 0 :
                    self.layers[n] = np.dot(x,self.input[k]) + self.bias[n]
                else :
                    self.layers[n] = np.dot(x,self.layers[n-1]) + self.bias[n]
                self.layer_deriv[n] = sigmoid_deriv(self.layers[n])
                self.layers[n] = sigmoid(self.layers[n])
            costfunc.append(np.sum((self.outpts[k] - self.layers[-1]) ** 2))
        Error = sum(costfunc)
        print(Error,'\n',costfunc)
    def backprop(self):
        grad_vec = self.weights
        for n,x in enumerate(self.weights):
            if n ==0 :
                error[-(n+1)] = -(self.outpts-self.layers[-1])
                grad_vec[-(n+1)] = error[-(n+1)]*self.layer_deriv[-(n+1)]
            else:
                error[-(n+1)] = error[-n].dot(self.weights[-(n+1)])
                grad_vec[-(n+1)] = error[-(n+1)]*self.layer_deriv[-(n+1)]





        ''' 
        C(y) = 0.5*SUM(Y2-y1)^2 + 0.5*SUM(Y2-y2)^2 
        
        therefore dC/dw = dC/dyi * dyi/dwij where yi = sigmoid(w1*y + w2*y2 + ...)
        '''

    def Run_net(self):
        self.feedforward()
        self.backprop()


inpts = np.array([[2,1,3,4],[1,2,3,4]])
outpts = np.array([[0,0,1],[0,1,0]])
for n,x in enumerate(inpts):
    inpts[n].shape = (len(inpts[n]),1)
    outpts[n].shape = (len(outpts[n]),1)

neuralnet = Neuralnet((3,5),inpts,outpts,0.1)
neuralnet.feedforward()

print(neuralnet.layers)
