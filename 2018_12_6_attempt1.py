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
    
    def __init__(self,nodes_in_layers,inpts_and_outpts,learning_rate):
        self.nodes_in_layer = nodes_in_layers
        self.inpts_and_outpts = inpts_and_outpts
        self.learnrate = learning_rate
        self.weights = []
        self.bias = []
        ''' generating the weight matrices  '''
        for n,x in enumerate(self.nodes_in_layer):
            if len(self.weights)>=3:
                self.bias.append(np.random.uniform(0,1))
                pass
            else:
                self.bias.append(np.random.uniform(0,1))
                self.weights.append(np.random.uniform(-1,1,(self.nodes_in_layer[n],self.nodes_in_layer[n+1])))
    
    def feedforward(self,inpts,oupts):
        ''' Generating the node outputs '''
        self.layer = [np.matrix(inpts)]
        for n,x in enumerate(self.weights):
            self.layer.append(sigmoid(np.matrix(self.layer[n])*x))
        
    def backprop(self,inpts,outpts):
        ''' calculate the delta/stuff that gets reused to calculate the next error '''
        delta = [np.matrix(-(outpts-self.layer[-1]).A1*self.layer[-1].A1*(1-self.layer[-1].A1))]
        for n,x in enumerate(self.weights):
            delta.append(np.matrix((delta[-1]*self.weights[-(n+1)].T).A1*self.layer[-(n+2)].A1*(1-self.layer[-(n+2)].A1)))
        ''' update the weights '''
        for m,x in enumerate(self.weights):
            self.weights[-(m+1)] += self.learnrate*(self.layer[-(m+2)].T*delta[m])
             
            
    def train_net(self):
        for x in self.inpts_and_outpts:
            self.feedforward(x[0],x[1])
            self.backprop(x[0],x[1])
            
inpts_and_outpts = np.array([[[2,1,3,4],[0,0,1]],[[1,1,3,4],[0,1,0]]])

nodes_in_hidden_layer = [3,5]
nodes_in_layers = [len(inpts_and_outpts[0][0])]+nodes_in_hidden_layer + [len(inpts_and_outpts[0][1])]
neuralnet = Neuralnet(nodes_in_layers,inpts_and_outpts,0.1)
neuralnet.train_net()


