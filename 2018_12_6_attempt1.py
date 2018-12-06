# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:48:32 2018

@author: oliver
"""

import numpy as np 
import scipy as sp 
import sklearn as sk

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

class Neuralnet(): 
    
    def __init__(self,node_nums,inpts,outpts):
        self.nodes_in_layer = node_nums
        self.input = inpts 
        self.outpts = outpts
        self.weights = []
        for n,x in enumerate(node_nums):
            if n == 0 :
                self.weights.append(np.random.uniform(0,1,(x,len(self.input))))
            else :
                self.weights.append(np.random.uniform(0,1,(node_nums[n],node_nums[n-1])))
        
        
    def feedforward(self):
        self.layers = []
        for x in self.nodes_in_layer:
            self.layers.append(np.zeros((x,1)))
        for n,x in enumerate(self.weights):
            if n == 0 :
                self.layers[n] = np.dot(self.input,x)
            else :
                self.layers[n] = np.dot(self.layers[n-1],x)
    
            


neuralnet = Neuralnet((3,5),[2,1,3],[0,0,1])
#neuralnet.feedforward()
print(neuralnet.weights)
