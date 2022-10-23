#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import random

class LDPRandomizer:
    def __init__(self,
                 epsilon = 0.5,
                 alpha = 10,
                 m = 4,
                 n = 5
                 ):
    
        self.epsilon = epsilon
        self.alpha  = alpha
        self.m = m
        self.n = n
        
    def float_to_binary(self, x, m, n):
        x_scaled = round(x * 2 ** n)   
        abs_xsaled=abs(x_scaled)
        format_string = '0{}b'.format(m + n)
        if x_scaled<0:
            bin_string = '1' + format(abs_xsaled, format_string)
        else:
            bin_string = '0' + format(abs_xsaled, format_string)
        
        return bin_string[0:10] 

    def flip(self, p):
        return 1 if random.random() < p else 0

    def randomize(self, bstr, epsilon, sensitivity, alpha):      
        multiplied=[]
        flip1 = 0
        flip2 = 0
        
        prob1_1 = alpha/(1+alpha)
        prob1_2 = 1/(1+math.pow(alpha,3))
        prob2 = (alpha*math.exp(epsilon/sensitivity))/(alpha*math.exp(epsilon/sensitivity)+1)
        prob2_1 = prob2
        prob2_2 = prob2
        
        it = 0
        for i in range(len(bstr)):       
            if (it % 2 == 0):    
                if bstr[i]==1:
                    flip1 = self.flip(prob1_1)
                    if flip1 == 1:
                        multiplied.append(1)
                    else:
                        multiplied.append(0)
                else:                    
                    flip2 = self.flip(prob2_1)
                    if flip2 == 1:
                        multiplied.append(0)
                    else:
                        multiplied.append(1)                       
            else:
                if bstr[i]==1:
                    flip1 = self.flip(prob1_2)
                    if flip1 == 1:
                        multiplied.append(1)
                    else:
                        multiplied.append(0)
                else:                
                    flip2 = self.flip(prob2_2)
                    if flip2 == 1:
                        multiplied.append(0)
                    else:
                        multiplied.append(1)
            it = it+1                
            
        return multiplied

    def flattenrand(self, x_train):      
        sensitivity = (len(x_train[0]) * (self.m+self.n+1))/2
        Xb_train=[]
        for ele in x_train:
            
            bval='';
            for subele in ele:       
                bval=bval + self.float_to_binary(float(subele), self.m, self.n)
            bval=np.array(list(bval))
            
            bval = bval.transpose()
            bval = bval.astype(np.float) 
            bval1 = np.array(self.randomize(bval, self.epsilon, sensitivity, self.alpha))
            Xb_train.append(bval1)

        return np.array(Xb_train)     
    
