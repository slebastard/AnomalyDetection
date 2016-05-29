# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:51:26 2016

@author: Simon
"""
import sys, os

import numpy as np
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DataExtraction'))
import artiSet as art
import networkData as nd

### Quadratic Programming using pseudo-SMO
## See http://research.microsoft.com/pubs/69731/tr-99-87.pdf
## p.7

def gaussianKernel(inst1, inst2, C = 0.3):
    return np.exp((-np.linalg.norm(inst1 - inst2)**2)/C)

def shuffle(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

def batch(data, batch_size):
    return shuffle(data)[:batch_size]
    
def dropFeatures(df, remainingFeatures):
    # Here only the first #remainingFeatures# features will be kept
    return df[list(df.columns[:remainingFeatures])]

class oneClassSMO:
# This class is meant to contain:
## a set of data
## a kernel function
## the parameters necessary to
    def __init__(self, data, kernel, nu):
        self.data = data.as_matrix()
        self.l = self.data.shape[0]
        self.alphas = np.zeros(self.l)
        self.kernel = kernel
        self.nu = nu
        self.partSum = np.zeros(self.l)
        self.O = np.zeros(self.l)
        self.rho = 0
        self.alphasInit()
        self.energy = self.getEnergy()
        
    def getEnergy(self):
        indexes = range(self.l)
        # Partial energy used in SMO algorithm
        # You might want to launch that on batched data only
        # The first two lines must correspond to the variables
        # For all other exemples, the corresponding alpha value will be fixed
        part_energy = 0
        for ind1 in indexes:
            for ind2 in indexes:
                part_energy += 0.5 * self.alphas[ind1] * self.alphas[ind2] * self.kernel(self.data[ind2], self.data[ind1])

    def compPartSums(self):
        for ind in range(self.l):
            for ind2 in range(self.l):
                self.partSum[ind] += self.alphas[ind2] * self.kernel(self.data[ind], self.data[ind2])
        self.O = np.array([self.kernel(self.data[ind], self.data[0]) * self.alphas[0] + self.kernel(self.data[ind], self.data[1]) * self.alphas[1] + self.partSum[ind] for ind in range(self.l)])

    def stepOpt(self, ind1, ind2):
        Delta = self.alphas[ind1] + self.alphas[ind2]
        self.compPartSums()
        newAlpha = np.zeros(2)
        newAlpha[1] = (Delta*(self.kernel(self.alphas[ind1], self.alphas[ind1]) - self.kernel(self.alphas[ind1], self.alphas[ind2])) + self.partSum[ind1] - self.partSum[ind2])/(self.kernel(self.data[ind1], self.data[ind1]) + self.kernel(self.data[ind2], self.data[ind2]) - 2 * self.kernel(self.data[ind1], self.data[ind2]))
        newAlpha[0] = Delta - newAlpha[1]
        ## On teste si alphas[ind1] et alphas[ind2] verifient ou non KKT
        while(~(newAlpha[0] < 0 or newAlpha[0] > 1/(self.nu*self.l) or newAlpha[1] < 0 or newAlpha[1] > 1/(self.nu*self.l))):
            self.alphas[0] = self.projectionKKT(newAlpha[1])
            newAlpha[0] = Delta - newAlpha[1]
        self.alphas[ind1] = newAlpha[0]
        self.alphas[ind2] = newAlpha[1]
        ## A faire: exception pour verifier que data[ind2] est un SV
        self.rho = self.partSum[ind2]
                
    def projectionKKT(self, init_val):
        val = init_val
        if (init_val < 0):
            val = 0
        if (init_val > self.nu * self.l):
            val = self.nu * self.l
        return val
        
    def LowerKKT(self, ind):
        return ((self.O[ind] - self.rho) * self.alphas[ind] > 0)
        
    def UpperKKT(self, ind):
        return ((self.rho - self.O[ind]) * (1/(self.nu * self.l) - self.alphas[ind]) > 0)     
        
    def isSV(self, ind):
        return (0 < self.alphas[ind])
        
    def isExclusiveSV(self, ind):
        return (self.isSV(ind) and self.alphas[ind] < 1/(self.nu * self.l))
    
    def scanKKT(self):
        KKTViolationIndex = -1
        KKTViolationPoleIndex = -1
        for ind in range(self.l):
            if(~(self.LowerKKT(ind) and self.UpperKKT(ind))):
                KKTViolationIndex = ind
                break
        if (KKTViolationIndex == -1):
            return np.array([[-1,-1]])
            
        OGap = np.zeros(self.l)
        for ind2 in range(self.l):
            if (self.isExclusiveSV(ind2)):
                OGap[ind2] = np.abs(self.O[ind2] - self.O[KKTViolationIndex])
            else:
                OGap[ind2] = -1
        KKTViolationPoleIndex = np.argmax(OGap)
        return np.array([[KKTViolationIndex,KKTViolationPoleIndex]])

    def alphasInit(self):
        tmp = np.arange(self.l)
        np.random.shuffle(tmp)
        init_indexes = tmp[:int(self.nu*self.l)]
        for ind in range(int(self.nu*self.l)):
            self.alphas[init_indexes[ind]] = 1/(self.nu*self.l)
        if((self.nu*self.l) - int(self.nu*self.l) != 0):
            self.alphas[int(self.nu*self.l)] = (self.nu*self.l -int(self.nu*self.l))/(self.nu*self.l)
        self.compPartSums()     
        self.rho = np.max(self.O)
        
    def SMO(self):  
        ## Initialisation SMO
        print('Be patient, SMO is booting...')
        self.alphasInit()
        print('...done')
        ## Itérations
        [KKTViolationIndex, KKTViolationPoleIndex] = self.scanKKT()[0]
        while(KKTViolationIndex != -1):
            self.stepOpt(KKTViolationIndex, KKTViolationPoleIndex)
            [KKTViolationIndex, KKTViolationPoleIndex] = self.scanKKT()[0]

## Pour créer une instance de oneClassSMO:
## smo = oneClassSMO(data = dropFeatures(batch(#YOUR DATAFRAME#, #NUMBER OF ENTRIES TO KEEP#), #NUMBER OF FTRS TO KEEP#), kernel = gaussianKernel, nu = 0.5)

# print('Optimisation par descente SMO')
# smo.SMO()

artiData = art.createArtiSet()
batchSize = 700
featNumber = 2
smo = oneClassSMO(data = dropFeatures(batch(artiData, batchSize), featNumber), kernel = gaussianKernel, nu = 0.5)
