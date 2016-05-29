# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:30:51 2016

@author: Simon
"""
import pandas as pd

def loadBDD(verbose = False):
    zfilename = 'kddcup.data_10_percent.gz'
    dataSetPath = '../../Datasets/'
    if(verbose):
        print('Chargement de la BDD...')
    networkData = pd.read_csv(dataSetPath + zfilename,
                   sep=",",
                   engine = "python")
    if(verbose):
        print('...effectué')
    return networkData