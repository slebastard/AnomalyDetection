# -*- coding: utf-8 -*-
"""
Created on Sat May 28 17:41:49 2016

@author: Simon
"""
import numpy as np
import pandas as pd

def createArtiSet(nb_inliers = 1000, nb_outliers = 120, verbose = False):
    if(verbose):
        print('Création de la BDD...')
    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate train data
    X = 0.3 * np.random.randn(nb_inliers/2, 2)
    data_inliers = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    data_outliers = np.random.uniform(low=-4, high=4, size=(nb_outliers, 2))
    if(verbose):
        print('...effectué')
    return pd.DataFrame(np.vstack((data_inliers, data_outliers)))
    
    