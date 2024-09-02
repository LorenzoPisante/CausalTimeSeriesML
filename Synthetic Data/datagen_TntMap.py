# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:19:02 2023

@author: loren
"""

"""
Created on Thu Nov  9 20:40:17 2023

@author: loren

y: Master_timeseries : class 0

x : Slave_timeseries : class 1

TENTMAP1:  B1 = 0.65, B2 = 0.47
TENTMAP2:  B1 = 0.6, B2 = 0.4
TENTMAP3:  B1 = 0.1, B2 = 0.

EQ: B1 = 0.3, B2= 0.3


TM4: 0,3  0,3
TL1: 0,6   0,6
TL2  0,7   0,4
"""


import random
import os
import numpy as np
import matplotlib.pyplot as plt

def skew_tent(x,b):
    if x < b:
        return x/b
    return  (1-x)/(1-b)

DATA_NAME = 'SKEW-TENTMAP-TL2'
LENGTH = 2500
TRANSIENT_LENGTH = 500
SAMPLES_PER_CLASS = 1000
B1 = 0.7
B2 = 0.4

COUP_COEFF = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


for COEFF in COUP_COEFF:

    class_0_data = np.zeros((SAMPLES_PER_CLASS, LENGTH - TRANSIENT_LENGTH))
    class_0_label = np.zeros((SAMPLES_PER_CLASS,1))
    
    class_1_data = np.zeros((SAMPLES_PER_CLASS, LENGTH - TRANSIENT_LENGTH))
    class_1_label = np.ones((SAMPLES_PER_CLASS,1))
    
    for NUM_TRIALS in range(0, SAMPLES_PER_CLASS):
        random.seed(NUM_TRIALS)
        x_in = (random.uniform(0, 1))
        y_in = (random.uniform(0, 1))
        master_timeseries = np.zeros(LENGTH)
        slave_timeseries = np.zeros(LENGTH)
            
        master_timeseries[0] = x_in
        slave_timeseries[0] = y_in
    
        for num_instance in range(1, LENGTH):
            master_timeseries[num_instance] = skew_tent(master_timeseries[num_instance - 1], B1)
            slave_timeseries[num_instance] = (1-COEFF/4) * skew_tent(slave_timeseries[num_instance - 1], B2) + COEFF/4 * master_timeseries[num_instance]
        
        class_0_data[NUM_TRIALS, :] = master_timeseries[TRANSIENT_LENGTH:]
        class_1_data[NUM_TRIALS, :] = slave_timeseries[TRANSIENT_LENGTH:]
    
    PATH = 'D:/loren/Documents/cod/'
    RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' + str(COEFF) +'/'
    
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    
    
    from scipy import io
    
    
    io.savemat(RESULT_PATH + 'Y_independent_data_class_0.mat', {'class_0_indep_raw_data': class_0_data})
    io.savemat(RESULT_PATH + 'Y_independent_label_class_0.mat', {'class_0_indep_raw_data_label': class_0_label})
    
    
    io.savemat(RESULT_PATH + 'X_dependent_data_class_1.mat', {'class_1_dep_raw_data': class_1_data})
    io.savemat(RESULT_PATH + 'X_dependent_label_class_1.mat', {'class_1_dep_raw_data_label': class_1_label})
    