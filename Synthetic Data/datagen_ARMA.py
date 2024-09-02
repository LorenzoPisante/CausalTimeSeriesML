# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 05:14:39 2023

@author: loren
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:33:35 2023

@author: loren
"""

#AR GENERATOR
import random
import os
import numpy as np
import matplotlib.pyplot as plt




DATA_NAME = 'ARIMA2'
LENGTH = 2500
TRANSIENT_LENGTH = 500
SAMPLES_PER_CLASS = 1000
a1 = 0.9
a2= -0.4
c = 0.8
e1=0.1
e2=0.03

COUP_COEFF = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1])


for COEFF in COUP_COEFF:

    class_0_data = np.zeros((SAMPLES_PER_CLASS, LENGTH - TRANSIENT_LENGTH))
    class_0_label = np.zeros((SAMPLES_PER_CLASS,1))
    
    class_1_data = np.zeros((SAMPLES_PER_CLASS, LENGTH - TRANSIENT_LENGTH))
    class_1_label = np.ones((SAMPLES_PER_CLASS,1))
    
    
    
    for NUM_TRIALS in range(0, SAMPLES_PER_CLASS):
        random.seed(NUM_TRIALS)
        x_in = np.random.normal(0, 1)
        y_in = np.random.normal(0, 1)
        x = np.zeros(LENGTH)
        y = np.zeros(LENGTH)
    
        noise_x = np.random.normal(0, 1, LENGTH)
        noise_y = np.random.normal(0, 1, LENGTH)
            
        x[0] = x_in
        y[0] = y_in
        x[1] = a1 * x[0] + COEFF * y[0] + e1*noise_x[1] +e2*noise_x[0]
        y[1] = c * y[0] + e1*noise_y[1]
        
    
        for n in range(2, LENGTH):
            y[n] = c * y[n - 1] + e1*noise_y[n] 
            x[n] = a1 * x[n - 1] + a2 * x[n - 2] + COEFF * y[n-1] + e1*noise_x[n] + e2*noise_x[n-1]
        
        
        
         #standardizzo x e y tenendo conto dei massimi e minimi di entrambi, specialmente x
        
        sup = np.max([np.max(x), np.max(y), np.abs(np.min(x)), np.abs(np.min(y))])
        inf = -1*sup
    
        x = (x - inf)/(sup - inf)
        y = (y - inf)/(sup - inf)
        
        
        class_0_data[NUM_TRIALS, :] = y[TRANSIENT_LENGTH:]
        class_1_data[NUM_TRIALS, :] = x[TRANSIENT_LENGTH:]
    
    


    # Applica la trasformazione sia a class_0_data che a class_1_data
    '''class_0_data= np.apply_along_axis(
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)),
        axis=1,
        arr=class_0_data
    )
    class_1_data= np.apply_along_axis(
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)),
        axis=1,
        arr=class_1_data
    )'''
    
    
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
    