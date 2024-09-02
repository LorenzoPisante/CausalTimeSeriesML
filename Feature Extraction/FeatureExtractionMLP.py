# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:49:16 2024

@author: loren
"""

import os
import numpy as np
# import scipy
from scipy import io
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
# from sklearn import datasets
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
# from Codes import chaosnet, k_cross_validation
from sklearn.model_selection import train_test_split
# from ETC import CCC

from keras.models import Sequential
from keras.layers import Dense, Convolution1D, Dropout, Flatten, MaxPooling1D

from scipy import io

from scipy.io import savemat
from keras import backend

PATH = 'D:/loren/Documents/cod/'
DATA_NAME = 'ARIMA2_N'  #nome dataset in cod
FOLDER_NAME = 'arima-500'  #nome in exp-report per pesi

VAR = 500
LEN_VAL = 2000

# CNN architecture details
input_dim = 2000
out_dim = 2
batch_size_val = 32
epochs_val = 1000


COUP_COEFF1 =  np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Sync_Error = np.zeros(len(COUP_COEFF1))
F1_score_Result_array = np.zeros(len(COUP_COEFF1))







ROW = -1
for COUP_COEFF in COUP_COEFF1:
    

    
    ROW = ROW+1
    RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' + str(COUP_COEFF) +'/'
    # Loading the normalized Raw Data
    Y_independent_data = io.loadmat(RESULT_PATH + 'Y_independent_data_class_0.mat')
    Y_independent_label = io.loadmat(RESULT_PATH + 'Y_independent_label_class_0.mat')
    
    X_dependent_data = io.loadmat(RESULT_PATH + 'X_dependent_data_class_1.mat' )
    X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')
    
    M_data = Y_independent_data['class_0_indep_raw_data'][:VAR]
    class_0_label = Y_independent_label['class_0_indep_raw_data_label'][:VAR]
    S_data = X_dependent_data['class_1_dep_raw_data'][:VAR]
    class_1_label = X_dependent_label['class_1_dep_raw_data_label'][:VAR]    

    
  
   

    # model.add(Dense(65, activation='relu'))
   

    model = Sequential()
    model.add(Dropout(0.1, input_shape=(input_dim,)))
    layer_1 = model.add(Dense(500, activation='relu', name="layer_1"))
    model.add(Dropout(0.2))
    layer_2 = model.add(Dense(500, activation='relu', name="layer_2"))
    model.add(Dropout(0.2))
    layer_3 = model.add(Dense(500, activation='relu', name="layer_3"))
    model.add(Dropout(0.3))
    layer_4 = model.add(Dense(out_dim, activation='softmax', name="layer_4"))


    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #checkpointer = callbacks.ModelCheckpoint(filepath=TRAIN_DATA_PATH + "checkpoint.hdf5", verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)
    model.load_weights('C:/Users/loren/Desktop/EXP_REPORT/MLP' + '/' + FOLDER_NAME +'/' +'MLPKaggle'+ str(ROW) +'.hdf5')
    
    
    from keras import backend as K

    # with a Sequential model
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[0].output])
    M_DL0 = get_3rd_layer_output([M_data])[0]
    S_DL0 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[1].output])
    M_DL1 = get_3rd_layer_output([M_data])[0]
    S_DL1 = get_3rd_layer_output([S_data])[0]
    
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[3].output])
    M_DL2 = get_3rd_layer_output([M_data])[0]
    S_DL2 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[5].output])
    M_DL3 = get_3rd_layer_output([M_data])[0]
    S_DL3 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[7].output])
    M_DL4 = get_3rd_layer_output([M_data])[0]
    S_DL4 = get_3rd_layer_output([S_data])[0]
    
    #Salvataggio per Matlab
    Save_PATH = 'D:/loren/Documents/cod/'
    Feature1 = '/DL1/'
    
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature1 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_DL1': M_DL1})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_DL1': S_DL1})
    
    Feature1 = '/DL2/'
    
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature1 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_DL2': M_DL2})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_DL2': S_DL2})
    
    Feature1 = '/DL3/'
    
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature1 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_DL3': M_DL3})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_DL3': S_DL3})
    
    Feature1 = '/DL4/'
    
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature1 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_DL4': M_DL4})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_DL4': S_DL4})
    
    
       