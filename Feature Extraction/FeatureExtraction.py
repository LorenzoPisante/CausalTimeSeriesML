# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 02:24:39 2023

@author: loren


METTI 

-NOME DATA
-I SUOI HYPERPARAMETRI NL TROVATI IN TUNING
-IL SUO MODELLO DL TROVATO IN TUNING
"""



from scipy import io
from scipy.io import savemat
import sklearn
import os
import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
# from sklearn import datasets
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
# from Codes import chaosnet, k_cross_validation
import ChaosFEX.feature_extractor as CFX
from sklearn.model_selection import train_test_split
from codes import k_cross_validation_refined_search2

import keras
from kerastuner.tuners import Hyperband
from kerastuner import HyperModel
from keras import callbacks
from keras import backend
from classification_DL_wTuning  import MyHyperModel
from classification_ResNet import build_resnet
#from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from scipy.spatial.distance import euclidean, cityblock, jaccard
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import random

def pearson_similarity(X, Y):
    # Assicurati che X e Y siano array 2D
    similarities = np.array([pearsonr(x, y)[0] for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return similarities

def spearman_similarity(X, Y):
    similarities = np.array([spearmanr(x, y)[0] for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return similarities

# Modifica il dizionario delle funzioni per utilizzare cdist per le metriche di distanza
metric_functions = {
    'euclidean': lambda X, Y: cdist(X, Y, 'euclidean'),
    'manhattan': lambda X, Y: cdist(X, Y, 'cityblock'),
    'jaccard': lambda X, Y: cdist(X, Y, 'jaccard'),
    'pearson': pearson_similarity,
    'spearman': spearman_similarity,
    'cosine': cosine_similarity
}

VAR = 1000
LEN_VAL = 2000


#back to reality
COUP_COEFF1 =  np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Sync_Error = np.zeros(len(COUP_COEFF1))
F1_score_Result_array = np.zeros(len(COUP_COEFF1))

PATH = 'D:/loren/Documents/cod/'
DATA_NAME = 'SKEW-TENTMAP-EQ'


#ChaosFEX Feature  #best hyper cross val TM2 (INA=0.48,DT=0.26,EPS=0.11)
#best hyper per ARIMA2(INA=0.78,DT=0.59, EPS=0.14) per cosine almeno

#ora per arima 2 sono nel mood con manhattan con 0.78,0.59,0.09

INA = 0.48
DT = 0.26
EPSILON_1 = 0.11
best_metric='manhattan'





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

    # ChaosFEX feature Extraction
    M_transf=CFX.transform(M_data, INA, 10000, EPSILON_1, DT)
    S_transf=CFX.transform(S_data, INA, 10000, EPSILON_1, DT)
    
    #Extraction Firing Time
    M_FT=M_transf[:,(2*LEN_VAL):(3*LEN_VAL)]
    S_FT=S_transf[:,(2*LEN_VAL):(3*LEN_VAL)]
        
    #extraction Firing Rate 
    M_FR=M_transf[:,:LEN_VAL]
    S_FR=S_transf[:,:LEN_VAL]
    
    #Extraction Energy
    M_En=M_transf[:,(LEN_VAL):(2*LEN_VAL)]
    S_En=S_transf[:,(LEN_VAL):(2*LEN_VAL)]
    
    #Extraction Entropy
    
    M_H=M_transf[:,(3*LEN_VAL):(4*LEN_VAL)]
    S_H=S_transf[:,(3*LEN_VAL):(4*LEN_VAL)]
    
    
    #Salvataggio per Matlab
    Save_PATH = 'D:/loren/Documents/cod/'
    Feature1 = '/Firing Rate/'
    
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature1 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_FR': M_FR})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_FR': S_FR})
    
    Feature2= '/Firing Time/'
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature2 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_FT': M_FT})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_FT': S_FT})
    
    Feature= '/Energy/'
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_En': M_En})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_En': S_En})
    
    Feature= '/Entropy/'
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_H': M_H})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_H': S_H})
    
    
        
""" ##DL
    input_dim = 2000
    out_dim = 2
    
    hypermodel = MyHyperModel(input_dim=input_dim, out_dim=out_dim)
    
    tuner = Hyperband(
       hypermodel,
       objective='val_accuracy',
       max_epochs=20,  # Adatta questo valore in base alle esigenze
       hyperband_iterations=1,  # Imposta a 1 per una ricerca più simile alla Grid Search
       directory='D:/loren/Documents/cod/RealData/',
       project_name='Tuner_TM2_Grid'
    )
    
    model = tuner.get_best_models(num_models=1)[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[0].output])
    M_DL1 = get_3rd_layer_output([M_data])[0]
    S_DL1 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[1].output])
    M_DL2 = get_3rd_layer_output([M_data])[0]
    S_DL2 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[2].output])
    M_DL3 = get_3rd_layer_output([M_data])[0]
    S_DL3 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[3].output])
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
    """
'''    

    ##RESNET

    METHOD = 'ResNet'

    
    weights_PATH = PATH + '/DATA/'  + DATA_NAME + '/' + METHOD + '/ResNetKaggle' + str(ROW) +'.hdf5'
    
    nb_classes=2
    x , y = build_resnet((LEN_VAL,1), 64, nb_classes)
    model = keras.models.Model(inputs=x, outputs=y)
    

    model.load_weights(weights_PATH)
    
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[0].output])
    M_DL1 = get_3rd_layer_output([M_data])[0]
    S_DL1 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[1].output])
    M_DL2 = get_3rd_layer_output([M_data])[0]
    S_DL2 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[2].output])
    M_DL3 = get_3rd_layer_output([M_data])[0]
    S_DL3 = get_3rd_layer_output([S_data])[0]
    get_3rd_layer_output = backend.function([model.layers[0].input],
                                            [model.layers[31].output])
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


import os

if os.path.exists(weights_PATH):
    print("Il file esiste.")
else:
    print("Il file non esiste. Verifica il percorso.")
    
def build_resnet(input_shape, n_feature_maps, nb_classes):
    print('build conv_x')
    x = keras.layers.Input(shape=(input_shape))
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv1D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    
    print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    
    print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps, 1, 1, padding='same')(x)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
    
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps*2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps*2, 1, 1,padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
     
    print ('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps*2, 8, 1, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
     
    print ('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps*2, 5, 1, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
     
    print ('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps*2, 3, 1, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps*2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps*2, 1, 1,padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    print ('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
    
    full = keras.layers.GlobalAveragePooling1D()(y)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    print('        -- model was built.')
    #return keras.models.Model(inputs=x, outputs=out)
    return x, out



'''







