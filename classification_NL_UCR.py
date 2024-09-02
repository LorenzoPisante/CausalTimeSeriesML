# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 23:12:21 2024

@author: loren
"""

import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
import ChaosFEX.feature_extractor as CFX
from codes import k_cross_validation_refined_search2

def readucr(filename):
    data = np.loadtxt(PATH + filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

PATH = 'D:/loren/Documents/cod/UCR/'

flist = ['Adiac', 'Beef', 'CBF', 'ChlorineConcentration', 'CinC_ECG_torso', 'Coffee', 'Cricket_X', 'Cricket_Y', 'Cricket_Z', 
'DiatomSizeReduction', 'ECGFiveDays', 'FaceAll', 'FaceFour', 'FacesUCR', '50words', 'FISH', 'Gun_Point', 'Haptics', 
'InlineSkate', 'ItalyPowerDemand', 'Lighting2', 'Lighting7', 'MALLAT', 'MedicalImages', 'MoteStrain', 'NonInvasiveFatalECG_Thorax1', 
'NonInvasiveFatalECG_Thorax2', 'OliveOil', 'OSULeaf', 'SonyAIBORobotSurface', 'SonyAIBORobotSurfaceII', 'StarLightCurves', 'SwedishLeaf', 'Symbols', 
'synthetic_control', 'Trace', 'TwoLeadECG', 'Two_Patterns', 'uWaveGestureLibrary_X', 'uWaveGestureLibrary_Y', 'uWaveGestureLibrary_Z', 'wafer', 'WordsSynonyms', 'yoga']

flist = ['WormsTwoClass']

#ChaosFEX Feature
INA = 0.7
DT = 0.7
EPSILON_1 = 0.04

for each in flist:
    fname = each
    x_train, y_train = readucr(fname+'/'+fname+'_TRAIN')
    x_test, y_test = readucr(fname+'/'+fname+'_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0]/10, 16)
     
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)
     
    y_train=y_train.reshape(y_train.shape + (1,))
#    Y_train = keras.utils.to_categorical(y_train, nb_classes)
#    Y_test = keras.utils.to_categorical(y_test, nb_classes)
     
    #x_train_mean = x_train.mean()
    #x_train_std = x_train.std()
    #x_train = (x_train - x_train_mean)/(x_train_std)
      
    #x_test = (x_test - x_train_mean)/(x_train_std)
    sup = np.max([np.max(x_train), np.max(x_test)])
    inf = np.min([np.min(x_train), np.min(x_test)])
    #x_train = (x_train - x_train.min())/(x_train.max()-x_train.min())
    #x_test = (x_test - x_test.min())/(x_test.max()-x_test.min())
    x_train = (x_train - inf)/(sup-inf)
    x_test = (x_test - inf)/(sup-inf)
    
    datasets= []
    datasets.append((x_train, y_train))

    INA, DT, EPSILON_1, best_values_array=k_cross_validation_refined_search2(datasets, 4)
    
    # ChaosFEX feature Extraction
    feat_mat_traindata = CFX.transform(x_train, INA, 10000, EPSILON_1, DT)
    feat_mat_testdata = CFX.transform(x_test, INA, 10000, EPSILON_1, DT)
    
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = feat_mat_traindata.shape[1]
    NUM_CLASSES = len(np.unique(y_train))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(feat_mat_traindata[(y_train == label)[:,0], :], axis=0)
    
    predicted_label = np.argmax(cosine_similarity(feat_mat_testdata, mean_each_class), axis = 1)

    
    ACC = accuracy_score(y_test, predicted_label)
    #RECALL = recall_score(y_label, predicted_label, average="macro")
    #PRECISION = precision_score(testlabel, predicted_label, average="macro")
    #F1SCORE = f1_score(testlabel, predicted_label, average="macro")
    print("ChaosFEX - Accuracy-score for ",fname ," is ", ACC)
    


from scipy.spatial.distance import euclidean, cityblock, jaccard
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cdist



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
for cic in best_values_array.T:
    INA= cic[1]
    DT= cic[0]
    EPSILON_1= cic[2]
    # ChaosFEX feature Extraction
    feat_mat_traindata = CFX.transform(x_train, INA, 10000, EPSILON_1, DT)
    feat_mat_testdata = CFX.transform(x_test, INA, 10000, EPSILON_1, DT)
    
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = feat_mat_traindata.shape[1]
    NUM_CLASSES = len(np.unique(y_train))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for metric_name, metric_function in metric_functions.items():
        for label in range(0, NUM_CLASSES):
            
            mean_each_class[label, :] = np.mean(feat_mat_traindata[(y_train == label)[:,0], :], axis=0)
        
        # Applica la metrica selezionata
        if metric_name in ['cosine', 'pearson', 'spearman']:
        # Per le metriche di similarità, usa argmax
            similarities = metric_function(feat_mat_testdata, mean_each_class)
            predicted_label = np.argmax(similarities, axis=1)
        else:
        # Per le metriche di distanza, usa argmin
            distances = metric_function(feat_mat_testdata, mean_each_class)
            predicted_label = np.argmin(distances, axis=1)


        
        ACC = accuracy_score(y_test, predicted_label)
        #RECALL = recall_score(y_label, predicted_label, average="macro")
        #PRECISION = precision_score(testlabel, predicted_label, average="macro")
        #F1SCORE = f1_score(testlabel, predicted_label, average="macro")
        if(ACC>0.6):
            #print('INA= ', INA)
            #print('DT= ', DT)
            print('EPSILON= ', EPSILON_1)
            print("ChaosFEX - Accuracy-score for ",fname, "with ", metric_name ," is ", ACC)
        
        

    

        
        