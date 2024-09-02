# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 07:00:25 2023

@author: loren
"""

from scipy import io

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
'''
##kvalidation
COUP_COEFF1 =  np.array([0.2, 0.4, 0.6])
PATH = 'D:/loren/Documents/cod/'
DATA_NAME = 'SKEW-TENTMAP2'
datasets = []

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
    
    total_data = np.concatenate((M_data, S_data))
    total_label = np.concatenate((class_0_label, class_1_label))
    traindata, testdata, trainlabel, testlabel = train_test_split(total_data, total_label, test_size=0.2, random_state=42)
    datasets.append((traindata, trainlabel))

best_INA, best_DT, best_EPSILON, best_values_array=k_cross_validation_refined_search2(datasets, 4)'''

#back to reality
COUP_COEFF1 =  np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#COUP_COEFF1 =  np.array([0.0])

Sync_Error = np.zeros(len(COUP_COEFF1))
F1_score_Result_array = np.zeros(len(COUP_COEFF1))

PATH = 'D:/loren/Documents/cod/'
DATA_NAME = 'SKEW-TENTMAP-EQ4'

#tentmap= SKEW-TENTMAP-EQ
#arma= ARIMA2_N

#ChaosFEX Feature  #BEST PAR PER TENT MAP 2
#INA = best_INA   #0.48
#DT = best_DT   #0.26
#EPSILON_1 = best_EPSILON  #0.11
#per arima2 0.78,0.59,0.11 per cosine
#per arima 2 per manhattan uso 0.78,0.59,0.09

#ChaosFEX Feature
INA = 0.48
DT = 0.26
EPSILON_1 = 0.11
best_metric='cosine'



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
    
    traindata_M, testdata_M, trainlabel_M, testlabel_M = train_test_split(M_data, class_0_label, test_size=0.96, random_state=424)
    traindata_S, testdata_S, trainlabel_S, testlabel_S = train_test_split(S_data, class_1_label, test_size=0.96, random_state=424)
    
    traindata=np.concatenate((traindata_M, traindata_S))
    testdata=np.concatenate((testdata_M, testdata_S))
    trainlabel=np.concatenate((trainlabel_M, trainlabel_S))
    testlabel=np.concatenate((testlabel_M, testlabel_S))
    
    traindata, trainlabel = shuffle(traindata, trainlabel, random_state=42)
    
    """#tentativo di shuffle del test datset in maniera ordinata
    lun = int(np.shape(testdata)[0]/2)
    for i in range(lun):
        ran=random.random()
        
        if(ran>0.5):
            # Scambia le righe
            testdata[[[i,i+lun]]]=testdata[[i+lun,i]]
            testlabel[[[i,i+lun]]]=testlabel[[i+lun,i]]"""
    
    
    
    # ChaosFEX feature Extraction
    feat_mat_traindata = CFX.transform(traindata, INA, 10000, EPSILON_1, DT)
    feat_mat_testdata = CFX.transform(testdata, INA, 10000, EPSILON_1, DT)
    

    
    
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = feat_mat_traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(feat_mat_traindata[(trainlabel == label)[:,0], :], axis=0)
    # Seleziona la metrica di similarità o distanza in base a metric_name
    metric_function = metric_functions[best_metric]
    
    if best_metric in ['cosine', 'pearson', 'spearman']:
    # Per le metriche di similarità, usa argmax
        similarities = metric_function(feat_mat_testdata, mean_each_class)
        scores=similarities
    else:
    # Per le metriche di distanza, usa argmin
        distances = metric_function(feat_mat_testdata, mean_each_class)
        distances[:, [0, 1]] = distances[:, [1, 0]]
        scores=distances
    
    y_pred_testdata = np.zeros(np.shape(testdata)[0])    
    lun = int(np.shape(testdata)[0]/2)    

    
    
    for i in range (lun):

        
        if (abs(scores[i,0]-scores[i,1]) > abs(scores[i+lun,0] - scores[i+lun,1])):
            if(scores[i,0]-scores[i,1] > 0):
                y_pred_testdata[i] = 0
                y_pred_testdata[i+lun] = 1
            else:
                y_pred_testdata[i] = 1
                y_pred_testdata[i+lun] = 0
        elif (abs(scores[i,0]-scores[i,1]) < abs(scores[i+lun,0] - scores[i+lun,1])):
            if(scores[i+lun,0]-scores[i+lun,1] < 0):
                y_pred_testdata[i] = 0
                y_pred_testdata[i+lun] = 1
            else:
                y_pred_testdata[i] = 1
                y_pred_testdata[i+lun] = 0
        elif(abs(scores[i,0]-scores[i,1]) == abs(scores[i+lun,0] - scores[i+lun,1])):
            if(random.random()>0.5):
                y_pred_testdata[i] = 0
                y_pred_testdata[i+lun] = 1
            else:
                y_pred_testdata[i] = 1
                y_pred_testdata[i+lun] = 0
    
    ACC = accuracy_score(testlabel, y_pred_testdata)
    RECALL = recall_score(testlabel, y_pred_testdata, average="macro")
    PRECISION = precision_score(testlabel, y_pred_testdata, average="macro")
    F1SCORE = f1_score(testlabel, y_pred_testdata, average="macro")
    print("ChaosFEX - F1-score for COUP_COEFF =  ",COUP_COEFF,"is", F1SCORE)
    
    F1_score_Result_array[ROW] = ACC 
    
    # Final Result Plot
plt.figure(figsize=(15,10))
plt.plot(COUP_COEFF1,F1_score_Result_array, '-*b', markersize = 10, label = "Accuracy score")
#plt.plot(COUP_COEFF1,Sync_Error, '-or', markersize = 10, label = "MSE")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=30)
plt.ylabel('Accuracy score', fontsize=30)
plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
#plt.savefig(rf"C:\Users\loren\Desktop\EXP_REPORT\NL\generalization\fewshot40_NL_TM", format='png', dpi=300)
#plt.savefig(RESULT_PATH_FINAL+"/Chaosnet-"+DATA_NAME+"-F1_score_vs_coup_coeff.jpg", format='jpg', dpi=300)
#plt.savefig(RESULT_PATH_FINAL+"/Chaosnet-"+DATA_NAME+"-F1_score_vs_coup_coeff.eps", format='eps', dpi=300)
plt.show()
np.save(rf"C:\Users\loren\Desktop\EXP_REPORT\NL\generalization\fewshot40_TM_1", F1_score_Result_array)    
