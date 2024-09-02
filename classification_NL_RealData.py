# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:46:40 2023

@author: loren
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
import ChaosFEX.feature_extractor as CFX
from codes import k_cross_validation_refined_search2

PATH = 'D:/loren/Documents/cod/RealData/'
DATA_NAME='pair0050.txt'

#TRUTH= 'X->Y'
TRUTH= 'Y->X'

#ChaosFEX Feature
INA = 0.48
DT = 0.26
EPSILON_1 = 0.11


RESULT_PATH= PATH + DATA_NAME

# Legge il file e converte i dati in un numpy array
# Assumendo che i dati siano separati da spazi o tabulazioni
data = np.loadtxt(RESULT_PATH, delimiter='\t')

# Separare i dati nelle colonne X e Y
X = data[:, 0]
Y = data[:, 1]

X = X.reshape(1, -1)
Y = Y.reshape(1, -1)

length = Y.shape[1]

print(np.shape(X)[1])

X = (X - np.min(X)) / (np.max(X) - np.min(X))
Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

# Assicurati che la lunghezza di X sia pari
if length % 2 != 0:
    X = X[:, :-1]  # Rimuovi l'ultimo elemento se la lunghezza è dispari

# Dividi Y in due matrici Y1 e Y2 di dimensioni uguali
half_length = X.shape[1] // 2
X_train = X[:, :half_length]
X_test = X[:, half_length:]


# Assicurati che la lunghezza di Y sia pari
if length % 2 != 0:
    Y = Y[:, :-1]  # Rimuovi l'ultimo elemento se la lunghezza è dispari

# Dividi Y in due matrici Y1 e Y2 di dimensioni uguali
half_length = Y.shape[1] // 2
Y_train = Y[:, :half_length]
Y_test = Y[:, half_length:]



total_data = np.concatenate((X_train, Y_train))
testdata = np.concatenate((X_test, Y_test))
if (TRUTH=='X->Y'):
    class_0_label=np.zeros((np.shape(X)[0],1))
    class_1_label=np.ones((np.shape(Y)[0],1))
    total_label = np.concatenate((class_0_label, class_1_label))
else:
    class_0_label=np.zeros((np.shape(Y)[0],1))
    class_1_label=np.ones((np.shape(X)[0],1))
    total_label = np.concatenate((class_1_label, class_0_label))
        
#traindata, testdata, trainlabel, testlabel = train_test_split(total_data, total_label, test_size=0.2, random_state=42)
testlabel=total_label
#datasets = []
#datasets.append((traindata,trainlabel))
#INA, DT, EPSILON_1, values= best_values_array=k_cross_validation_refined_search2(datasets, 4)

# ChaosFEX feature Extraction
feat_mat_traindata = CFX.transform(total_data, INA, 10000, EPSILON_1, DT)
feat_mat_testdata = CFX.transform(testdata, INA, 10000, EPSILON_1, DT)


from sklearn.metrics.pairwise import cosine_similarity
NUM_FEATURES = feat_mat_traindata.shape[1]
NUM_CLASSES = len(np.unique(total_label))
#NUM_CLASSES=2
mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
for label in range(0, NUM_CLASSES):
    
    
        
    mean_each_class[label, :] = np.mean(feat_mat_traindata[(total_label == label)[:,0], :], axis=0)
        
predicted_label = np.argmax(cosine_similarity(feat_mat_testdata, mean_each_class), axis = 1)
    
    
ACC = accuracy_score(testlabel, predicted_label)*100
RECALL = recall_score(testlabel, predicted_label, average="macro")
PRECISION = precision_score(testlabel, predicted_label, average="macro")
F1SCORE = f1_score(testlabel, predicted_label, average="macro")
print("ChaosFEX - F1-score is ", F1SCORE)
print("ChaosFEX - Accuracy is ", ACC)
