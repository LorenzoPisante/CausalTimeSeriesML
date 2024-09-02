from scipy import io
import sklearn
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
import ChaosFEX.feature_extractor as CFX
from sklearn.model_selection import train_test_split
from codes import k_cross_validation_refined_search2
from sklearn.utils import shuffle
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
import random

def pearson_similarity(X, Y):
    similarities = np.array([pearsonr(x, y)[0] for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return similarities

def spearman_similarity(X, Y):
    similarities = np.array([spearmanr(x, y)[0] for x in X for y in Y]).reshape(X.shape[0], Y.shape[0])
    return similarities

metric_functions = {
    'euclidean': lambda X, Y: cdist(X, Y, 'euclidean'),
    'manhattan': lambda X, Y: cdist(X, Y, 'cityblock'),
    'jaccard': lambda X, Y: cdist(X, Y, 'jaccard'),
    'pearson': pearson_similarity,
    'spearman': spearman_similarity,
    'cosine': cosine_similarity
}

def remove_random_points(data, percentage):
    data_modified = data.copy()
    num_points_to_remove = int(len(data) * percentage)
    indices_to_remove = random.sample(range(len(data)), num_points_to_remove)
    data_modified[indices_to_remove] = 0
    return data_modified

VAR = 1000
LEN_VAL = 2000

COUP_COEFF1 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

PATH = 'D:/loren/Documents/cod/'
DATA_NAME = 'SKEW-TENTMAP-EQ4'

INA = 0.48
DT = 0.26
EPSILON_1 = 0.11
best_metric = 'cosine'
percentages_to_remove = [0.1, 0.2, 0.3]  # 10%, 20% e 30%

results = {}

for percentage_to_remove in percentages_to_remove:
    F1_score_Result_array = np.zeros(len(COUP_COEFF1))
    
    ROW = -1
    for COUP_COEFF in COUP_COEFF1:
        ROW = ROW + 1
        RESULT_PATH = PATH + '/DATA/' + DATA_NAME + '/' + str(COUP_COEFF) + '/'
        Y_independent_data = io.loadmat(RESULT_PATH + 'Y_independent_data_class_0.mat')
        Y_independent_label = io.loadmat(RESULT_PATH + 'Y_independent_label_class_0.mat')
        
        X_dependent_data = io.loadmat(RESULT_PATH + 'X_dependent_data_class_1.mat')
        X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')
        
        M_data = Y_independent_data['class_0_indep_raw_data'][:VAR]
        class_0_label = Y_independent_label['class_0_indep_raw_data_label'][:VAR]
        S_data = X_dependent_data['class_1_dep_raw_data'][:VAR]
        class_1_label = X_dependent_label['class_1_dep_raw_data_label'][:VAR]
        
        # Rimuovi punti casuali dalle serie temporali
        M_data = np.array([remove_random_points(series, percentage_to_remove) for series in M_data])
        S_data = np.array([remove_random_points(series, percentage_to_remove) for series in S_data])
        
        traindata_M, testdata_M, trainlabel_M, testlabel_M = train_test_split(M_data, class_0_label, test_size=0.3, random_state=424)
        traindata_S, testdata_S, trainlabel_S, testlabel_S = train_test_split(S_data, class_1_label, test_size=0.3, random_state=424)
        
        traindata = np.concatenate((traindata_M, traindata_S))
        testdata = np.concatenate((testdata_M, testdata_S))
        trainlabel = np.concatenate((trainlabel_M, trainlabel_S))
        testlabel = np.concatenate((testlabel_M, testlabel_S))
        
        traindata, trainlabel = shuffle(traindata, trainlabel, random_state=42)
        
        feat_mat_traindata = CFX.transform(traindata, INA, 10000, EPSILON_1, DT)
        feat_mat_testdata = CFX.transform(testdata, INA, 10000, EPSILON_1, DT)

        NUM_FEATURES = feat_mat_traindata.shape[1]
        NUM_CLASSES = len(np.unique(trainlabel))
        mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
        
        for label in range(0, NUM_CLASSES):
            mean_each_class[label, :] = np.mean(feat_mat_traindata[(trainlabel == label)[:,0], :], axis=0)
        
        metric_function = metric_functions[best_metric]
        
        if best_metric in ['cosine', 'pearson', 'spearman']:
            similarities = metric_function(feat_mat_testdata, mean_each_class)
            scores = similarities
        else:
            distances = metric_function(feat_mat_testdata, mean_each_class)
            distances[:, [0, 1]] = distances[:, [1, 0]]
            scores = distances
        
        y_pred_testdata = np.zeros(np.shape(testdata)[0])
        lun = int(np.shape(testdata)[0] / 2)
        
        for i in range(lun):
            if abs(scores[i, 0] - scores[i, 1]) > abs(scores[i+lun, 0] - scores[i+lun, 1]):
                if scores[i, 0] - scores[i, 1] > 0:
                    y_pred_testdata[i] = 0
                    y_pred_testdata[i+lun] = 1
                else:
                    y_pred_testdata[i] = 1
                    y_pred_testdata[i+lun] = 0
            elif abs(scores[i, 0] - scores[i, 1]) < abs(scores[i+lun, 0] - scores[i+lun, 1]):
                if scores[i+lun, 0] - scores[i+lun, 1] < 0:
                    y_pred_testdata[i] = 0
                    y_pred_testdata[i+lun] = 1
                else:
                    y_pred_testdata[i] = 1
                    y_pred_testdata[i+lun] = 0
            elif abs(scores[i, 0] - scores[i, 1]) == abs(scores[i+lun, 0] - scores[i+lun, 1]):
                if random.random() > 0.5:
                    y_pred_testdata[i] = 0
                    y_pred_testdata[i+lun] = 1
                else:
                    y_pred_testdata[i] = 1
                    y_pred_testdata[i+lun] = 0
        
        ACC = accuracy_score(testlabel, y_pred_testdata)
        RECALL = recall_score(testlabel, y_pred_testdata, average="macro")
        PRECISION = precision_score(testlabel, y_pred_testdata, average="macro")
        F1SCORE = f1_score(testlabel, y_pred_testdata, average="macro")
        print(f"ChaosFEX - F1-score for COUP_COEFF = {COUP_COEFF} with {int(percentage_to_remove*100)}% points removed is {F1SCORE}")
        
        F1_score_Result_array[ROW] = ACC
    
    results[percentage_to_remove] = F1_score_Result_array

# Plotting the results
plt.figure(figsize=(15, 10))

for percentage_to_remove in percentages_to_remove:
    plt.plot(COUP_COEFF1, results[percentage_to_remove], '-*', markersize=10, label=f"Accuracy score ({int(percentage_to_remove*100)}% points removed)")

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=30)
plt.ylabel('Accuracy score', fontsize=30)
plt.ylim(0, 1.1)
plt.legend(fontsize=20)
plt.tight_layout()
plt.show()


percentages = sorted(results.keys())
num_percentages = len(percentages)
num_coeffs = len(COUP_COEFF1)

results_matrix = np.zeros((num_percentages, num_coeffs))

for i, percentage in enumerate(percentages):
    results_matrix[i, :] = results[percentage]

# Save the results
#for percentage_to_remove in percentages_to_remove:
np.save(rf"C:\Users\loren\Desktop\EXP_REPORT\NL\generalization\point_remove_NL_TM_5", results_matrix)

