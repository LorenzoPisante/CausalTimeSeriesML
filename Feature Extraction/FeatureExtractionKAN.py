# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:08:13 2024

@author: loren
"""

import os
import numpy as np
import torch
from scipy import io
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
import torch
from kan import KAN
from scipy import io

from scipy.io import savemat


def forward_k(model,k,x):
    model.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
    model.spline_preacts = []
    model.spline_postsplines = []
    model.spline_postacts = []
    model.acts_scale = []
    model.acts_scale_std = []
    # self.neurons_scale = []

    model.acts.append(x)  # acts shape: (batch, width[l])

    for l in range(k):

        x_numerical, preacts, postacts_numerical, postspline = model.act_fun[l](x)

        if model.symbolic_enabled == True:
            x_symbolic, postacts_symbolic = model.symbolic_fun[l](x)
        else:
            x_symbolic = 0.
            postacts_symbolic = 0.

        x = x_numerical + x_symbolic
        postacts = postacts_numerical + postacts_symbolic

        # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
        grid_reshape = model.act_fun[l].grid.reshape(model.width[l + 1], model.width[l], -1)
        input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
        output_range = torch.mean(torch.abs(postacts), dim=0)
        model.acts_scale.append(output_range / input_range)
        model.acts_scale_std.append(torch.std(postacts, dim=0))
        model.spline_preacts.append(preacts.detach())
        model.spline_postacts.append(postacts.detach())
        model.spline_postsplines.append(postspline.detach())

        x = x + model.biases[l].weight
        model.acts.append(x)
    return x


PATH = 'D:/loren/Documents/cod/'
DATA_NAME = 'SKEW-TENTMAP-EQ4'  # nome dataset in cod
FOLDER_NAME = 'tentmap4'  # nome in exp-report per pesi
Folder = 'C:/Users/loren/Desktop/EXP_REPORT/KAN/' + FOLDER_NAME

VAR = 100
LEN_VAL = 100

COUP_COEFF1 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Sync_Error = np.zeros(len(COUP_COEFF1))


ROW = -1
for COUP_COEFF in COUP_COEFF1:
    ROW = ROW + 1
    RESULT_PATH = PATH + '/DATA/' + DATA_NAME + '/' + str(COUP_COEFF) + '/'
    # Loading the normalized Raw Data
    Y_independent_data = io.loadmat(RESULT_PATH + 'Y_independent_data_class_0.mat')
    Y_independent_label = io.loadmat(RESULT_PATH + 'Y_independent_label_class_0.mat')
    
    X_dependent_data = io.loadmat(RESULT_PATH + 'X_dependent_data_class_1.mat')
    X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')
    
    M_data = torch.tensor(Y_independent_data['class_0_indep_raw_data'][:VAR, :LEN_VAL], dtype=torch.float32)
    class_0_label = torch.tensor(Y_independent_label['class_0_indep_raw_data_label'][:VAR, :LEN_VAL], dtype=torch.float32)
    S_data = torch.tensor(X_dependent_data['class_1_dep_raw_data'][:VAR, :LEN_VAL], dtype=torch.float32)
    class_1_label = torch.tensor(X_dependent_label['class_1_dep_raw_data_label'][:VAR, :LEN_VAL], dtype=torch.float32)

    # Load the entire model
    model = KAN(width=[100,50,50,50,2], grid=3, k=3)
    Name = 'KANKaggle' + str(ROW) + '.pt'
    model.load_ckpt(Name , Folder)
    
    
    
    dataset_M = {
                "test_input": M_data,
                "test_label": class_0_label,
            }
    
    dataset_S = {
                "test_input": S_data,
                "test_label": class_1_label,
            }
    
    M_KAN1 = forward_k(model,1,dataset_M['test_input'])
    M_KAN2 = forward_k(model,2,dataset_M['test_input'])
    M_KAN3 = forward_k(model,3,dataset_M['test_input'])
    S_KAN1 = forward_k(model,1,dataset_S['test_input'])
    S_KAN2 = forward_k(model,2,dataset_S['test_input'])
    S_KAN3 = forward_k(model,3,dataset_S['test_input'])
    
    #Salvataggio per Matlab
    Save_PATH = 'D:/loren/Documents/cod/'
    Feature1 = '/KAN1/'
    
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature1 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    #savemat(RESULT_Save_PATH + 'M.mat', {'M_DL1': M_KAN1})
    savemat(RESULT_Save_PATH + 'M.mat', {'M_DL1': M_KAN1.detach().numpy()})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_DL1': S_KAN1.detach().numpy()})
    
    Feature1 = '/KAN2/'
    
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature1 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_DL2': M_KAN2.detach().numpy()})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_DL2': S_KAN2.detach().numpy()})
    
    Feature1 = '/KAN3/'
    
    RESULT_Save_PATH = Save_PATH + '/DATA/'  + DATA_NAME + '/Feature/' + Feature1 + '/' + str(COUP_COEFF) +'/'
    try:
        os.makedirs(RESULT_Save_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_Save_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_Save_PATH)
    savemat(RESULT_Save_PATH + 'M.mat', {'M_DL3': M_KAN3.detach().numpy()})
    savemat(RESULT_Save_PATH + 'S.mat', {'S_DL3': S_KAN3.detach().numpy()})
    

    
   
