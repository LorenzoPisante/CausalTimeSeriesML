# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:01:51 2024

@author: loren
"""

def chaosnetN(traindata, trainlabel, testdata, metric_name):
    
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import cdist
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
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
    
    NUM_FEATURES = traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(traindata[(trainlabel == label)[:,0], :], axis=0)
     
    # Seleziona la metrica di similarità o distanza in base a metric_name
    metric_function = metric_functions[metric_name]
    
    if metric_name in ['cosine', 'pearson', 'spearman']:
    # Per le metriche di similarità, usa argmax
        similarities = metric_function(testdata, mean_each_class)
        predicted_label = np.argmax(similarities, axis=1)
    else:
    # Per le metriche di distanza, usa argmin
        distances = metric_function(testdata, mean_each_class)
        predicted_label = np.argmin(distances, axis=1)

    return mean_each_class, predicted_label


def k_cross_validation_refined_search2(datasets, FOLD_NO=4):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.model_selection import KFold
    import ChaosFEX.feature_extractor as CFX
    from codesN import chaosnetN
    from sklearn.model_selection import StratifiedKFold
    from scipy.spatial.distance import euclidean, cityblock, jaccard
    from scipy.stats import pearsonr, spearmanr
    from scipy.spatial.distance import cdist
    from sklearn.metrics.pairwise import cosine_similarity



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
    
    
    tolerance=0.015
    INITIAL_NEURAL_ACTIVITY_coarse = np.arange(0.1, 0.8, 0.1)
    DISCRIMINATION_THRESHOLD_coarse = np.arange(0.1, 0.8, 0.1)
    EPSILON_coarse = np.arange(0.01, 0.10, 0.01)
    
    def run_validation(INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON, metric_name):
        total_ACCSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
        for q,dataset in enumerate(datasets):
            traindata, trainlabel = dataset
            ACCSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
            KF = StratifiedKFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
            KF.get_n_splits(traindata)

            for i, DT in enumerate(DISCRIMINATION_THRESHOLD):               
                for j, INA in enumerate(INITIAL_NEURAL_ACTIVITY):
                    for k, EPSILON_1 in enumerate(EPSILON):
                        ACCSCORE_TEMP = []
                        #ACCSCORE = np.zeros((DISCRIMINATION_THRESHOLD.shape[0], INITIAL_NEURAL_ACTIVITY.shape[0],EPSILON.shape[0] ))

                        for TRAIN_INDEX, VAL_INDEX in KF.split(traindata,trainlabel):
                            X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                            Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]

                            FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                            FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)

                            _, Y_PRED = chaosnetN(FEATURE_MATRIX_TRAIN, Y_TRAIN, FEATURE_MATRIX_VAL, metric_name)
                            ACC = accuracy_score(Y_VAL, Y_PRED)
                            ACCSCORE_TEMP.append(ACC)

                        ACCSCORE[i, j, k] = np.mean(ACCSCORE_TEMP)
            total_ACCSCORE += ACCSCORE        
        return total_ACCSCORE
    
    best_ACC=0
    
    for metric_name, metric_function in metric_functions.items():
        ACC_score = run_validation(INITIAL_NEURAL_ACTIVITY_coarse, DISCRIMINATION_THRESHOLD_coarse, EPSILON_coarse,metric_name)
        val_max = np.max(ACC_score)
        if(val_max==1):
            print(metric_name, "ce l'ha pure")
        if(val_max>best_ACC):
            best_ACC=val_max
            best_metric=metric_name
            max_indices = np.unravel_index(np.argmax(ACC_score), ACC_score.shape)
        else:
            print(metric_name, val_max)
    
    best_DT= DISCRIMINATION_THRESHOLD_coarse[max_indices[0]]
    best_INA= INITIAL_NEURAL_ACTIVITY_coarse[max_indices[1]]
    best_EPSILON= EPSILON_coarse[max_indices[2]]
    
    print(best_INA)
    print(best_DT)
    print(best_EPSILON)
    print(best_metric)
    
    return best_INA, best_DT, best_EPSILON, best_metric












def chaosnetNARIMA(traindata, trainlabel, testdata, metric_name):
    
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import cdist
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
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
    
    NUM_FEATURES = traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(traindata[(trainlabel == label)[:,0], :], axis=0)
     
    # Seleziona la metrica di similarità o distanza in base a metric_name
    metric_function = metric_functions[metric_name]
    
    if metric_name in ['cosine', 'pearson', 'spearman']:
    # Per le metriche di similarità, usa argmax
        similarities = metric_function(testdata, mean_each_class)
        scores=similarities
    else:
    # Per le metriche di distanza, usa argmin
        distances = metric_function(testdata, mean_each_class)
        scores=distances
    
    predicted_label = np.zeros(np.shape(testdata)[0])    
    lun = int(np.shape(testdata)[0]/2)
    for i in range (lun):

        
        if (abs(scores[i,0]-scores[i,1])>abs(scores[i+lun,0] - scores[i+lun,1])):
            if(scores[i,0]-scores[i,1]>0):
                predicted_label[i]=0
                predicted_label[i+lun]=1
            else:
               predicted_label[i]=1
               predicted_label[i+lun]=0
        else:
            if(scores[i+lun,0]-scores[i+lun,1]<0):
                predicted_label[i]=0
                predicted_label[i+lun]=1
            else:
               predicted_label[i]=1
               predicted_label[i+lun]=0    

    return mean_each_class, predicted_label


def k_cross_validation_ARIMA(datasets, FOLD_NO=4):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.model_selection import KFold
    import ChaosFEX.feature_extractor as CFX
    from codesN import chaosnetN
    from sklearn.model_selection import StratifiedKFold
    from scipy.spatial.distance import euclidean, cityblock, jaccard
    from scipy.stats import pearsonr, spearmanr
    from scipy.spatial.distance import cdist
    from sklearn.metrics.pairwise import cosine_similarity



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
    
    
    tolerance=0.015
    INITIAL_NEURAL_ACTIVITY_coarse = np.arange(0.1, 0.8, 0.1)
    DISCRIMINATION_THRESHOLD_coarse = np.arange(0.1, 0.8, 0.1)
    EPSILON_coarse = np.arange(0.01, 0.10, 0.01)
    
    def run_validation(INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON, metric_name):
        total_ACCSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
        for q,dataset in enumerate(datasets):
            traindata, trainlabel = dataset
            ACCSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
            KF = StratifiedKFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
            KF.get_n_splits(traindata)

            for i, DT in enumerate(DISCRIMINATION_THRESHOLD):               
                for j, INA in enumerate(INITIAL_NEURAL_ACTIVITY):
                    for k, EPSILON_1 in enumerate(EPSILON):
                        ACCSCORE_TEMP = []
                        #ACCSCORE = np.zeros((DISCRIMINATION_THRESHOLD.shape[0], INITIAL_NEURAL_ACTIVITY.shape[0],EPSILON.shape[0] ))

                        for TRAIN_INDEX, VAL_INDEX in KF.split(traindata,trainlabel):
                            X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                            Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]

                            FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                            FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)

                            _, Y_PRED = chaosnetNARIMA(FEATURE_MATRIX_TRAIN, Y_TRAIN, FEATURE_MATRIX_VAL, metric_name)
                            ACC = accuracy_score(Y_VAL, Y_PRED)
                            ACCSCORE_TEMP.append(ACC)

                        ACCSCORE[i, j, k] = np.mean(ACCSCORE_TEMP)
            total_ACCSCORE += ACCSCORE        
        return total_ACCSCORE
    
    best_ACC=0
    
    for metric_name, metric_function in metric_functions.items():
        ACC_score = run_validation(INITIAL_NEURAL_ACTIVITY_coarse, DISCRIMINATION_THRESHOLD_coarse, EPSILON_coarse,metric_name)
        print('uno è fatto')
        val_max = np.max(ACC_score)
        if(val_max==1):
            print(metric_name, "ce l'ha pure")
        if(val_max>best_ACC):
            best_ACC=val_max
            best_metric=metric_name
            max_indices = np.unravel_index(np.argmax(ACC_score), ACC_score.shape)
        else:
            print(metric_name, val_max)
    
    best_DT= DISCRIMINATION_THRESHOLD_coarse[max_indices[0]]
    best_INA= INITIAL_NEURAL_ACTIVITY_coarse[max_indices[1]]
    best_EPSILON= EPSILON_coarse[max_indices[2]]
    
    print(best_INA)
    print(best_DT)
    print(best_EPSILON)
    print(best_metric)
    
    return best_INA, best_DT, best_EPSILON, best_metric










            
    