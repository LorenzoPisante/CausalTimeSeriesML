# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:07:21 2023

@author: loren
"""

def k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON, DATA_NAME, COUP_COEFF ):
    """

    Parameters
    ----------
    FOLD_NO : TYPE-Integer
        DESCRIPTION-K fold classification.
    traindata : TYPE-numpy 2D array
        DESCRIPTION - Traindata
    trainlabel : TYPE-numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE-numpy 2D array
        DESCRIPTION - Testdata
    testlabel : TYPE - numpy 2D array
        DESCRIPTION - Testlabel
    INITIAL_NEURAL_ACTIVITY : TYPE - numpy 1D array
        DESCRIPTION - initial value of the chaotic skew tent map.
    DISCRIMINATION_THRESHOLD : numpy 1D array
        DESCRIPTION - thresholds of the chaotic map
    EPSILON : TYPE numpy 1D array
        DESCRIPTION - noise intenity for NL to work (low value of epsilon implies low noise )
    DATA_NAME : TYPE - string
        DESCRIPTION.

    Returns
    -------
    FSCORE, Q, B, EPS, EPSILON

    """
    ACCURACY = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    Q = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    B = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))
    EPS = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY),  len(EPSILON)))


    KF = KFold(n_splits= FOLD_NO, random_state=42, shuffle=True) # Define the split - into 2 folds 
    KF.get_n_splits(traindata) # returns the number of splitting iterations in the cross-validator
    print(KF) 
    
    ROW = -1
    COL = -1
    WIDTH = -1
    for DT in DISCRIMINATION_THRESHOLD:
        ROW = ROW+1
        COL = -1
        WIDTH = -1
        for INA in INITIAL_NEURAL_ACTIVITY:
            COL =COL+1
            WIDTH = -1
            for EPSILON_1 in EPSILON:
                WIDTH = WIDTH + 1
                
                ACC_TEMP =[]
                FSCORE_TEMP=[]
            
                for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                    
                    X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                    Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
                    print("Validation data shape",X_VAL.shape)
                    print("train data shape",X_TRAIN.shape)
                    # Extract features
                    FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                    FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)            
                    
                   
                    mean_each_class, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN,Y_TRAIN, FEATURE_MATRIX_VAL)
                    
                    ACC = accuracy_score(Y_VAL, Y_PRED)*100
                    RECALL = recall_score(Y_VAL, Y_PRED , average="macro")
                    PRECISION = precision_score(Y_VAL, Y_PRED , average="macro")
                    F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                                 
                    
                    ACC_TEMP.append(ACC)
                    FSCORE_TEMP.append(F1SCORE)
                Q[ROW, COL, WIDTH ] = INA # Initial Neural Activity
                B[ROW, COL, WIDTH ] = DT # Discrimination Threshold
                EPS[ROW, COL, WIDTH ] = EPSILON_1 
                ACCURACY[ROW, COL, WIDTH ] = np.mean(ACC_TEMP)
                FSCORE[ROW, COL, WIDTH ] = np.mean(FSCORE_TEMP)
                print("Mean F1-Score for Q = ", Q[ROW, COL, WIDTH ],"B = ", B[ROW, COL, WIDTH ],"EPSILON = ", EPS[ROW, COL, WIDTH ]," is  = ",  np.mean(FSCORE_TEMP)  )
    
    print("Saving Hyperparameter Tuning Results")
    
    
    '''PATH = os.getcwd()
    RESULT_PATH = PATH + '/SR-PLOTS/'  + DATA_NAME + '/'+ str(COUP_COEFF )+'/'+ '/NEUROCHAOS-RESULTS/'
    
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    np.save(RESULT_PATH+"/h_fscore.npy", FSCORE )    
    np.save(RESULT_PATH+"/h_accuracy.npy", ACCURACY ) 
    np.save(RESULT_PATH+"/h_Q.npy", Q ) 
    np.save(RESULT_PATH+"/h_Q.npy", B )
    np.save(RESULT_PATH+"/h_EPS.npy", EPS )     '''          
    
    
    MAX_FSCORE = np.max(FSCORE)
    MAX_ACCURACY = np.max(ACCURACY)
    if DATA_NAME=="single_variable_classification":
        Perf_Metric = ACCURACY
        MAX_metric = np.max(Perf_Metric)
        print("BEST Accuracy", MAX_metric)
    else:
        Perf_Metric = FSCORE
        MAX_metric = np.max(Perf_Metric)
        print("BEST F1SCORE", MAX_metric)
    
   
    Q_MAX = []
    B_MAX = []
    EPSILON_MAX = []
    
    for ROW in range(0, len(DISCRIMINATION_THRESHOLD)):
        for COL in range(0, len(INITIAL_NEURAL_ACTIVITY)):
            for WID in range(0, len(EPSILON)):
                if Perf_Metric[ROW, COL, WID] == MAX_metric:
                    Q_MAX.append(Q[ROW, COL, WID])
                    B_MAX.append(B[ROW, COL, WID])
                    EPSILON_MAX.append(EPS[ROW, COL, WID])
    
    
   
    print("BEST INITIAL NEURAL ACTIVITY = ", Q_MAX)
    print("BEST DISCRIMINATION THRESHOLD = ", B_MAX)
    print("BEST EPSILON = ", EPSILON_MAX)
    return Perf_Metric, Q, B, EPS, EPSILON

def chaosnet(traindata, trainlabel, testdata):
    '''
    

    Parameters
    ----------
    traindata : TYPE - Numpy 2D array
        DESCRIPTION - traindata
    trainlabel : TYPE - Numpy 2D array
        DESCRIPTION - Trainlabel
    testdata : TYPE - Numpy 2D array
        DESCRIPTION - testdata

    Returns
    -------
    mean_each_class : Numpy 2D array
        DESCRIPTION - mean representation vector of each class
    predicted_label : TYPE - numpy 1D array
        DESCRIPTION - predicted label

    '''
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(traindata[(trainlabel == label)[:,0], :], axis=0)
        
    predicted_label = np.argmax(cosine_similarity(testdata, mean_each_class), axis = 1)

    return mean_each_class, predicted_label

def k_cross_validation_multi_dataset(datasets, FOLD_NO, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON):
    """
    Parameters
    ----------
    datasets : List of tuples
        Each tuple in the list should contain (traindata, trainlabel).
    FOLD_NO : Integer
        K fold classification.
    INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON : numpy 1D arrays
        Parameters for chaosnet.

    Returns
    -------
    Best combination of INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON.
    """
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold
    import ChaosFEX.feature_extractor as CFX
    from codes import chaosnet  # Assicurati che chaosnet sia importato correttamente

    # Initialize arrays to store F1 scores for each dataset and parameter combination
    total_FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))

    for dataset in datasets:
        traindata, trainlabel = dataset

        # Initialize arrays for individual dataset performance
        FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))

        KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
        KF.get_n_splits(traindata)

        for i, DT in enumerate(DISCRIMINATION_THRESHOLD):
            for j, INA in enumerate(INITIAL_NEURAL_ACTIVITY):
                for k, EPSILON_1 in enumerate(EPSILON):
                    FSCORE_TEMP = []
                    for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                        X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                        Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]

                        # Trasformazione dei dati con CFX
                        FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                        FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)

                        # Model training e predizione
                        _, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, Y_TRAIN, FEATURE_MATRIX_VAL)

                        # Calcolo del punteggio F1
                        F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                        FSCORE_TEMP.append(F1SCORE)

                    # Calcolo del punteggio F1 medio per questa combinazione di parametri
                    FSCORE[i, j, k] = np.mean(FSCORE_TEMP)

        # Somma dei punteggi F1 per ogni combinazione di parametri
        total_FSCORE += FSCORE

    # Identifica la combinazione con il punteggio F1 totale più alto
    max_f1_index = np.unravel_index(np.argmax(total_FSCORE), total_FSCORE.shape)
    best_DT, best_INA, best_EPSILON = DISCRIMINATION_THRESHOLD[max_f1_index[0]], INITIAL_NEURAL_ACTIVITY[max_f1_index[1]], EPSILON[max_f1_index[2]]

    print(f"Best Initial Neural Activity: {best_INA}, Best Discrimination Threshold: {best_DT}, Best Epsilon: {best_EPSILON}")

    return best_INA, best_DT, best_EPSILON

def k_cross_validation_refined_search(datasets, FOLD_NO):
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold
    import ChaosFEX.feature_extractor as CFX
    from codes import chaosnet

    tolerance=0.3
    INITIAL_NEURAL_ACTIVITY_coarse = np.arange(0.1, 1, 0.1)
    DISCRIMINATION_THRESHOLD_coarse = np.arange(0.1, 1, 0.1)
    EPSILON_coarse = np.arange(0.01, 0.3, 0.01)
    
    
    def run_validation(INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON):
        total_FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
        for dataset in datasets:
            traindata, trainlabel = dataset
            FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
            KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
            KF.get_n_splits(traindata)

            for i, DT in enumerate(DISCRIMINATION_THRESHOLD):
                for j, INA in enumerate(INITIAL_NEURAL_ACTIVITY):
                    for k, EPSILON_1 in enumerate(EPSILON):
                        FSCORE_TEMP = []
                        for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                            X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                            Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]

                            FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                            FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)

                            _, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, Y_TRAIN, FEATURE_MATRIX_VAL)
                            F1SCORE = f1_score(Y_VAL, Y_PRED, average="macro")
                            FSCORE_TEMP.append(F1SCORE)

                        FSCORE[i, j, k] = np.mean(FSCORE_TEMP)
                        
            total_FSCORE += FSCORE
        return total_FSCORE

    # Fase 1: Ricerca a Grana Grossa
    

    coarse_scores = run_validation(INITIAL_NEURAL_ACTIVITY_coarse, DISCRIMINATION_THRESHOLD_coarse, EPSILON_coarse)
    max_score = np.max(coarse_scores)
    promising_indices = np.argwhere(coarse_scores >= max_score - tolerance)

    # Fase 2: Ricerca a Grana Fine
    best_score = 0
    best_params = None
    for index in promising_indices:
        DT_index, INA_index, EPS_index = index
        INA_range = INITIAL_NEURAL_ACTIVITY_coarse[INA_index] + np.array([-0.065,-0.04,-0.02, 0, 0.02,0.04,0.065])
        DT_range = DISCRIMINATION_THRESHOLD_coarse[DT_index] + np.array([-0.065,-0.04,-0.02, 0, 0.02,0.04,0.065])
        EPS_range = EPSILON_coarse[EPS_index] + np.array([-0.02,-0.01, 0, 0.01, 0.02])

        fine_scores = run_validation(INA_range, DT_range, EPS_range)
        max_fine_score = np.max(fine_scores)
        if max_fine_score > best_score:
            best_score = max_fine_score
            best_params = np.unravel_index(np.argmax(fine_scores), fine_scores.shape)

    best_INA, best_DT, best_EPSILON = INA_range[best_params[1]], DT_range[best_params[0]], EPS_range[best_params[2]]
    return best_INA, best_DT, best_EPSILON

def k_cross_validation_refined_search2(datasets, FOLD_NO=4):
    import numpy as np
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.model_selection import KFold
    import ChaosFEX.feature_extractor as CFX
    from codes import chaosnet

    tolerance=0.015
    INITIAL_NEURAL_ACTIVITY_coarse = np.arange(0.2, 0.8, 0.1)
    DISCRIMINATION_THRESHOLD_coarse = np.arange(0.2, 0.8, 0.1)
    EPSILON_coarse = np.arange(0.02, 0.12, 0.02)
    
    
    def run_validation(INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON):
        total_ACCSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
        for q,dataset in enumerate(datasets):
            traindata, trainlabel = dataset
            ACCSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
            KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
            KF.get_n_splits(traindata)

            for i, DT in enumerate(DISCRIMINATION_THRESHOLD):               
                for j, INA in enumerate(INITIAL_NEURAL_ACTIVITY):
                    for k, EPSILON_1 in enumerate(EPSILON):
                        ACCSCORE_TEMP = []
                        #ACCSCORE = np.zeros((DISCRIMINATION_THRESHOLD.shape[0], INITIAL_NEURAL_ACTIVITY.shape[0],EPSILON.shape[0] ))

                        for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                            X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                            Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]

                            FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                            FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)

                            _, Y_PRED = chaosnet(FEATURE_MATRIX_TRAIN, Y_TRAIN, FEATURE_MATRIX_VAL)
                            ACC = accuracy_score(Y_VAL, Y_PRED)
                            ACCSCORE_TEMP.append(ACC)

                        ACCSCORE[i, j, k] = np.mean(ACCSCORE_TEMP)
            total_ACCSCORE += ACCSCORE        
        return total_ACCSCORE



    # Fase 1: Ricerca a Grana Grossa
    
    ACC_score = run_validation(INITIAL_NEURAL_ACTIVITY_coarse, DISCRIMINATION_THRESHOLD_coarse, EPSILON_coarse)
    #Trova il valore massimo e i suoi indici
    val_max = np.max(ACC_score)
    max_indices = np.unravel_index(np.argmax(ACC_score), ACC_score.shape)

    # Trova gli indici dove i valori sono compresi tra val_max e val_max - tollerance
    best_indices = np.array(np.where((ACC_score >= val_max - tolerance) & (ACC_score <= val_max)))
    
    best_values = []
    for i in range(best_indices.shape[1]):
        DT_index, INA_index, eps_index = best_indices[:, i]
        DT_value = DISCRIMINATION_THRESHOLD_coarse[DT_index]
        INA_value = INITIAL_NEURAL_ACTIVITY_coarse[INA_index]
        eps_value = EPSILON_coarse[eps_index]
        best_values.append((DT_value, INA_value, eps_value)) # Itera sulle colonne

      
    # Ogni set di valori come colonna
    best_values_array = np.array(best_values).T

    # Fase 2: Ricerca a Grana Fine
    best_score = 0
    best_params = None
    num_ind=np.shape(best_values_array)[1]
    for i in range(num_ind):
        DT_index, INA_index, EPS_index = best_values_array[:,i] 
        INA_range = INA_index + np.array([-0.065,-0.04,-0.02, 0, 0.02,0.04,0.065])
        DT_range = DT_index + np.array([-0.065,-0.04,-0.02, 0, 0.02,0.04,0.065])
        EPS_range = EPS_index + np.array([-0.01, 0, 0.01])

        fine_scores = run_validation(INA_range, DT_range, EPS_range)
        
        max_fine_score = np.max(fine_scores)
        print (max_fine_score)
        if max_fine_score > best_score:
            best_score = max_fine_score
            best_params = np.unravel_index(np.argmax(fine_scores), fine_scores.shape)
            best_params_array = np.array(best_params)
            best_DT= DT_range[best_params_array[0]]
            best_INA= INA_range[best_params_array[1]]
            best_EPSILON= EPS_range[best_params_array[2]]
           

    print(best_INA)
    print(best_DT)
  
    print(best_EPSILON)
    
    # Colonna da aggiungere
    B = np.array([[best_DT],
                 [best_INA],
                 [best_EPSILON]])

    # Aggiungi la colonna
    best_values_array = np.hstack((best_values_array, B))
    #best_INA, best_DT, best_EPSILON = INA_range[best_params[1]], DT_range[best_params[0]], EPS_range[best_params[2]]
    return best_INA, best_DT, best_EPSILON, best_values_array

"""def k_cross_validation_realdata(datasets, Weight, FOLD_NO):
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold
    import ChaosFEX.feature_extractor as CFX
    from codes import chaosnet

    tolerance=0.015
    INITIAL_NEURAL_ACTIVITY_coarse = np.arange(0.2, 0.8, 0.1)
    DISCRIMINATION_THRESHOLD_coarse = np.arange(0.2, 0.8, 0.1)
    EPSILON_coarse = np.arange(0.03, 0.12, 0.02)
    
    
    def run_validation(INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON):
        total_FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
        for q,dataset in enumerate(datasets):
            traindata, trainlabel = dataset
            FSCORE = np.zeros((len(DISCRIMINATION_THRESHOLD), len(INITIAL_NEURAL_ACTIVITY), len(EPSILON)))
            KF = KFold(n_splits=FOLD_NO, random_state=42, shuffle=True)
            KF.get_n_splits(traindata)
            

            for i, DT in enumerate(DISCRIMINATION_THRESHOLD):
                
                for j, INA in enumerate(INITIAL_NEURAL_ACTIVITY):
                    for k, EPSILON_1 in enumerate(EPSILON):
                        
                        
                        FSCORE_TEMP = []
                        for TRAIN_INDEX, VAL_INDEX in KF.split(traindata):
                            X_TRAIN, X_VAL = traindata[TRAIN_INDEX], traindata[VAL_INDEX]
                            Y_TRAIN, Y_VAL = trainlabel[TRAIN_INDEX], trainlabel[VAL_INDEX]
                            Weight_TRAIN, Weight_VAL = Weight[TRAIN_INDEX], Weight[VAL_INDEX]
                            FEATURE_MATRIX_TRAIN = CFX.transform(X_TRAIN, INA, 10000, EPSILON_1, DT)
                            FEATURE_MATRIX_VAL = CFX.transform(X_VAL, INA, 10000, EPSILON_1, DT)

 
                            from sklearn.metrics.pairwise import cosine_similarity
                            NUM_FEATURES = FEATURE_MATRIX_TRAIN.shape[1]
                            NUM_CLASSES = len(np.unique(trainlabel))

                            mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
                            for label in range(0, NUM_CLASSES):
            
                                 mean_each_class[label, :] = np.mean(FEATURE_MATRIX_TRAIN[(trainlabel[TRAIN_INDEX] == label)[:,0], :], axis=0)
        
                            #predicted_label = np.argmax(cosine_similarity(feat_mat_testdata, mean_each_class), axis = 1)
                            predicted_label = np.zeros(np.shape(FEATURE_MATRIX_VAL)[0])
                            lun = int(np.shape(FEATURE_MATRIX_VAL)[0]/2)
                            scores = cosine_similarity(FEATURE_MATRIX_VAL, mean_each_class)
                            for t in range (lun):
                                
                                if (abs(scores[t,0]-scores[t,1])>abs(scores[t+lun,0] - scores[t+lun,1])):
                                    if(scores[t,0]-scores[t,1]>0):
                                        predicted_label[t]=0
                                        predicted_label[t+lun]=1
                                    else:
                                        predicted_label[t]=1
                                        predicted_label[t+lun]=0
                                else:
                                    if(scores[t+lun,0]-scores[t+lun,1]<0):
                                        predicted_label[t]=0
                                        predicted_label[t+lun]=1
                                    else:
                                        predicted_label[t]=1
                                        predicted_label[t+lun]=0
                            sum_weight = np.sum(Weight_VAL)
                            num=0
                            for t in range(int(len(predicted_label)/2)):
                                
                                if (Y_VAL[t] == predicted_label[t]):
                                    num=num+Weight_VAL[t]
                                          
                            Weight_ACC=num/sum_weight
                                        
                            
                            F1SCORE = f1_score(Y_VAL, predicted_label, average="macro")
                            FSCORE_TEMP.append(Weight_ACC)

                        FSCORE[i, j, k] = np.mean(FSCORE_TEMP)
            total_FSCORE += FSCORE        
        return total_FSCORE



    # Fase 1: Ricerca a Grana Grossa
    
    F_score = run_validation(INITIAL_NEURAL_ACTIVITY_coarse, DISCRIMINATION_THRESHOLD_coarse, EPSILON_coarse)
    #Trova il valore massimo e i suoi indici
    val_max = np.max(F_score)
    max_indices = np.unravel_index(np.argmax(F_score), F_score.shape)

    # Trova gli indici dove i valori sono compresi tra val_max e val_max - tollerance
    best_indices = np.array(np.where((F_score >= val_max - tolerance) & (F_score <= val_max)))
    
    best_values = []
    for i in range(best_indices.shape[1]):
        DT_index, INA_index, eps_index = best_indices[:, i]
        DT_value = DISCRIMINATION_THRESHOLD_coarse[DT_index]
        INA_value = INITIAL_NEURAL_ACTIVITY_coarse[INA_index]
        eps_value = EPSILON_coarse[eps_index]
        best_values.append((DT_value, INA_value, eps_value)) # Itera sulle colonne

      
    # Ogni set di valori come colonna
    best_values_array = np.array(best_values).T

    # Fase 2: Ricerca a Grana Fine
    best_score = 0
    best_params = None
    num_ind=np.shape(best_values_array)[1]
    for i in range(num_ind):
        DT_index, INA_index, EPS_index = best_values_array[:,i] 
        INA_range = INA_index + np.array([-0.065,-0.04,-0.02, 0, 0.02,0.04,0.065])
        DT_range = DT_index + np.array([-0.065,-0.04,-0.02, 0, 0.02,0.04,0.065])
        EPS_range = EPS_index + np.array([-0.01, 0, 0.01])

        fine_scores = run_validation(INA_range, DT_range, EPS_range)
        max_fine_score = np.max(fine_scores)
        if max_fine_score > best_score:
            best_score = max_fine_score
            best_params = np.unravel_index(np.argmax(fine_scores), fine_scores.shape)
            best_params_array = np.array(best_params)
            best_DT= DT_range[best_params_array[0]]
            best_INA= INA_range[best_params_array[1]]
            best_EPSILON= EPS_range[best_params_array[2]]
           

    print(best_INA)
    print(best_DT)
  
    print(best_EPSILON)
    #best_INA, best_DT, best_EPSILON = INA_range[best_params[1]], DT_range[best_params[0]], EPS_range[best_params[2]]
    return best_INA, best_DT, best_EPSILON, best_values_array"""
