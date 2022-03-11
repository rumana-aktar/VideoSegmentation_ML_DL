# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 03/10/2022
#   
#   Problem: Compute predcition for shot boundaries from decision_tree, random_forest and logistic_regression/svm
#            Call ensemble learning for final result
# 
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

# # -------------------------------------------given histogram data, shot_boundary (shot_bd) and samplying type, prepares the daya
# # -------------------------------------------into two classes, df0, df1--------------------------------
def prepareData(data, shot_bd, sample_type):
    #-------------------------------add output/shot_bd (y) column to data-----------------------------------------
    target = np.zeros((len(data), 1))
    for x in shot_bd:
        target[x]=1
    data['shot_bd'] = target

    #-------------------------------target_bd_index: True where there is a shot_boundary frame-----------------------------------------    
    #-------------------------------target_shot_index: True where there is a shot frame-----------------------------------------    
    target_bd_index = [False for i in range(len(data))]
    target_shot_index = [True for i in range(len(data))]
    for bd in shot_bd:
        target_bd_index[bd] = True
        target_shot_index[bd] = False

    #-------------------------------seperate boundary class (df1) and shot frame class (df0)-----------------------------------------    
    df1 = data[target_bd_index]
    df0 = data[target_shot_index]

    
    print("Len of class_0: ", end = "")
    print(df0.shape)
    print("Len of class_1: ", end = "")
    print(df1.shape)
    print("Unique values in df0: "+str(df0['shot_bd'].unique()))
    print("Unique values in df1: "+str(df1['shot_bd'].unique()))

    
    #------------------------------ Handle Imbalanced Data--------    
    df = handleImbalanceData(sample_type, df0, df1, data)
    return df 



# # -------------------------------------------samplying type, apply different technique to bring balance between two classes: df0, df1
def handleImbalanceData(sample_type, df0, df1, data):

    if sample_type == 0: 
        print('\n\n-------------------------------NO SAMPLING-----------------------------------------')
        df = pd.concat([df0,df1], axis=0)   
    
    elif sample_type == 1:
        print('\n\n-------------------------------UNDER SAMPLING DF0-----------------------------------------')

        df0_new = df0.sample(len(df1))
        df1_new = df1
        df = pd.concat([df0_new, df1_new], axis=0)  

        
    elif sample_type == 2:
        print('\n\n-------------------------------OVER SAMPLING DF1-----------------------------------------')
        df1_new = df1.sample(len(df0), replace=True)    
        df0_new = df0
        df = pd.concat([df0_new, df1_new], axis=0)  
        
    elif sample_type == 3:        
        print('\n\n-------------------------------UNDER SAMPLING DF0 and OVER SAMPLING DF1-----------------------------------------')
        desired_no = len(df0)//2
        df0_new = df0.sample(desired_no)
        df1_new = df1.sample(desired_no, replace=True)    
        df = pd.concat([df0_new, df1_new], axis=0)  
    
    elif sample_type == 4:        
        print('\n\n-------------------------------SMOTE-----------------------------------------')
        x = data.drop('shot_bd',axis='columns')
        y = data['shot_bd']
        
        smote = SMOTE(sampling_strategy='minority')
        x_sm, y_sm = smote.fit_resample(x, y)
        
        # reconstruct df
        df = x_sm
        df['shot_bd'] = y_sm
        
        # reconstract df0 and df1
        df0_new = df[(df['shot_bd'] == 0)]
        df1_new = df[(df['shot_bd'] == 1)]


    
    


    print("Len of class_0: ", end = "")
    print(df0_new.shape)
    print("Len of class_1: ", end = "")
    print(df1_new.shape)
    print("Unique values in df0: "+str(df0_new['shot_bd'].unique()))
    print("Unique values in df1: "+str(df1_new['shot_bd'].unique()))

    return df 

