# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 03/10/2022
#   
#   Problem: Compute predcition for shot boundaries from decision_tree, random_forest and logistic_regression/svm
#            Call ensemble learning for final result
#            This works basically for Training Dataset
# 
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------

import os; clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear'); clearConsole()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from PrepareData import prepareData
from ManyModels import manyModels, ensemble_learning, manyModels_crossValidation
from GridSearch import find_best_model_using_gridsearchcv


# -------------------------------prepare data----------------------------------------------------------------------------------
histogram = '/Volumes/E/Post_CMPE/MachineLearning/ShotDetection/Histogram/input_2_8/Histogram_9291.csv';
shot_bd = [1345, 1346, 1347, 1647, 1648, 2447, 3410, 3411, 4259, 4260, 5056, 5057, 5819, 5820, 6244, 6245, 6496, 6497, 7283, 7284]
modelDir = '/Volumes/E/Post_CMPE/MachineLearning/ShotDetection/Model/'


#-------------------------------Initially histogram has 258 columns (257: histogramDiff, 258: cumulaitveHistogramDifference); discard the last two columns-----------------------------------------
data_258 = pd.read_csv(histogram);          
data = data_258.drop(['h256', 'h257'], axis="columns")


#---------------------------------------------------------------------------------------------------------------------------------
#-------- 1: undersmapling mejority class, 2: oversampling minority class, 3: undersample + oversampling, 4: SMOTE: Synthetic Minority Oversampling Technique}
sampling_type = 3
df = prepareData(data, shot_bd, sampling_type)


#---------------------------------------- shuffle the data-------------------------------------------------------------------------
from sklearn.utils import shuffle
df = shuffle(df)
x = df.drop('shot_bd', axis='columns')
y = df.shot_bd


# ----------------------------------------train_test_split-------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 10)


# #-------------------------Try different models with cross validation and observe the result--------------------------------------
# manyModels_crossValidation(x, y)

# #-------------------------------After cross validation, try different to find best model with grid search------------------------
# best_model_info = find_best_model_using_gridsearchcv(x,y)
# print(best_model_info)


# #-------------------------------Finally with best found few models, TRY ensemble learning----------------------------------------
[y_predicted_dt, y_predicted_rf, y_predicted_lr] = manyModels(x_train, x_test, y_train, y_test, modelDir)
PREDICTION = ensemble_learning(y_predicted_dt, y_predicted_rf, y_predicted_lr)


# # -----------------------------Save shot boundary frames from ensemble learning----------------------------------------------
bd_frames = []
for i in range(len(PREDICTION)):
    if PREDICTION[i]:
        bd_frames.append(i)
np.savetxt('./Output/input_1_6_ensemble_bd_frames.txt',             bd_frames, fmt='%d')

