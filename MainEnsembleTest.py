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


import os; clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear'); clearConsole()
import numpy as np
import pandas as pd, pickle
from ManyModels import ensemble_learning
from GetShotsFromClass import getShotFromClass

# # ----------------------------------------READ INPUT DATA (x): image histograms (precomputed)-----------------------------------
rootDir = '/Volumes/E/Post_CMPE/MachineLearning/ShotDetection/'
data_258 = pd.read_csv(rootDir + 'Histogram/input_1_6/Histogram_9291.csv')
histoDiff = rootDir + 'Histogram/input_1_6/HistogramDiff_9291.csv'
smoothing_pct, min_shot_length = 10, 75

data = data_258.drop(['h256', 'h257'], axis="columns")
modelDir = rootDir + 'Model/'
outdir = rootDir + 'Output/'

# # -----------------------------Load decision tree model----------------------------------------------
filename = modelDir + 'decision_tree.sav'
dt = pickle.load(open(filename, 'rb'))
pred_dt  = dt.predict(data)

# # -----------------------------Load random forest model----------------------------------------------
filename = modelDir+'random_forest.sav'
rf = pickle.load(open(filename, 'rb'))
pred_rf  = rf.predict(data)

# # -----------------------------Load logistic regression model----------------------------------------------
filename = modelDir+'logistic_regression.sav'
lg = pickle.load(open(filename, 'rb'))
pred_lg  = lg.predict(data)

# # -----------------------------Load svm model----------------------------------------------
filename = modelDir+'svm.sav'
svm = pickle.load(open(filename, 'rb'))
pred_svm = svm.predict(data)

# # -----------------------------Apply ensemble learning----------------------------------------------
PREDICTION = ensemble_learning(pred_svm, pred_dt, pred_rf)

# # -----------------------------Save shot boundary frames from ensemble learning----------------------------------------------
bd_frames = []
for i in range(len(PREDICTION)):
    if PREDICTION[i]:
        bd_frames.append(i)


#print('\n\n-------------------------------GET SHOTS-----------------------------------------')
getShotFromClass(bd_frames, len(PREDICTION), histoDiff, outdir, smoothing_pct, min_shot_length)


