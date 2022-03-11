# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 03/10/2022
#   
#   Problem: predict shot boundaries and generates shot_boundary.txt from a new dataset 
#            VIRAT dataset has 9291 frames and this algorithm segments the video into 11 shots/segments
#            Imbalanced Dataset Challenge
# 
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------


import os; clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear'); clearConsole()
import numpy as np, cv2, glob, pickle

from PrepareData import prepareData
from ManyModels import manyModels, manyModels_crossValidation
from GridSearch import find_best_model_using_gridsearchcv
from PrepareDataDeepLearning import prepareDeepLearningData
from GetShotsFromClass import getShotFromClass
from PrepareDataDeepLearning import prepareDeepLearningData



#print('\n\n-------------------------------INPUT PARAMS-----------------------------------------')
smoothing_pct, min_shot_length = 10, 75
trainImageDir = '/Volumes/F/Data/VIRAT_Tape/09152008flight2tape2_8/';
testImageDir = '/Volumes/F/Data/VIRAT_Tape/09152008flight2tape1_6/';
shot_bd = [1345, 1346, 1347, 1647, 1648, 2447, 3410, 3411, 4259, 4260, 5056, 5057, 5819, 5820, 6244, 6245, 6496, 6497, 7283, 7284]

rootDir = '/Volumes/E/Post_CMPE/MachineLearning/ShotDetection/'
histoDiff = rootDir + 'Histogram/input_1_6/HistogramDiff_9291.csv'
model_name = rootDir + 'Model/deep.sav'
outdir = rootDir + 'Output_2/'



#print('\n\n-------------------------------PREPARE THE DATA FOR DEEP NETWORK, DEVELOP AND SAVE THE MODEL-----------------------------------------')
#prepareDeepLearningData(trainImageDir, shot_bd, model_name)



#print('\n\n-------------------------------READ THE DATA FOR TESTING-----------------------------------------')
x= []
files = sorted(glob.glob(testImageDir + "/Fr*.png"));     no_total_frames=len(files)
I = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE);     m, n = I.shape
IMG_width, IMG_height = n // 12, m // 12 

for i in range(no_total_frames):                                          
    if i % 500 == 0:
        print(i, end = " ")
    
    img_array = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE); 
    new_array = cv2.resize(img_array, (IMG_width, IMG_height)) / 255      # resize to normalize data size
    x.append(new_array)

#print('\n\n-------------------------------reshape-----------------------------------------')
x = np.array(x)


#print('\n\n-------------------------------PREICTION-----------------------------------------')
cnn = pickle.load(open(model_name, 'rb'))
PREDICTION = cnn.predict(x)
PREDICTION = [np.argmax(i) for i in PREDICTION]

boundry = []
for i in range(len(PREDICTION)):
    if PREDICTION[i] == 1:
        print([i, PREDICTION[i]])
        boundry.append(i)


#print('\n\n-------------------------------GET SHOTS-----------------------------------------')
getShotFromClass(boundry, len(PREDICTION), histoDiff, outdir, smoothing_pct, min_shot_length)
