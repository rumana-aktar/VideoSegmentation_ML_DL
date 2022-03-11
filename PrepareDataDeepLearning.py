# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 03/10/2022
#   
#   Problem: PREPARE GROUNDTRUTH DATA AND DEVELOP AND SAVE A MODEL
# 
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------

import numpy as np, cv2, os, glob, random
from tqdm import tqdm
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import train_test_split
import pickle


##--------------------------------------------------------------------------------------------------
def prepareDeepLearningData(imgDir, shot_bd, model_name):    
    #print('\n\n-------------------------------GROUNDTRUTH IMAGE AND BOUNDARY-----------------------------------------')
    # imgDir = '/Volumes/F/Data/VIRAT_Tape/09152008flight2tape2_8/';
    # shot_bd = [1345, 1346, 1347, 1647, 1648, 2447, 3410, 3411, 4259, 4260, 5056, 5057, 5819, 5820, 6244, 6245, 6496, 6497, 7283, 7284]
    # model_dir = '/Volumes/E/Post_CMPE/MachineLearning/ShotDetection/Model/'

    target = np.zeros((9291, 1))
    for x in shot_bd:
        target[x]=1

    #print('\n\n-------------------------------READ THRE DATA FOR TRAINING-----------------------------------------')
    training_data, x, y = [], [], [] 
    IMG_width, IMG_height = 720 // 12, 480 // 12 
    files = sorted(glob.glob(imgDir + "/Fr*.png")); no_total_frames=len(files)
    
    for i in range(no_total_frames):                                             # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE); 
            new_array = cv2.resize(img_array, (IMG_width, IMG_height)) / 255      # resize to normalize data size
            training_data.append([new_array, target[i]])                          # add this to our training_data
            x.append(new_array)
            y.append(target[i])
        except Exception as e:                                                     # in the interest in keeping the output clean...
            print(e)
            pass
        i += 1

    #print('\n\n-------------------------------RESHAPE-----------------------------------------')
    training_data = np.array(training_data)
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(-1, )

    #print('\n\n-------------------------------OVERSAMPLING, UNDERSAMPLING, SHUFFLE HERE-----------------------------------------')
    x, y = [], []
    data = overUnderSampling(training_data, shot_bd)
    for features,label in data:
        x.append(features)
        y.append(label)
    x = np.array(x)   
    y = np.array(y)    
    y = y.reshape(-1, )

    #print('\n\n-------------------------------SPLIT-----------------------------------------')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 10)
    
    #print('\n\n-------------------------------MODEL-----------------------------------------')
    cnn = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(40, 60, 1)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    #print('\n\n-------------------------------COMPILE-----------------------------------------')
    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    cnn.fit(x_train, y_train, epochs = 30)
    # class_weights = {0: 0.501, 1: 232}                                 
    # cnn.fit(x_train, y_train, epochs=10, class_weight=class_weights, validation_data = (x_test, y_test))
    
    #print('\n\n-------------------------------SAVE THE MODEL-----------------------------------------')
    #model_name = model_dir + 'shot_detection_over_under.sav'
    pickle.dump(cnn, open(model_name, 'wb'))


    #print('\n\n-------------------------------PREDICTION-----------------------------------------')
    score = cnn.evaluate(x_test,y_test)
    y_pred_prob = cnn.predict(x_test)
    y_predicted = [np.argmax(i) for i in y_pred_prob]

    #print('\n\n-------------------------------CLASSIFICATION REPORT-----------------------------------------')
    print(classification_report(y_test, y_predicted))




##--------------------------------------------------------------------------------------------------
# OVERSAMPLE BOUNDARY DATA AND UNDERSAMPLE SHOT DATA AS THERE IS NOT SAMPLE FUNCTION FOR NP.ARRAY
def overUnderSampling(traningData, bdList):
    
    #print('\n\n-------------------------------PREPARE SHOT LIST-----------------------------------------')
    shotList = set([i for i in range(len(traningData))])
    for idx in bdList:
        shotList.remove(idx)
    shotList = list(shotList)   

    #print('\n\n-------------------------------EXTRACT SHOT AND BOUNDARY ROWS-----------------------------------------')
    shot_rows = traningData[shotList, :]
    bd_rows = traningData[bdList, :]
    
    #print('\n\n-------------------------------HOW MANY DATA WE WANT-----------------------------------------')
    target = len(shot_rows) // 2
    

    #print('\n\n-------------------------------OVERSAMPLE BD FRAMES-----------------------------------------')
    times = target//len(bd_rows)
    bd_rows_new = []
    for i in range(times):
        bd_rows_new.extend(bd_rows)

    #print('\n\n-------------------------------UNDERSAMPLE SHOT FRAMES-----------------------------------------')
    idx_list = random.sample(range(0, len(shot_rows)-1), target)
    shot_rows_new = shot_rows[idx_list, :]
    
    #print('\n\n-------------------------------RECREATE NEW DATASET-----------------------------------------')
    new_trainData = []    
    new_trainData.extend(shot_rows_new)
    new_trainData.extend(bd_rows_new)

    print('Before sampling: Len of shot frames, boundary frames ', end = " "); print([len(shot_rows), len(bd_rows)])
    print('After  sampling: Len of shot frames, boundary frames ', end = " "); print([len(shot_rows_new), len(bd_rows_new)])

    from sklearn.utils import shuffle
    new_trainData = shuffle(new_trainData)
    return new_trainData


