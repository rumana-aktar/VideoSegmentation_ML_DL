# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 03/10/2022
#   
#   Function: manyModels_crossValidation(x, y):             Apply cross validation for data[x,y] to make sure the consistency of the models
#   Function: manyModels(x_train, x_test, y_train, y_test): Apply knn, kmeans, decision_tree, random_forest, svm, naive_bayes, logistic regression 
#                                                           given train and test dataset
#   Function: ensemble_learning(pred1, pred2, pred3):       Apply 3 models and combine their reults for the final predcition; 
#                                                           if atleast two of them says YES, consider it as a YES
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------


from sklearn.metrics import confusion_matrix , classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

import numpy as np, pickle


# # ---------------------------------------------------------------------------
## apply cross validation to make sure the models results are consistent for all sets of training data
def manyModels_crossValidation(x, y):
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score

    cv = ShuffleSplit(n_splits=5, test_size = 0.2, random_state = 0)

    svm_score = cross_val_score(svm.SVC(kernel='linear'), x, y, cv = cv)
    svm_precision = cross_val_score(svm.SVC(kernel='linear'), x, y, cv = cv, scoring='precision')
    svm_recall = cross_val_score(svm.SVC(kernel='linear'), x, y, cv = cv, scoring='recall')
    
    dt_score = cross_val_score(tree.DecisionTreeClassifier(), x, y, cv = cv)
    dt_precision = cross_val_score(tree.DecisionTreeClassifier(), x, y, cv = cv, scoring='precision')
    dt_recall = cross_val_score(tree.DecisionTreeClassifier(), x, y, cv = cv, scoring='recall')
  
    rf_score = cross_val_score(RandomForestClassifier(n_estimators=20), x, y, cv = cv)
    rf_precision = cross_val_score(RandomForestClassifier(n_estimators=20), x, y, cv = cv)
    rf_recall = cross_val_score(RandomForestClassifier(n_estimators=20), x, y, cv = cv, scoring='recall')

    lr_score = cross_val_score(LogisticRegression(), x, y, cv = cv)
    lr_precision = cross_val_score(LogisticRegression(), x, y, cv = cv, scoring='precision')
    lr_recall = cross_val_score(LogisticRegression(), x, y, cv = cv, scoring='recall')

    gnv_score = cross_val_score(GaussianNB(), x, y, cv = cv)
    gnv_precision = cross_val_score(GaussianNB(), x, y, cv = cv, scoring='precision')
    gnv_recall = cross_val_score(GaussianNB(), x, y, cv = cv, scoring='recall')

    np.set_printoptions(precision=3)
    print("svm                   score with 5k Cross Validation (accuracy, precision, recall): ", end = " ")
    printUptoDecimal([np.around(np.mean(svm_score),3), np.mean(svm_precision), np.mean(svm_recall)])

    print("Decision Tree         score with 5k Cross Validation (accuracy, precision, recall): ", end = " ")
    printUptoDecimal([np.mean(dt_score),   np.mean(dt_precision), np.mean(dt_recall)])
    #print([format(np.mean(dt_score), ".3f")])

    print("Random Forest         score with 5k Cross Validation (accuracy, precision, recall): ", end = " ")
    printUptoDecimal([np.mean(rf_score), np.mean(rf_precision), np.mean(rf_recall)])

    print("Logistic Regression   score with 5k Cross Validation (accuracy, precision, recall): ", end = " ")
    printUptoDecimal([np.mean(lr_score), np.mean(lr_precision), np.mean(lr_recall)])

    print("Naive Bayes           score with 5k Cross Validation (accuracy, precision, recall): ", end = " ")
    printUptoDecimal([np.mean(gnv_score), np.mean(gnv_precision), np.mean(gnv_recall)])



def printUptoDecimal(arr):
    for x in arr:
        print(np.around(x, 3), end="  ")
    print("")    



# # ---------------------------------------------------------------------------
# just experiments: given train and test dataset, apply different models and return results

def manyModels(x_train, x_test, y_train, y_test, modelDir):
    print('\n\n-------------------------------KNN-----------------------------------------')
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    knn_score= knn.score(x_test, y_test)
    y_predicted = knn.predict(x_test)
    print(classification_report(y_test, y_predicted))


    print('\n\n-------------------------------kmeans-----------------------------------------')
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(x_train, y_train)
    kmeans_score= kmeans.score(x_test, y_test)
    y_predicted = kmeans.predict(x_test)
    print(classification_report(y_test, y_predicted))


    print('\n\n-------------------------------SVM-----------------------------------------')
    #from sklearn import svm
    sv = svm.SVC(kernel='linear')
    sv.fit(x_train, y_train)
    sv_score= sv.score(x_test, y_test)
    y_predicted = sv.predict(x_test)
    print(classification_report(y_test, y_predicted))

    filename = modelDir + 'svm.sav'
    pickle.dump(sv, open(filename, 'wb'))

    print('\n\n-------------------------------Decision Tree-----------------------------------------')
    #from sklearn import tree
    dt = tree.DecisionTreeClassifier(criterion= 'gini')
    dt.fit(x_train, y_train)
    score = dt.score(x_test, y_test)
    y_predicted = dt.predict(x_test)
    print(classification_report(y_test, y_predicted))
    y_predicted_dt = y_predicted
    
    filename = modelDir + 'decision_tree.sav'
    pickle.dump(dt, open(filename, 'wb'))




    print('\n\n-------------------------------Random Forest-----------------------------------------')
    #from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(x_train, y_train)
    rf_score = rf.score(x_test, y_test)
    y_predicted = rf.predict(x_test)
    print(classification_report(y_test, y_predicted))
    y_predicted_rf = y_predicted

    filename = modelDir + 'random_forest.sav'
    pickle.dump(rf, open(filename, 'wb'))



    print('\n\n-------------------------------Logistic Regression-----------------------------------------')
    #from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    lr_score= lr.score(x_test, y_test)
    y_predicted = lr.predict(x_test)
    print(classification_report(y_test, y_predicted))
    y_predicted_lr = y_predicted

    filename = modelDir + 'logistic_regression.sav'
    pickle.dump(lr, open(filename, 'wb'))



    print('\n\n-------------------------------Naive Bayes-----------------------------------------')
    #from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    lr.score(x_test, y_test)
    y_predicted = gnb.predict(x_test)
    print(classification_report(y_test, y_predicted))

    return [y_predicted_dt, y_predicted_rf, y_predicted_lr]

# # ---------------------------------------------------------------------------
# apply 3 models and combine their reults for the final predcition; if atleast two of them says YES, consider it as a YES

def ensemble_learning(pred1, pred2, pred3):
    pred = []
    for i in range(len(pred1)):
        val = pred1[i] + pred2[i] + pred3[i]
        if val >= 2:
            pred.append(1)
        else:
            pred.append(0)

    return pred            
