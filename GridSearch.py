# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 03/10/2022
#   
#   Function: find_best_model_using_gridsearchcv(x, y)  : For given data (x,y), find the best model and parameter combination out of
#                                                         svm, random forest, logistic regression, naive bayes, decision tree
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------

import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        # sv = svm.SVC(kernel='linear')
        'svm': {
            'model': svm.SVC(),
            'params' : {
                'C': [1,10,20]
            }  
        },
        'random_forest': {
            #rf = RandomForestClassifier(n_estimators=20)
            'model': RandomForestClassifier(),
            'params' : {
                'n_estimators': [1,5,10, 20]
            }
        },
        'logistic_regression' : {
            'model': LogisticRegression(solver='liblinear',multi_class='auto'),
            'params': {
                'C': [1,5,10]
            }
        },
        'naive_bayes_gaussian': {
            'model': GaussianNB(),
            'params': {}
        },
        'naive_bayes_multinomial': {
            'model': MultinomialNB(),
            'params': {}
        },
        'decision_tree': {
            'model': tree.DecisionTreeClassifier(),
            'params': {
                'criterion': ['gini','entropy'],
                
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
