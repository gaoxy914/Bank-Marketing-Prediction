# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:45:55 2018

@author: gaoxy
"""
import math
import numpy as np
import pickle

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print('SVM')
def train(data, lable):         
    #cross validation
    c_range = np.logspace(-2, 2, 20, base=10)
    g_range = np.logspace(-2, 2, 20)
    mean_acc = np.zeros((20, 20))
    stdv_acc = np.zeros((20, 20))
    for j in range(0, 20):
        for k in range(0, 20):
            c = c_range[j];
            g = g_range[k];
            Kfold = StratifiedKFold(n_splits=5, shuffle=True)
            i = 0
            mean = 0
            stdv = 0
            for train_index, valid_index in Kfold.split(data, lable):
                data_train_cv, data_valid_cv = data[train_index], data[valid_index]
                lable_train_cv, lable_valid_cv = lable[train_index], lable[valid_index]
                model = SVC(C = c, gamma = g)
                model.fit(data_train_cv, lable_train_cv)
                lable_valid_pred_cv = model.predict(data_valid_cv)
                acc_valid = accuracy_score(lable_valid_cv, lable_valid_pred_cv)
                mean = mean + acc_valid
                stdv = stdv + math.pow(acc_valid, 2)
                i = i + 1
            mean = mean / i
            stdv = math.sqrt((stdv/i) - math.pow(mean, 2))
            mean_acc[j, k] = mean
            stdv_acc[j, k] = stdv
    maxacc = 0
    for j in range(0, 20):
        for k in range(0, 20):
            if mean_acc[j, k] >= maxacc:
                maxacc = mean_acc[j, k]
                c_optimal = c_range[j]
                g_optimal = g_range[k]
    count = 0
    minstdv = 100
    for j in range(0, 20):
        for k in range(0, 20):
            if mean_acc[j, k] == maxacc:
                count = count + 1
                if stdv_acc[j, k] <= minstdv:
                    minstdv = stdv_acc[j, k]
                    c_optimal = c_range[j]
                    g_optimal = g_range[k]
    model = SVC(C = c_optimal, gamma = g_optimal, class_weight='balanced')
    model.fit(data, lable)
    with open('svm.model', 'wb') as f:
        pickle.dump(model, f)
    return model, maxacc, minstdv
            
def load_model(file='svm.model'):
    with open(file, 'rb') as f:
        return pickle.load(f)

def test(model, data, label):
    label_pred = model.predict(data)
    print(label_pred)
    acc = accuracy_score(label, label_pred)
    print('Test Dataset accuracy: ', acc)
    roc = roc_auc_score(label, label_pred)
    print('Test Dataset AUC: ', roc)
    f1 = f1_score(label, label_pred)
    print('Test Dataset f1_score: ', f1)
    print(classification_report(label, label_pred))
    print()
    print('Cofusion Matrix: ')
    print(confusion_matrix(label, label_pred))
    print()    





                