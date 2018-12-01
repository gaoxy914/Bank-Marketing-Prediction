# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:23:02 2018

@author: gaoxy
"""

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn import preprocessing

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler   
        
def load_training_data(path):
    data = pd.read_csv(path,delimiter=';')
    #print(data.shape)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if (data.iloc[j, i] == 'unknown'):
                data.iloc[j, i] = '0'
    le = preprocessing.LabelEncoder()
    #job marital education default housing loan contact month day_of_week(2-10) poutcome(15) y(21)
    for i in range(1, 10):
        encode = le.fit_transform(data.iloc[:, i])
        data.iloc[:, i] = encode
    encode = le.fit_transform(data.iloc[:, 14])
    data.iloc[:, 14] = encode
    encode = le.fit_transform(data.iloc[:, 20])
    data.iloc[:, 20] = encode
    D = data.values
    D = np.array(D)
    #print(D)
    negative = 0
    positive = 0
    D1 = np.empty((0, 21))
    D2 = np.empty((0, 21))
    for i in range(D.shape[0]):
        if (D[i][20] == 0):
            negative = negative + 1
            D1 = np.append(D1, [D[i, :]], axis=0)
        else:
            positive = positive + 1
            D2 = np.append(D2, [D[i, :]], axis=0)
    #filling missing data job marital education default housing loan(2-7)
    imp = Imputer(missing_values=0, strategy='most_frequent', axis=0)
    filler = imp.fit_transform(D1[:, 1:7])
    D1[:, 1:7] = filler
    filler = imp.fit_transform(D2[:, 1:7])
    D2[:, 1:7] = filler
    D = np.append(D1, D2, axis=0)
    ohe = preprocessing.OneHotEncoder(n_values='auto', categorical_features=[1, 2, 3, 4, 5, 6, 7, 8, 9, 14], sparse=False, handle_unknown='error')
    ohe.fit(D)
    D = ohe.transform(D)
    #print(D.shape)
    X = D[:, 0:57]
    Y = D[:, 57]
    #Y = Y.reshape(-1, 1)
    #normalize
    ss = StandardScaler()
    ss.fit(X[:, 47:51])
    scaler = ss.transform(X[:, 47:51])
    X[:, 47:51] = scaler
    ss.fit(X[:, 52:])
    scaler = ss.transform(X[:, 52:])
    X[:, 52:] = scaler
    X, Y = SMOTE().fit_sample(X, Y)
    positive = 0
    negative = 0
    for i in range(Y.shape[0]):
        if (Y[i] == 0):
            negative = negative + 1
        else:
            positive = positive + 1
    return X, Y, positive, negative

def load_testing_data(path):
    data = pd.read_csv(path,delimiter=';')
    #print(data.shape)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if (data.iloc[j, i] == 'unknown'):
                data.iloc[j, i] = '0'
    le = preprocessing.LabelEncoder()
    #job marital education default housing loan contact month day_of_week(2-10) poutcome(15) y(21)
    for i in range(1, 10):
        encode = le.fit_transform(data.iloc[:, i])
        data.iloc[:, i] = encode
    encode = le.fit_transform(data.iloc[:, 14])
    data.iloc[:, 14] = encode
    encode = le.fit_transform(data.iloc[:, 20])
    data.iloc[:, 20] = encode
    D = data.values
    D = np.array(D)
    #filling missing data job marital education default housing loan(2-7)
    imp = Imputer(missing_values=0, strategy='most_frequent', axis=0)
    filler = imp.fit_transform(D[:, 1:7])
    D[:, 1:7] = filler
    ohe = preprocessing.OneHotEncoder(n_values='auto', categorical_features=[1, 2, 3, 4, 5, 6, 7, 8, 9, 14], sparse=False, handle_unknown='error')
    ohe.fit(D)
    D = ohe.transform(D)
    #print(D.shape)
    X = D[:, 0:57]
    Y = D[:, 57]
    #Y = Y.reshape(-1, 1)
    #normalize
    ss = StandardScaler()
    ss.fit(X[:, 47:51])
    scaler = ss.transform(X[:, 47:51])
    X[:, 47:51] = scaler
    ss.fit(X[:, 52:])
    scaler = ss.transform(X[:, 52:])
    X[:, 52:] = scaler
    return X, Y

if __name__=='__main__':
    X, Y, p, n = load_training_data('bank-additional/bank-additional.csv')
    print(p, n)
    '''
    test_data, test_lable = load_testing_data('bank-additional/bank-additional.csv')
    print(test_data[1], test_lable[1])
    '''
    #pass