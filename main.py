# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:43:49 2018

@author: gaoxy
"""
import math

from data_loading import load_testing_data, load_training_data
from bpnn import BPNN

trainingData, trainingLabel, positive, negative = load_training_data('bank-additional/bank-additional.csv')
#print(trainingData)
#svm, acc, stdv = train(trainingData, trainingLabel)
#print('acc:', acc, '; stdv:', stdv)
testingData, testingLabel = load_testing_data('bank-additional/bank-additional-full.csv')
'''
svm = load_model()
test(svm, testingData, testingLabel)
'''
bpnn = BPNN(trainingData.shape[1], math.ceil(0.8*trainingData.shape[1]), 1)
bpnn.load_weights('bank2.weights')
bpnn.test(testingData, testingLabel)
