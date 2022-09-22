#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 08:57:57 2022

@author: MichelQu
"""

import csv 
from datetime import datetime
import time 

import matplotlib.pyplot as plt
import numpy as np

from sklearn import decomposition, tree
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix

###----------------------------------
#           Import Data 
###----------------------------------

file = open('./UTKFaces/labels.csv')
csvreader = csv.reader(file)
labels = []
for row in csvreader : 
    labels.append(row)
labels = np.array(labels).astype(int)
file.close()

ageLabels = labels[:,0]
genderLabels = labels[:,1]
# raceLabels = labels[:,2]

###----------------------------------
#           Useful Function
###----------------------------------

def conversionGray() :
    print(f'---------- Loading {n_picture} images in Gray --------- ')
    listGray = []
    start = time.time()
    print('Loading data - Beginning : ', datetime.now())
    for i,x in enumerate(ageIndex) :
        if (i < n_picture) :
            listGray.append(rgb2gray(plt.imread(f'./UTKFaces/Faces/{i}.jpg')).reshape(1,-1)[0])
    print('Loading data - End Time : ', datetime.now())
    print(f'       Total time : {round(time.time()-start,2)}s')
    return np.array(listGray)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

###----------------------------------
#           Main
###----------------------------------

#%% #

# We choose the people between 20 - 30 yo
ageIndex = []

for i,x in enumerate(ageLabels) : 
    if ( x > 24 and x < 36 ) : 
        ageIndex.append(i)

# We convert the pictures of those guys 
print(f'There are {len(ageIndex) } of people between 25 and 35 y/o')

###----------------------------------
#           Main
###----------------------------------

#%% #
n_picture = 200 #up to 23705
opt_Components = 100

X = conversionGray()
n,p = np.shape(X)

### PCA Decomposition 
print('\nStart PCA decomposition')
start = time.time()

model_PCA = decomposition.PCA(n_components=opt_Components)
X_new = model_PCA.fit_transform(X)

print(f'The explained variance is : {np.sum(model_PCA.explained_variance_ratio_)}')

temps = round(time.time()-start,2)
print(f'    Total time : {temps}s')