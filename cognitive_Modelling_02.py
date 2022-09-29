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

from sklearn import decomposition
from sklearn import model_selection

import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix

###----------------------------------
#           Import Data 
###----------------------------------

# filename = './UTKFaces/labels.csv'
filename = '/Users/MichelQu/Documents/GitHub/Cognitive-Modelling-Homework-2-Modelling-of-face-perception?fbclid=IwAR0WLjkD2X-aoSyhTr23nbiolQL2A30awoG-Dsfo0TfsBfGrPU0-PDICHIk'
file = open(filename + '/UTKFaces/labels.csv')
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
            listGray.append(rgb2gray(plt.imread(filename + f'/UTKFaces/Faces/{i}.jpg')).reshape(1,-1)[0])
    print('Loading data - End Time : ', datetime.now())
    print(f'       Total time : {round(time.time()-start,2)}s')
    return np.array(listGray)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

###----------------------------------
#        2. Select the Images
###----------------------------------

# We choose people who are 25
ageIndex = []

for i,x in enumerate(ageLabels) : 
    if ( x == 25 ) : 
        ageIndex.append(i)

# We convert the pictures of those guys 
print(f'There are {len(ageIndex) } who are 25')

###----------------------------------
#        3. Run the experiment
###----------------------------------
#%% #
import random
import xlsxwriter

n_samples = 5
ageIndex_Train = random.choices(ageIndex, k=n_samples)
ageIndex_Test = []
Ratings = []

# Turn it on True if you want to rate yourself a random sub-dataset 
do_Task = True 
filename = '/Users/MichelQu/Documents/GitHub/Cognitive-Modelling-Homework-2-Modelling-of-face-perception?fbclid=IwAR0WLjkD2X-aoSyhTr23nbiolQL2A30awoG-Dsfo0TfsBfGrPU0-PDICHIk'

if do_Task : 
    for x in ageIndex_Train : 
        image = plt.imread(filename + f'/UTKFaces/Faces/{x}.jpg')
        plt.imshow(image)
        plt.show()
        rating = input('Rating of gender from 1 (Male) to 7 (Female) \n')
        assert type(int(rating)) == int
        Ratings.append((x,rating))
    print('--- Rating task done ! ----')

    workbook = xlsxwriter.Workbook('python_ratings.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0; col = 0;
    # Write on the file 
    for item in Ratings : 
        worksheet.write(row, col,     item[0])
        worksheet.write(row, col + 1, item[1])
        row += 1
    workbook.close()
#%% #

###----------------------------------
#        4. Data pre-processing
###----------------------------------

import pandas as pd

filepath = filename + '/our_Ratings.xlsx'
# Import the excel file
df = pd.read_excel (filepath)
# Normalized the Ratings
df['Normalized_Rating'] = (df['Rating']-df['Rating'].mean())/df['Rating'].std()
# Histogram of Normalized data
df.hist(column = 'Normalized_Rating', bins = 7)

###---------------------------------------
#        5. PCA and Feature Selection
###---------------------------------------

#%% #
n_picture = len(ageIndex) #up to 23705
opt_Components = 3

X = conversionGray()
X = X - X.mean()
n,p = np.shape(X)

### PCA Decomposition 
print('\nStart PCA decomposition')
start = time.time()

model_PCA = decomposition.PCA(n_components=opt_Components)
X_new = model_PCA.fit_transform(X)

print(f'The explained variance is : {np.sum(model_PCA.explained_variance_ratio_)}')

temps = round(time.time()-start,2)
print(f'    Total time : {temps}s')
#%%

###--------------------------------------------------------------------
#               6. Select a subset of revelant PCs
###--------------------------------------------------------------------


###----------------------------------
#        7. Linear Model
###----------------------------------

###--------------------------------------------------------------------
#               8. Generate samples from the model
###--------------------------------------------------------------------

###--------------------------------------------------------------------
#               9. Set up a second experiment
###--------------------------------------------------------------------


