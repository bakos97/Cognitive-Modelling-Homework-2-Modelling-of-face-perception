import csv 
from datetime import datetime
import time 

import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn import decomposition
from sklearn import model_selection
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

import seaborn as sns;
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
            listGray.append(rgb2gray(plt.imread(f'./UTKFaces/Faces/{x}.jpg')).reshape(1,-1)[0])
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

genderLabels = [genderLabels[i] for i in ageIndex]

# We convert the pictures of those guys 
print(f'There are {len(ageIndex)} people who are 25 \n')

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
do_Task = False 

if do_Task : 
    for x in ageIndex_Train : 
        image = plt.imread(f'./UTKFaces/Faces/{x}.jpg')
        plt.imshow(image)
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

filepath = './our_Ratings.xlsx'
# Import the excel file
df = pd.read_excel (filepath)
# Normalized the Ratings
df['Normalized_Rating'] = (df['Rating']-df['Rating'].min())/(df['Rating'].max()-df['Rating'].min())
# Histogram of Normalized data
df.hist(column = 'Normalized_Rating', bins = 7)
plt.show()

###---------------------------------------
#        5. PCA and Feature Selection
###---------------------------------------

#%%
# We import the data
n_picture = len(ageIndex) # Up to 734
X_data = conversionGray()
n,p = np.shape(X_data)

# Normalize the dataset (Don't standardise)
X_mean = X_data.mean(); X_max = X_data.max(); X_min = X_data.min();
X = (X_data-X_min)/(X_max-X_min)

#%%
X_test_true = X[n-10:] 
X = X[:n-10]

y = (df['Normalized_Rating'].values)
y_test = y[n-10:] 
y = y[:n-10]
#%%

def plotting_Reconstruction(model_Name,X,X_Reconstructed,n_image2plot,text) :
    # Plot the reconstructed image
    for i in range(n_image2plot) : 
        plt.figure()
        f, axarr = plt.subplots(1,2)
        img_initial = ((X[i].reshape(200,200))*(X_max-X_min)) + X_min
        img_pc = ((X_Reconstructed[i].reshape(200,200))*(X_max-X_min)) + X_min
        img_reconstructed = ((X_Reconstructed[i].reshape(200,200))*(X_max-X_min)) + X_min
        axarr[0].imshow(img_initial, cmap='gray', vmin=0, vmax=255)
        axarr[0].set_title('Initial Picture')
        axarr[1].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        axarr[1].set_title('Reconstructed Picture')
        f.suptitle(text, fontsize=16)
    return 0

for i in range (1,2) : 
    model_PCA = decomposition.PCA(n_components=i)
    X_5 = model_PCA.fit_transform(X)
    X_5 = model_PCA.inverse_transform(X_5)
    a = np.cumsum(model_PCA.explained_variance_ratio_)[-1]
    a = round(a,3)
    plotting_Reconstruction(model_PCA, X, X_5, 1, f'The explained variance is {a} for {i} PCs')

#%%
# Action of PCs - The effect of the PCs on the decomposition 

def PCsAction() : 
    for i in range (1,10) :
        model_PCA = decomposition.PCA(n_components=i)
        X_5 = model_PCA.fit_transform(X)
        a = np.cumsum(model_PCA.explained_variance_ratio_)[-1]
        a = round(a,3)
        X_column = X_5[:,i-1]
        i_maxi = np.where(X_column == X_column.max())
        i_mini = np.where(X_column == X_column.min())
        X_5 = model_PCA.inverse_transform(X_5)
        
        plt.figure()
        f, axarr = plt.subplots(2,4)
        img_initial = X[i_maxi].reshape(200,200) + X_mean
        img_pc = X_5[i_maxi].reshape(200,200)
        img_mean = (np.ones(40000)*X_mean).reshape(200,200)
        img_reconstructed = X_5[i_maxi].reshape(200,200) + X_mean
        axarr[0][0].imshow(img_initial, cmap='gray', vmin=0, vmax=255)
        axarr[0][0].set_title('Initial Picture')
        axarr[0][1].imshow(img_pc, cmap='gray', vmin=0, vmax=255)
        axarr[0][1].set_title('PC Picture')
        axarr[0][2].imshow(img_mean, cmap='gray', vmin=0, vmax=255)
        axarr[0][2].set_title('Mean Picture')
        axarr[0][3].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        axarr[0][3].set_title('Reconstructed Picture')
        
        img_initial = X[i_mini].reshape(200,200) + X_mean
        img_pc = X_5[i_mini].reshape(200,200)
        img_mean = (np.ones(40000)*X_mean).reshape(200,200)
        img_reconstructed = X_5[i_mini].reshape(200,200) + X_mean
        axarr[1][0].imshow(img_initial, cmap='gray', vmin=0, vmax=255)
        axarr[1][0].set_title('Initial Picture')
        axarr[1][1].imshow(img_pc, cmap='gray', vmin=0, vmax=255)
        axarr[1][1].set_title('PC Picture')
        axarr[1][2].imshow(img_mean, cmap='gray', vmin=0, vmax=255)
        axarr[1][2].set_title('Mean Picture')
        axarr[1][3].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        axarr[1][3].set_title('Reconstructed Picture')
        f.suptitle(f'The explained variance is {a} for {i} PCs', fontsize=16)
        
        plt.show()
        return 0
def PCsAction_V2(pc_number = 10) : 
    for i in range (1,pc_number) :
        model_PCA = decomposition.PCA(n_components=i)
        X_5 = model_PCA.fit_transform(X)
        a = np.cumsum(model_PCA.explained_variance_ratio_)[-1]
        a = round(a,3)
        X_column = X_5[:,i-1]
        i_maxi = np.where(X_column == X_column.max())
        i_mini = np.where(X_column == X_column.min())
        X_5 = model_PCA.inverse_transform(X_5)
        
        plt.figure()
        f, axarr = plt.subplots(2,2)
        
        img_initial = ((X[i_maxi].reshape(200,200))*(X_max-X_min)) + X_min
        img_reconstructed = ((X_5[i_maxi].reshape(200,200))*(X_max-X_min)) + X_min
        axarr[0][0].imshow(img_initial, cmap='gray', vmin=0, vmax=255)
        axarr[0][0].set_title('Initial Picture')
        axarr[0][1].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        axarr[0][1].set_title('Reconstructed Picture')
        
        img_initial = ((X[i_mini].reshape(200,200))*(X_max-X_min)) + X_min
        img_reconstructed = ((X_5[i_mini].reshape(200,200))*(X_max-X_min)) + X_min
        axarr[1][0].imshow(img_initial, cmap='gray', vmin=0, vmax=255)
        axarr[1][0].set_title('Initial Picture')
        axarr[1][1].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        axarr[1][1].set_title('Reconstructed Picture')
        f.suptitle(f'The explained variance is {a} for {i} PCs', fontsize=16)
        
        plt.show()
    return 0
PCsAction_V2(10)

###--------------------------------------------------------------------
#               6. Select a subset of revelant PCs
###--------------------------------------------------------------------

#%% 
#%%
# To know how many Component we need
temp_x = []; temp_y = [];
print('\nStart the search of Optimal Component of PCA decomposition')
start = time.time()

k = 1;
searchOpt = True; 
wanted_explained_variance = 0.95

while searchOpt :
    model_PCA = decomposition.PCA(n_components=k)
    model_PCA.fit(X)
    a = np.cumsum(model_PCA.explained_variance_ratio_)[-1]
    temp_x.append(k); temp_y.append(a)
    if(a > wanted_explained_variance) :
        searchOpt = False
        break
    k += 1

print(f'We need {k} components to get {wanted_explained_variance*100}% of the explained variance (here, we have : {a})')
plt.plot(temp_x,temp_y)
plt.grid()
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Search for the optimal number of Components')
plt.show()
print(f'    Total time : {round(time.time()-start,2)}s')

#%%
from sklearn.model_selection import train_test_split

# Normalize the dataset (Don't standardise)
X_mean = X_data.mean(); X_max = X_data.max(); X_min = X_data.min();
X = (X_data-X_min)/(X_max-X_min); y = (df['Normalized_Rating'].values);
# The testing set 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state = 42)
#%%
opt_Components_90 = 57 # k
opt_Components_95 = 122 #k

### PCA Decomposition 
print('\nStart PCA decomposition')
start = time.time()

model_PCA = decomposition.PCA(n_components=opt_Components_95)
X_pca = model_PCA.fit_transform(X_train)

print(f'    The explained variance is : {np.sum(model_PCA.explained_variance_ratio_)}')

temps = round(time.time()-start,2)
print(f'        Total time : {temps}s')
print('End PCA decomposition')
#%%

###--------------------------------------------------------------------
#               6. Select a subset of revelant PCs
###--------------------------------------------------------------------



###----------------------------------
#        7. Linear Model

###----------------------------------

#%%
# Fit the Linear Regression
# list_weight = np.linalg.inv(np.transpose(X)@X) @ (np.transpose(X)@df['Normalized_Rating'].values)
reg = LinearRegression().fit(X_pca, y_train)
slope = reg.coef_
intercept = reg.intercept_

###--------------------------------------------------------------------
#               8. Generate samples from the model
###--------------------------------------------------------------------

def synthetic_faces(x_synthetic, x_initial) : 
    n_synthetic_to_generate = 5
    if (n_synthetic_to_generate > len(x_synthetic)):
        n_synthetic_to_generate = len(x_synthetic)
    for i in range (n_synthetic_to_generate) : 
        plt.figure()
        f, axarr = plt.subplots(1,2)
        img_initial = ((x_initial[i].reshape(200,200))*(X_max-X_min)) + X_min
        img_reconstructed = ((x_synthetic[i].reshape(200,200))*(X_max-X_min)) + X_min   
        axarr[0].imshow(img_initial, cmap='gray', vmin=0, vmax=255)
        axarr[0].set_title('Initial Picture')
        axarr[1].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        axarr[1].set_title(f'Synthetic face Picture {i+1}')
        plt.show()
    return 0

# Generate images in of our data set 
synthetic_in = random.choices(y_train,k=5)
alpha_in = [(y-intercept)/(np.transpose(slope)@slope) for y in synthetic_in]
x_synthetic_in = [alp*slope for alp in alpha_in]
x_synthetic_in = model_PCA.inverse_transform(x_synthetic_in)

synthetic_faces(x_synthetic_in,X_train)

# Generate images out of our data set 
# synthetic_out = random.choices(y_test,k=5)
# alpha_out = [(y-intercept)/(np.transpose(slope)@slope) for y in synthetic_out]
# x_synthetic_out = [alp*slope for alp in alpha_out]
# x_synthetic_out = model_PCA.inverse_transform(x_synthetic_out)

# synthetic_faces(x_synthetic_out,X_test)

###--------------------------------------------------------------------
#               9. Set up a second experiment
###--------------------------------------------------------------------
#%%
y_train = y_train.ravel()
y_test = y_test.ravel()

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)
#%%
# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=5,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=5)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)