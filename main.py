import csv 
import time 
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import xlsxwriter

from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
print(f'Part 2 : There are {len(ageIndex)} people who are 25 (This will be our dataset)')

###----------------------------------
#        3. Run the experiment
###----------------------------------

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

###----------------------------------
#        4. Data pre-processing
###----------------------------------

print(f'Part 4 : We import our ratings of the {len(ageIndex)} images')
filepath = './our_Ratings.xlsx'
# Import the excel file
df = pd.read_excel(filepath)
# Normalized the Ratings
df['Normalized_Rating'] = (df['Rating']-df['Rating'].min())/(df['Rating'].max()-df['Rating'].min())
# Histogram of Normalized data
df.hist(column = 'Normalized_Rating', bins = 7)
plt.suptitle('Histogram of our Normalized Ratings', fontsize = 16)
plt.title('from 0 (Male) to 1 (Female)', fontsize = 10)
plt.xlabel('Normalized Rating')
plt.ylabel('Frequency')
plt.show()

###---------------------------------------
#        5. PCA and Feature Selection
###---------------------------------------

print('Part 5 : We will do a PCA decomposition of our dataset and we will select the best PCs components')

# We import the data
n_picture = len(ageIndex) # Up to 734
X_data = conversionGray()
n,p = np.shape(X_data)

# Normalize the dataset (Don't standardise)
X_mean = X_data.mean(); X_max = X_data.max(); X_min = X_data.min();
X = (X_data-X_min)/(X_max-X_min); y = (df['Normalized_Rating'].values);
# The testing set 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.02, random_state = 42)

# This function plots the reconstruted picture.
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

# Action of PCs - The effect of the PCs on the decomposition 
def PCsAction(pc_number = 10) : 
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
        axarr[0][0].axis('off')
        axarr[0][1].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        axarr[0][1].set_title('Reconstructed Picture')
        axarr[0][1].axis('off')
        
        img_initial = ((X[i_mini].reshape(200,200))*(X_max-X_min)) + X_min
        img_reconstructed = ((X_5[i_mini].reshape(200,200))*(X_max-X_min)) + X_min
        axarr[1][0].imshow(img_initial, cmap='gray', vmin=0, vmax=255)
        axarr[1][0].set_title('Initial Picture')
        axarr[1][0].axis('off')
        axarr[1][1].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        axarr[1][1].set_title('Reconstructed Picture')
        axarr[1][1].axis('off')
        f.suptitle(f'The explained variance is {a} for {i} PCs', fontsize=16)
        
        plt.show()
    return 0

see_PC_actions = True; dont_see_PC_action = False;
if dont_see_PC_action : # If True, we plot the action of PCs otherwise we don't
    PCsAction(10)

#---------------------------------------
# --       PCA decomposition 90%
#---------------------------------------

searchOpt = False # True if we don't want to do it 

if not(searchOpt) :
    print('---------------------------------------------- ')
    print('The search for optimal Component is turned off (we use a default parameters)')
    print('          Change the variable of searchOpt if we want to fo the search')
    opt_Components_90 = 60
    print(f'We need {opt_Components_90} components to get at least 90% of the explained variance')
else :
    k = 1; wanted_explained_variance = 0.90;
    # To know how many Component we need
    temp_x = []; temp_y = [];
    print('---------------------------------------------- ')
    print('Start the search of Optimal Component of PCA decomposition')
    start = time.time()
    while searchOpt :
        model_PCA = decomposition.PCA(n_components=k)
        model_PCA.fit(X_train)
        a = np.cumsum(model_PCA.explained_variance_ratio_)[-1]
        temp_x.append(k); temp_y.append(a)
        if(a > wanted_explained_variance) :
            searchOpt = False
            break
        k += 1
    opt_Components_90 = k
    print(f'We need {opt_Components_90} components to get {wanted_explained_variance*100}% of the explained variance (here, we have : {a})')
    plt.plot(temp_x,temp_y)
    plt.grid()
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.title('Search for the optimal number of Components')
    plt.show()
    print(f'    Total time : {round(time.time()-start,2)}s')

### PCA Decomposition 
print('---------------------------------------------- ')
print('Start the PCA decomposition')
start = time.time()

model_PCA = decomposition.PCA(n_components=opt_Components_90)
X_pca = model_PCA.fit_transform(X_train)

print(f'    The explained variance is : {np.sum(model_PCA.explained_variance_ratio_)}')

temps = round(time.time()-start,2)
print(f'        Total time : {temps}s')
print('End PCA decomposition')

###--------------------------------------------------------------------
#               6. Select a subset of revelant PCs
###--------------------------------------------------------------------

from sklearn.feature_selection import SequentialFeatureSelector

### Feature Selection 
print('---------------------------------------------- ')
print('Part 6 : Start Feature Selection')
start = time.time()

reg = LinearRegression()
sfs = SequentialFeatureSelector(reg,n_features_to_select=20)
X_pca_selected = sfs.fit_transform(X_pca,y_train)

selected_features = (sfs.get_support()*1).sum()
index_of_chosen_PC = (np.argwhere((sfs.get_support()*1)>0.5)).reshape(-1) + 1
print(f'    We selected {selected_features} features')
print(f'    The index chosen are : {index_of_chosen_PC}')

temps = round(time.time()-start,2)
print(f'        Total time : {temps}s')
print('End Feature Selection')

###----------------------------------
#        7. Linear Model

###----------------------------------

# Fit the Linear Regression
print('---------------------------------------------- ')
print('Part 7 : We fit the linear regression model with the training dataset')

reg = LinearRegression().fit(X_pca_selected, y_train)
slope = reg.coef_
intercept = reg.intercept_

print(f'     The slope of the Linear Regression model is : \n {slope}')
print(f'     The slope of the Linear Regression model is : {intercept}')

###--------------------------------------------------------------------
#               8. Generate samples from the model
###--------------------------------------------------------------------

print('---------------------------------------------- ')
print('Part 8-1 : We generate some synthetic pictures')

def synthetic_faces(x_synthetic, x_initial) : 
    n = len(x_synthetic)
    plt.figure()
    f, axarr = plt.subplots(n,2, figsize = (16,50))
    for i in range (n) : 
        img_initial = ((x_initial[i].reshape(200,200))*(X_max-X_min)) + X_min
        img_reconstructed = ((x_synthetic[i].reshape(200,200))*(X_max-X_min)) + X_min   
        axarr[i][0].imshow(img_initial, cmap='gray', vmin=0, vmax=255)
        #axarr[i][0].set_title('Initial Picture')
        axarr[i][0].axis('off')
        axarr[i][1].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
        #axarr[i][1].set_title('Synthetic face')
        axarr[i][1].axis('off')
    f.suptitle('Initial Picture and Associated Synthetic Picture', fontsize = 50)
    plt.show()
    return 0

# Generate images in of our data set
nb_synthetic = 8
print(f'      We reconstruct {nb_synthetic} images from our dataset')
index_selected = random.choices(np.arange(0,len(y_train),1),k=nb_synthetic)
synthetic_in = y_train[index_selected]
alpha_in = [(y-intercept)/(np.transpose(slope)@slope) for y in synthetic_in]
x_synthetic_in = [alp*slope for alp in alpha_in]
x_synthetic_in = sfs.inverse_transform(x_synthetic_in)
x_synthetic_in = model_PCA.inverse_transform(x_synthetic_in)

synthetic_faces(x_synthetic_in,X_train[index_selected])

print('Part 8-2 : We generate the continuum picture')

# Do the continuum image
ratings = np.array([1,2,3,4,5,6,7])
ratings_normalized = (ratings - ratings.min())/(ratings.max()-ratings.min())
alpha_continuum = [(y-intercept)/(np.transpose(slope)@slope) for y in ratings_normalized]
x_synthetic_continuum = [alp*slope for alp in alpha_continuum]
x_synthetic_continuum = sfs.inverse_transform(x_synthetic_continuum)
x_synthetic_continuum = model_PCA.inverse_transform(x_synthetic_continuum)

def plotting_Continuum(x_continuum = x_synthetic_continuum, plotting = True) : 
    images = []
    for x in x_continuum : 
        images.append( ((x.reshape(200,200))*(X_max-X_min)) + X_min )
    images = np.array(images)
    # We build the continuum    
    continuum = np.concatenate([images[0],images[1],images[2],images[3],images[4],images[5],images[6]],axis=1)
    if plotting :
        plt.imshow(continuum, cmap='gray',vmin = 0,vmax = 255)
        plt.axis('off')
        plt.title("The Continuum Picture")
        plt.savefig('continuum_image.png')
        plt.show()
    return (images)

plotting_Continuum(x_synthetic_continuum,True)

###--------------------------------------------------------------------
#               9. Set up a second experiment
###--------------------------------------------------------------------

#%%

print('---------------------------------------------- ')
print('Part 9-1 : We do the Second experiment')

# Creation of the Second Experiment  

images = plotting_Continuum(x_synthetic_continuum, plotting = False)
images = np.repeat(images,10,axis=0) # Creates 70 images (10 times)
ratings_images = np.repeat(ratings,10)

# We shuffle all the pictures
c = list(zip(images,ratings_images))
np.random.shuffle(c)
images_shuffled, ratings_images_shuffled = zip(*c)

# Turn it on True if you want to rate yourself a random sub-dataset 
do_Ratings_Q9 = True 

if do_Ratings_Q9 :
    name = input('What is your name ?   ' )
    name = name.lower()
    # Save the images that we will evaluate in this task with the true ratings 
    f, axarr = plt.subplots(10,7, figsize=(100,100))
    for k in range (len(images_shuffled)) : 
        image = images_shuffled[k]
        true_rating = ratings_images_shuffled[k]
        i = k//7
        j = k%7
        axarr[i][j].imshow(image, cmap='gray', vmin=0, vmax=255)
        axarr[i][j].axis('off')
        axarr[i][j].set_title(f'True rate of image {k} : {true_rating}', fontsize = 48)
    f.suptitle(f'{name} dataset')
    plt.savefig(f'{name}_dataset.png')
    plt.show()
    
    print("\nYou can begin the rating of the continuum images")
    # We rate all the pictures now
    Ratings = []
    for i in range (len(images_shuffled)) :
        image = images_shuffled[i]
        true_rating = ratings_images_shuffled[i]
        # We plot the picture we want to evaluate 
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Picture n°{i}')
        plt.axis('off')
        plt.show()
        # We guarantee that the rating is an integer with this while loop
        while True:
            try : 
                rating = input(f'Picture n°{i+1} : Rating of gender from 1 (Male) to 7 (Female) :   ')
                rating = int(rating)
                if (rating > 7 or rating < 1) : 
                    rating = int('a')
                else : 
                    break
            except Exception : 
                print('The rating is not a number or out of range, try again')
        Ratings.append((f'{i}',rating,true_rating))
    print('--- Rating task done ! ----')
    
    # Store the rates in a excel file 
    workbook = xlsxwriter.Workbook(f'{name}_ratings.xlsx')
    worksheet = workbook.add_worksheet()
    
    row = 0; col = 0;
    for row, item in enumerate(Ratings) : 
        if (row == 0) : 
            # We set the name of the column 
            worksheet.write(row, 0 , 'Image_Index')
            worksheet.write(row, 1 , 'Rating')
            worksheet.write(row, 2 , 'True_Rating')
            # Fill excel with the first element, Add first element 
            worksheet.write(row+1, 0 , item[0]) # The index of the picture
            worksheet.write(row+1, 1 , item[1]) # The rating we gave
            worksheet.write(row+1, 2 , item[2]) # The true rating of the picture
        else :
            # We fill with the data
            worksheet.write(row+1, 0 , item[0]) # The index of the picture
            worksheet.write(row+1, 1 , item[1]) # The rating we gave
            worksheet.write(row+1, 2 , item[2]) # The true rating of the picture
        row += 1
    workbook.close()
    
#%%

print('Part 9-2 : We analyze the data from the second experiment')

filenames = ['./records/michel_ratings.xlsx', './records/arthur_ratings.xlsx']

ratings_hat = []; rating_true = [];

# We extract all data we need from the excel files
for file in filenames :
    dataframe = pd.read_excel(file)
    ratings_hat.append(dataframe['Rating'].values)
    rating_true.append(dataframe['True_Rating'].values)
    
ratings_hat = (np.array(ratings_hat)).reshape(-1)
gender_hat = []
for x in ratings_hat : 
    # 0 = male and 1 = female 
    if (x > 4) : 
        gender_hat.append(1)
    if (x < 4) : 
        gender_hat.append(0)
gender_hat = np.array(gender_hat)
        
rating_true = (np.array(rating_true)).reshape(-1)
gender_true = []
for x in rating_true : 
    # 0 = male and 1 = female 
    if (x > 4) : 
        gender_true.append(1)
    if (x < 4) : 
        gender_true.append(0)
gender_true = np.array(gender_true)

assert np.shape(gender_true) == np.shape(gender_hat), f'The shape doesnt coincide {np.shape(gender_true)} and {np.shape(gender_hat)}'

# ROC Curve
# We build the ROC curve
from sklearn import metrics 
fpr, tpr, thresholds = metrics.roc_curve(gender_true,gender_hat)

x = np.arange(0,1.1,0.1)
plt.plot(fpr,tpr, label = 'ROC Curve')
plt.plot(x,x, 'r--', label = 'y=x')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()



# Psychometric Function














