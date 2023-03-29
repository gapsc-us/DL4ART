#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 08:50:14 2021

@author: Murillo, Delgado

Version 2: We incorporate in create_datasetV9 the possibility of a seed as a last argument of the function. This way the data generated would be always the same.

Version 3: We incorporate in create_datasetV10 the histogram-equalization of each crop in order to test the ANN's performance with equalized images

Version 4: create_datasetV10DAV7 with dataAugmentationV7 (paper of segmentation)

Vserion 5: Create regVGGFC to include an expanded version of FC.
"""

# Common imports:
import numpy as np
import os
import random
#import pandas as pd

# Set a seed in order to obtain the same data augmentation for images and labels. 
#import random
#random.seed(30)

# For Tif images reading:
from PIL import Image


# To plot pretty figures:
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# tensorflow and keras imports:
import tensorflow as tf

#import skimage.io as io
#import skimage.transform as trans
#JJMF 20211118 Added tensorflow. Otherwise I got problems in the call to model.compile(optimizer = Adam....) as Adam is not #recognized
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
#from tensorflow.keras.optimizers import *
#from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import cv2 as cv

# Set some parameters
im_width = 200
im_height = 200
border = 0
X = []


" Function to use a custom sigmoid "
def custom_sigmoid(x):
    return 1 / (1 + K.exp(-7*x))

"Function to plot binary images"        
def plot_image(image):
    plt.imshow(255-image, cmap="binary")
    plt.axis("off")
    
"Function to show results from predictions on a set made with a trained CNN"        
def show_reconstructions(model, images=X, n_images=20):
    np.random.seed(41)
    randomIdx = np.random.permutation(len(images))
    idxs=randomIdx[:n_images]	
    reconstructions = model.predict(images[idxs]) #images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[idxs[image_index]])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        im = np.reshape(reconstructions[image_index],(200,200))
        plot_image(im)


def show_reconstructions_save(model, images=X, n_images=20):
    np.random.seed(41)
    randomIdx = np.random.permutation(len(images))
    idxs=randomIdx[:n_images]	
    reconstructions = model.predict(images[idxs]) #images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[idxs[image_index]])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        im = np.reshape(reconstructions[image_index],(200,200))
        #plot_image(im)
        imageMatrix = Image.fromarray(im)
        imageMatrix.save('/res/Cp_' + str(image_index) + '.tif')

        

"Function to plot learning and loss results from CNN training"    
def Learning_results(results):
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    
       
    
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter    
#JJMF added 20220417
def elastic_transform(image, alpha, sigma, random_state_seed=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state_seed is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(random_state_seed)
        
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)  
    

    
"Data Augmentation function based on V7 with angles between 1º-3.5º and 4º-6º"    
def dataAugmentationV8(X,Img,maxAngle=15,num_rotations=1, seed=73, elastic = False, alpha = 40, sigma = 6):
    random.seed(seed) #JJMF 20220413
    XA = []
    
    XA.append(np.array(Img)[0:200,0:200])
    XA.append(np.array(Img)[100:300,0:200])
    XA.append(np.array(Img)[0:200,100:300])
    XA.append(np.array(Img)[100:300,100:300])
    XA.append(np.array(Img)[50:250,50:250])
    XA.append(np.array(Img)[65:265,65:265])
    XA.append(np.array(Img)[35:235,35:235])
    XA.append(np.array(Img)[80:280,80:280])

    ImgLR = Img.transpose(Image.FLIP_LEFT_RIGHT)
    # Images  flipped horizontally, then taking the first 200x200 pixels 
    XA.append(np.array(ImgLR)[0:200,0:200])
    XA.append(np.array(ImgLR)[100:300,0:200])
    XA.append(np.array(ImgLR)[0:200,100:300])
    XA.append(np.array(ImgLR)[100:300,100:300])
    XA.append(np.array(ImgLR)[50:250,50:250])
    XA.append(np.array(ImgLR)[65:265,65:265])
    XA.append(np.array(ImgLR)[35:235,35:235])
    XA.append(np.array(ImgLR)[80:280,80:280])

    ImgTB = Img.transpose(Image.FLIP_TOP_BOTTOM)
    # Images  flipped horizontally, then taking the first 200x200 pixels 
    XA.append(np.array(ImgTB)[0:200,0:200])
    XA.append(np.array(ImgTB)[100:300,0:200])
    XA.append(np.array(ImgTB)[0:200,100:300])
    XA.append(np.array(ImgTB)[100:300,100:300])
    XA.append(np.array(ImgTB)[50:250,50:250])
    XA.append(np.array(ImgTB)[65:265,65:265])
    XA.append(np.array(ImgTB)[35:235,35:235])
    XA.append(np.array(ImgTB)[80:280,80:280])
    
    # Random rotations:
    for k1 in range(num_rotations):
        #angle = random.randint(2,7)
        angle = (random.random()*5+2)/2
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])        
        
    for k1 in range(num_rotations):
        #angle = random.randint(8,12)
        angle = (random.random()*4+8)/2
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])     
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])             
        
    for k1 in range(num_rotations):
        #angle = random.randint(-7,-2)
        angle = (random.random()*5-7)/2
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265]) 
        
    for k1 in range(num_rotations):
        #angle = random.randint(-12,-8)
        angle = (random.random()*4-12)/2
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])     
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])  
        
    if elastic:
        for k1 in range(len(XA)):
            XA.append(elastic_transform(XA[k1], alpha, sigma, random_state_seed=seed))    
    
    X.extend(XA) 
    
    
"Data Augmentation function based on V7 with angles between 2º-7º and 8º-12º"    
def dataAugmentationV8_GiroCompleto(X,Img,maxAngle=15,num_rotations=1, seed=73, elastic = False, alpha = 40, sigma = 6):
    random.seed(seed) #JJMF 20220413
    XA = []
    
    XA.append(np.array(Img)[0:200,0:200])
    XA.append(np.array(Img)[100:300,0:200])
    XA.append(np.array(Img)[0:200,100:300])
    XA.append(np.array(Img)[100:300,100:300])
    XA.append(np.array(Img)[50:250,50:250])
    XA.append(np.array(Img)[65:265,65:265])
    XA.append(np.array(Img)[35:235,35:235])
    XA.append(np.array(Img)[80:280,80:280])

    ImgLR = Img.transpose(Image.FLIP_LEFT_RIGHT)
    # Images  flipped horizontally, then taking the first 200x200 pixels 
    XA.append(np.array(ImgLR)[0:200,0:200])
    XA.append(np.array(ImgLR)[100:300,0:200])
    XA.append(np.array(ImgLR)[0:200,100:300])
    XA.append(np.array(ImgLR)[100:300,100:300])
    XA.append(np.array(ImgLR)[50:250,50:250])
    XA.append(np.array(ImgLR)[65:265,65:265])
    XA.append(np.array(ImgLR)[35:235,35:235])
    XA.append(np.array(ImgLR)[80:280,80:280])

    ImgTB = Img.transpose(Image.FLIP_TOP_BOTTOM)
    # Images  flipped horizontally, then taking the first 200x200 pixels 
    XA.append(np.array(ImgTB)[0:200,0:200])
    XA.append(np.array(ImgTB)[100:300,0:200])
    XA.append(np.array(ImgTB)[0:200,100:300])
    XA.append(np.array(ImgTB)[100:300,100:300])
    XA.append(np.array(ImgTB)[50:250,50:250])
    XA.append(np.array(ImgTB)[65:265,65:265])
    XA.append(np.array(ImgTB)[35:235,35:235])
    XA.append(np.array(ImgTB)[80:280,80:280])
    
    # Random rotations:
    for k1 in range(num_rotations):
        #angle = random.randint(2,7)
        angle = random.random()*5+2
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])        
        
    for k1 in range(num_rotations):
        #angle = random.randint(8,12)
        angle = random.random()*4+8
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])     
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])             
        
    for k1 in range(num_rotations):
        #angle = random.randint(-7,-2)
        angle = random.random()*5-7
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265]) 
        
    for k1 in range(num_rotations):
        #angle = random.randint(-12,-8)
        angle = random.random()*4-12
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])     
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])  
        
    if elastic:
        for k1 in range(len(XA)):
            XA.append(elastic_transform(XA[k1], alpha, sigma, random_state_seed=seed))    
    
    X.extend(XA)     
    
    
    
"Data Augmentation function to create a varied set of samples in order to train our CNN"    
def dataAugmentationV10(X,Img,maxAngle=15,num_rotations=1, seed=73, elastic = False, alpha = 40, sigma = 6):
    random.seed(seed) #JJMF 20220413

    XA = []
    
    XA.append(np.array(Img)[0:200,0:200])
    XA.append(np.array(Img)[100:300,0:200])
    XA.append(np.array(Img)[0:200,100:300])
    XA.append(np.array(Img)[100:300,100:300])
    
    ImgT = Img.transpose(Image.TRANSPOSE)
    XA.append(np.array(ImgT)[0:200,0:200])
    XA.append(np.array(ImgT)[100:300,0:200])
    XA.append(np.array(ImgT)[0:200,100:300])
    XA.append(np.array(ImgT)[100:300,100:300])     

    ImgLR = Img.transpose(Image.FLIP_LEFT_RIGHT)
    # Images  flipped horizontally, then taking the first 200x200 pixels 
    XA.append(np.array(ImgLR)[0:200,0:200])
    XA.append(np.array(ImgLR)[100:300,0:200])
    XA.append(np.array(ImgLR)[0:200,100:300])
    XA.append(np.array(ImgLR)[100:300,100:300])
    
    ImgLRT = ImgLR.transpose(Image.TRANSPOSE)
    # Images  flipped horizontally, then taking the first 200x200 pixels 
    XA.append(np.array(ImgLRT)[0:200,0:200])
    XA.append(np.array(ImgLRT)[100:300,0:200])
    XA.append(np.array(ImgLRT)[0:200,100:300])
    XA.append(np.array(ImgLRT)[100:300,100:300])    

    ImgTB = Img.transpose(Image.FLIP_TOP_BOTTOM)
    # Images  flipped horizontally, then taking the first 200x200 pixels 
    XA.append(np.array(ImgTB)[0:200,0:200])
    XA.append(np.array(ImgTB)[100:300,0:200])
    XA.append(np.array(ImgTB)[0:200,100:300])
    XA.append(np.array(ImgTB)[100:300,100:300])
    
    ImgTBT = ImgTB.transpose(Image.TRANSPOSE)
    # Images  flipped horizontally, then taking the first 200x200 pixels 
    XA.append(np.array(ImgTBT)[0:200,0:200])
    XA.append(np.array(ImgTBT)[100:300,0:200])
    XA.append(np.array(ImgTBT)[0:200,100:300])
    XA.append(np.array(ImgTBT)[100:300,100:300])    
    
    # Random rotations:
    for k1 in range(num_rotations):
        #angle = random.randint(2,7)
        angle = random.random()*5+2
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        angle = random.random()*5+2
        ImgA = ImgT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        angle = random.random()*5+2
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        angle = random.random()*5+2
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        angle = random.random()*5+2
        ImgA = ImgLRT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250]) 
        XA.append(np.array(ImgA)[65:265,65:265])
        angle = random.random()*5+2
        ImgA = ImgTBT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])        
        
        angle = -( random.random()*5+2)
        ImgA = Img.rotate(angle)        
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])      
        angle = -( random.random()*5+2)        
        ImgA = ImgT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])         
        XA.append(np.array(ImgA)[65:265,65:265])
        angle = -( random.random()*5+2)        
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        angle = -( random.random()*5+2)        
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        XA.append(np.array(ImgA)[65:265,65:265])
        angle = -( random.random()*5+2)        
        ImgA = ImgLRT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250]) 
        XA.append(np.array(ImgA)[65:265,65:265])        
        angle = -( random.random()*5+2)        
        ImgA = ImgTBT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250]) 
        XA.append(np.array(ImgA)[65:265,65:265])        
        
        
    for k1 in range(num_rotations):
        #angle = random.randint(8,12)
        angle = random.random()*4+8
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        angle = random.random()*4+8        
        ImgA = ImgT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        angle = random.random()*4+8        
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])   
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        angle = random.random()*4+8        
        ImgA = ImgLRT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        angle = random.random()*4+8        
        ImgA = ImgTBT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])           

#        angle = -1 * angle
        #angle = random.randint(-12,-8)
        angle = random.random()*4-12
        ImgA = Img.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])
        angle = random.random()*4-12
        ImgA = ImgT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])        
        angle = random.random()*4-12
        ImgA = ImgLR.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])     
        angle = random.random()*4-12
        ImgA = ImgTB.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])  
        angle = random.random()*4-12
        ImgA = ImgLRT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])     
        angle = random.random()*4-12
        ImgA = ImgTBT.rotate(angle)
        XA.append(np.array(ImgA)[50:250,50:250])         
        
    if elastic:
        for k1 in range(len(XA)):
            XA.append(elastic_transform(XA[k1], alpha, sigma, random_state_seed=seed))
     
    X.extend(XA)
    
    
    
# def show_reconstructionsOffset(model, images, n_images=10, offset = 0):
#     np.random.seed(41)
#     randomIdx = np.random.permutation(len(X_test))
#     idxs=randomIdx[:n_images]
#     reconstructions = model.predict(images[idxs+ offset]) #images[:n_images])
#     fig = plt.figure(figsize=(n_images * 1.5, 3))
#     for image_index in range(n_images):
#         plt.subplot(2, n_images, 1 + image_index)
#         plot_image(255-images[idxs[image_index]+ offset])
#         plt.subplot(2, n_images, 1 + n_images + image_index)
#         plot_image(255-reconstructions[image_index])
        
 #JJMF 20230324 Same as v10 belowbut with no equalization       
        
def create_datasetV9(folderCrops, folderNPZ, idxImgs, indexCanvas, labelsPerImage, trainImages, validImages, testImages, giro=False, seed=None):
    
    #In this version, if a value for seed is set we generate the same dataset
    #Note that if below no argument for seed is given to dataAugmentationV8 the seed 73 is used. 
    
    if seed: #JJMF20230201 We might generate same dataset every time
        random.seed(seed)
        
    numberRotations = 1
    Elastic = False
    
    cropsDir = folderCrops
    X_train_full=[]

    # Retrieve labels -> Y_train_full 
    npzDir = folderNPZ 
    aux_train_full=[]

    dataAugmentationNumber = numberRotations*(18) + 12*2
    if Elastic:
        dataAugmentationNumber = 2 * dataAugmentationNumber
    lastK1 = []

    labelNumber = 0
    counter=0
    Rot90Flag = 'NoRot90'
    
    print('Creating dataset...')
    for k1 in idxImgs: # k1 -> Index for list of images
        for k2,_ in enumerate(indexCanvas[k1]):

            ##### Si es el mismo crop que el anterior lo roto...
            seedDA = random.randrange(2**32 - 1)
            seedDA2 = seedDA #random.randrange(2**32 - 1)
            if k1 == lastK1:

                Img = Image.open(cropsDir+indexCanvas[k1][k2]+".tif")
                Img2 = Img.transpose(Image.ROTATE_90)
                np.random.seed(seedDA)
                
                if not(giro):
                    dataAugmentationV8(X_train_full,Img2,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)  
                else:
                    dataAugmentationV8_GiroCompleto(X_train_full,Img2,num_rotations=numberRotations, elastic = Elastic, seed = seedDA) 
                
                fileNPZ = np.load(npzDir+indexCanvas[k1][k2]+'_V.npz')
                cropNPZ = fileNPZ[fileNPZ.files[0]]  
                fileNPZ2 = np.load(npzDir+indexCanvas[k1][k2]+'_H.npz')
                cropNPZ2 = fileNPZ2[fileNPZ2.files[0]]
                cross_points = cropNPZ & cropNPZ2
                imgNPZ = Image.fromarray(np.uint8(255*cross_points) , 'L')
                ImgNPZrot = imgNPZ.transpose(Image.ROTATE_90)

                np.random.seed(seedDA2)
                #We perform this also to the labels
                if not(giro):
                    dataAugmentationV8(aux_train_full,ImgNPZrot,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)  
                else:
                    dataAugmentationV8_GiroCompleto(aux_train_full,ImgNPZrot,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)              
                counter += dataAugmentationNumber

            else:    

                ##### Comportamiento habitual

                Img = Image.open(cropsDir+indexCanvas[k1][k2]+".tif")

                np.random.seed(seedDA)
                if not(giro):
                    dataAugmentationV8(X_train_full,Img,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)  
                else:
                    dataAugmentationV8_GiroCompleto(X_train_full,Img,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)           
                fileNPZ = np.load(npzDir+indexCanvas[k1][k2]+'_V.npz')
                cropNPZ = fileNPZ[fileNPZ.files[0]]
                fileNPZ2 = np.load(npzDir+indexCanvas[k1][k2]+'_H.npz')
                cropNPZ2 = fileNPZ2[fileNPZ2.files[0]]
                cross_points = cropNPZ & cropNPZ2
                imgNPZ = Image.fromarray(np.uint8(255*cross_points) , 'L')

                np.random.seed(seedDA2)
                #We perform this also to the labels
                if not(giro):
                    dataAugmentationV8(aux_train_full,imgNPZ,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)  
                else:
                    dataAugmentationV8_GiroCompleto(aux_train_full,imgNPZ,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)    
                counter += dataAugmentationNumber

        lastK1 = k1
    
    
    labelsPerImageA = np.asarray(labelsPerImage)
    trainIdx =  np.arange(np.sum(labelsPerImageA[trainImages]) * dataAugmentationNumber)             
    validIdx = np.arange(trainIdx[-1:]+1,trainIdx[-1:]+1+np.sum(labelsPerImageA[validImages])* dataAugmentationNumber)
    testIdx  = np.arange(validIdx[-1:]+1,validIdx[-1:]+1+np.sum(labelsPerImageA[testImages]) * dataAugmentationNumber) 

    X_train_full = np.array(X_train_full)
    aux_train_full = np.array(aux_train_full)
    
    print('Rotating 90º each 200x200 crop...')
    # Rotate 90º each 200x200 crop
    X_rot_train_full = np.zeros((len(X_train_full)*2,200,200))
    aux_rot_train_full = np.zeros((len(aux_train_full)*2,200,200))

    X_rot_train_full[np.arange(0,len(X_rot_train_full),2)] = X_train_full
    aux_rot_train_full[np.arange(0,len(aux_rot_train_full),2)] = aux_train_full

    idxRot = np.arange(1,len(X_rot_train_full),2)

    for k1 in idxRot:
        X_rot_train_full[k1] = np.rot90(X_rot_train_full[k1-1],1,(0,1))
        aux_rot_train_full[k1] = np.rot90(aux_rot_train_full[k1-1],1,(0,1))

    del X_train_full, aux_train_full
    Rot90Flag = 'Rot90'
    dataAugmentationNumber = 2 * dataAugmentationNumber

    # Overwrite new index for each group
    # 2*dataAugmentationNumber is considered because we have rotated each 200x200 crop
    trainIdx =  np.arange(np.sum(labelsPerImageA[trainImages]) * dataAugmentationNumber)             
    validIdx = np.arange(trainIdx[-1:]+1,trainIdx[-1:]+1+np.sum(labelsPerImageA[validImages]) * dataAugmentationNumber)
    testIdx  = np.arange(validIdx[-1:]+1,validIdx[-1:]+1+np.sum(labelsPerImageA[testImages]) * dataAugmentationNumber)     
    
    if Rot90Flag == 'NoRot90':
        X_rot_train_full = X_train_full
        aux_rot_train_full = aux_train_full  
        del X_train_full, aux_train_full 

    # Normalizacion:
    print('Converting to 0-1 range...')
    # Restar minimos
    minimos = X_rot_train_full.min(1).min(1); ind = np.where(minimos != 0)[0]
    for k1 in range(len(ind)):
        X_rot_train_full[ind[k1]] = X_rot_train_full[ind[k1]]-minimos[ind[k1]]

    # Dividir por el maximo
    maximos = X_rot_train_full.max(1).max(1);
    for k1 in range(len(X_rot_train_full)):
        X_rot_train_full[k1] = X_rot_train_full[k1]/maximos[k1]

    aux_rot_train_full = aux_rot_train_full / 255.

    return X_rot_train_full, aux_rot_train_full, dataAugmentationNumber, trainIdx, validIdx, testIdx
                 

def create_datasetV10(folderCrops, folderNPZ, idxImgs, indexCanvas, labelsPerImage, trainImages, validImages, testImages, giro=False, seed=None):
    
    #In this version we include an histogram equalization for each crop
    #Note that if below no argument for seed is given to dataAugmentationV8 the seed 73 is used. 
    
    if seed: #JJMF20230201 We might generate same dataset every time
        random.seed(seed)
        
    numberRotations = 1
    Elastic = False
    
    cropsDir = folderCrops
    X_train_full=[]

    # Retrieve labels -> Y_train_full 
    npzDir = folderNPZ 
    aux_train_full=[]

    dataAugmentationNumber = numberRotations*(18) + 12*2
    if Elastic:
        dataAugmentationNumber = 2 * dataAugmentationNumber
    lastK1 = []

    labelNumber = 0
    counter=0
    Rot90Flag = 'NoRot90'
    
    print('Creating dataset...')
    for k1 in idxImgs: # k1 -> Index for list of images
        for k2,_ in enumerate(indexCanvas[k1]):

            ##### Si es el mismo crop que el anterior lo roto...
            seedDA = random.randrange(2**32 - 1)
            seedDA2 = seedDA #random.randrange(2**32 - 1)
            if k1 == lastK1:

                Img = Image.open(cropsDir+indexCanvas[k1][k2]+".tif")
                Img = cv.equalizeHist(np.array(Img))
                Img2 = Image.fromarray(Img, mode='L').transpose(Image.ROTATE_90)
                np.random.seed(seedDA)
                
                if not(giro):
                    dataAugmentationV8(X_train_full,Img2,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)  
                else:
                    dataAugmentationV8_GiroCompleto(X_train_full,Img2,num_rotations=numberRotations, elastic = Elastic, seed = seedDA) 
                
                fileNPZ = np.load(npzDir+indexCanvas[k1][k2]+'_V.npz')
                cropNPZ = fileNPZ[fileNPZ.files[0]]  
                fileNPZ2 = np.load(npzDir+indexCanvas[k1][k2]+'_H.npz')
                cropNPZ2 = fileNPZ2[fileNPZ2.files[0]]
                cross_points = cropNPZ & cropNPZ2
                imgNPZ = Image.fromarray(np.uint8(255*cross_points) , 'L')
                ImgNPZrot = imgNPZ.transpose(Image.ROTATE_90)

                np.random.seed(seedDA2)
                #We perform this also to the labels
                if not(giro):
                    dataAugmentationV8(aux_train_full,ImgNPZrot,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)  
                else:
                    dataAugmentationV8_GiroCompleto(aux_train_full,ImgNPZrot,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)              
                counter += dataAugmentationNumber

            else:    

                ##### Comportamiento habitual

                Img = Image.open(cropsDir+indexCanvas[k1][k2]+".tif")
                Img = cv.equalizeHist(np.array(Img))
                Img = Image.fromarray(Img, mode='L')

                np.random.seed(seedDA)
                if not(giro):
                    dataAugmentationV8(X_train_full,Img,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)  
                else:
                    dataAugmentationV8_GiroCompleto(X_train_full,Img,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)           
                fileNPZ = np.load(npzDir+indexCanvas[k1][k2]+'_V.npz')
                cropNPZ = fileNPZ[fileNPZ.files[0]]
                fileNPZ2 = np.load(npzDir+indexCanvas[k1][k2]+'_H.npz')
                cropNPZ2 = fileNPZ2[fileNPZ2.files[0]]
                cross_points = cropNPZ & cropNPZ2
                imgNPZ = Image.fromarray(np.uint8(255*cross_points) , 'L')

                np.random.seed(seedDA2)
                #We perform this also to the labels
                if not(giro):
                    dataAugmentationV8(aux_train_full,imgNPZ,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)  
                else:
                    dataAugmentationV8_GiroCompleto(aux_train_full,imgNPZ,num_rotations=numberRotations, elastic = Elastic, seed = seedDA)    
                counter += dataAugmentationNumber

        lastK1 = k1
    
    
    labelsPerImageA = np.asarray(labelsPerImage)
    trainIdx =  np.arange(np.sum(labelsPerImageA[trainImages]) * dataAugmentationNumber)             
    validIdx = np.arange(trainIdx[-1:]+1,trainIdx[-1:]+1+np.sum(labelsPerImageA[validImages])* dataAugmentationNumber)
    testIdx  = np.arange(validIdx[-1:]+1,validIdx[-1:]+1+np.sum(labelsPerImageA[testImages]) * dataAugmentationNumber) 

    X_train_full = np.array(X_train_full)
    aux_train_full = np.array(aux_train_full)
    
    print('Rotating 90º each 200x200 crop...')
    # Rotate 90º each 200x200 crop
    X_rot_train_full = np.zeros((len(X_train_full)*2,200,200))
    aux_rot_train_full = np.zeros((len(aux_train_full)*2,200,200))

    X_rot_train_full[np.arange(0,len(X_rot_train_full),2)] = X_train_full
    aux_rot_train_full[np.arange(0,len(aux_rot_train_full),2)] = aux_train_full

    idxRot = np.arange(1,len(X_rot_train_full),2)

    for k1 in idxRot:
        X_rot_train_full[k1] = np.rot90(X_rot_train_full[k1-1],1,(0,1))
        aux_rot_train_full[k1] = np.rot90(aux_rot_train_full[k1-1],1,(0,1))

    del X_train_full, aux_train_full
    Rot90Flag = 'Rot90'
    dataAugmentationNumber = 2 * dataAugmentationNumber

    # Overwrite new index for each group
    # 2*dataAugmentationNumber is considered because we have rotated each 200x200 crop
    trainIdx =  np.arange(np.sum(labelsPerImageA[trainImages]) * dataAugmentationNumber)             
    validIdx = np.arange(trainIdx[-1:]+1,trainIdx[-1:]+1+np.sum(labelsPerImageA[validImages]) * dataAugmentationNumber)
    testIdx  = np.arange(validIdx[-1:]+1,validIdx[-1:]+1+np.sum(labelsPerImageA[testImages]) * dataAugmentationNumber)     
    
    if Rot90Flag == 'NoRot90':
        X_rot_train_full = X_train_full
        aux_rot_train_full = aux_train_full  
        del X_train_full, aux_train_full 

    # Normalizacion:
    print('Converting to 0-1 range...')
    # Restar minimos
    minimos = X_rot_train_full.min(1).min(1); ind = np.where(minimos != 0)[0]
    for k1 in range(len(ind)):
        X_rot_train_full[ind[k1]] = X_rot_train_full[ind[k1]]-minimos[ind[k1]]

    # Dividir por el maximo
    maximos = X_rot_train_full.max(1).max(1);
    for k1 in range(len(X_rot_train_full)):
        X_rot_train_full[k1] = X_rot_train_full[k1]/maximos[k1]

    aux_rot_train_full = aux_rot_train_full / 255.

    return X_rot_train_full, aux_rot_train_full, dataAugmentationNumber, trainIdx, validIdx, testIdx
                 
          
        
############################################## REGRESSION MODELS ###################################################### 

"""
The following UNET model (and relationed functions) are taken from: 
https://github.com/hlamba28/UNET-TGS/blob/master/TGS%20UNET.ipynb
"""

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x





    #################################### INCEPTION MODELS ###################################

# def InceptionModule(inputs, numFilters = 32):
    
#     tower_0 = Convolution2D(numFilters, (1,1), padding='same', kernel_initializer = 'he_normal')(inputs)
#     tower_0 = BatchNormalization()(tower_0)
#     tower_0 = Activation("relu")(tower_0)
    
#     tower_1 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(inputs)
#     tower_1 = BatchNormalization()(tower_1)
#     tower_1 = Activation("relu")(tower_1)
#     tower_1 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_1)
#     tower_1 = BatchNormalization()(tower_1)
#     tower_1 = Activation("relu")(tower_1)
    
#     tower_2 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(inputs)
#     tower_2 = BatchNormalization()(tower_2)
#     tower_2 = Activation("relu")(tower_2)
#     tower_2 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_2)
#     tower_2 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_2)
#     tower_2 = BatchNormalization()(tower_2)
#     tower_2 = Activation("relu")(tower_2)
    
#     tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
#     tower_3 = Convolution2D(numFilters, (1,1), padding='same',kernel_initializer = 'he_normal')(tower_3)
#     tower_3 = BatchNormalization()(tower_3)
#     tower_3 = Activation("relu")(tower_3)
    
#     inception_module = concatenate([tower_0, tower_1, tower_2, tower_3], axis = 3)
#     return inception_module


def InceptionModule2(inputs, numFilters = 32):
   
    tower_1 = Convolution2D(numFilters, (3,3), padding='same',kernel_initializer = 'he_normal')(inputs)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = Activation("relu")(tower_1)
    
    tower_2 = Convolution2D(numFilters, (5,5), padding='same',kernel_initializer = 'he_normal')(inputs)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = Activation("relu")(tower_2)
    
    tower_3 = Convolution2D(numFilters, (7,7), padding='same',kernel_initializer = 'he_normal')(inputs)
    tower_3 = BatchNormalization()(tower_3)
    tower_3 = Activation("relu")(tower_3)

    
    inception_module = concatenate([tower_1, tower_2, tower_3], axis = 3)
    return inception_module


# def DilatedInceptionModule(inputs, numFilters = 32): 
#     tower_0 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (1,1), kernel_initializer = 'he_normal')(inputs)
#     tower_0 = BatchNormalization()(tower_0)
#     tower_0 = Activation("relu")(tower_0)
    
#     tower_1 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (2,2), kernel_initializer = 'he_normal')(inputs)
#     tower_1 = BatchNormalization()(tower_1)
#     tower_1 = Activation("relu")(tower_1)
    
#     tower_2 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (3,3), kernel_initializer = 'he_normal')(inputs)
#     tower_2 = BatchNormalization()(tower_2)
#     tower_2 = Activation("relu")(tower_2)
    
#     dilated_inception_module = concatenate([tower_0, tower_1, tower_2], axis = 3)
#     return dilated_inception_module


####### INCEPTION REGRESION MODEL USED
def modelo_pruebas(input_shape = (200,200,1), numFilters = 6, dropout = 0.15):

    inputs = Input(input_shape)
    
    conv1 = InceptionModule2(inputs, numFilters)
    conv1 = InceptionModule2(conv1, numFilters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = InceptionModule2(pool1, 2*numFilters)
    conv2 = InceptionModule2(conv2, 2*numFilters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = InceptionModule2(pool2, 4*numFilters)
    conv3 = InceptionModule2(conv3, 4*numFilters)    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = InceptionModule2(pool3, 8*numFilters)
    conv4 = InceptionModule2(conv4, 8*numFilters)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = InceptionModule2(pool4,16*numFilters)
    conv5 = InceptionModule2(conv5, 16*numFilters)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(dropout)(pool5)

    conv6 = InceptionModule2(pool5,16*numFilters)
    conv6 = InceptionModule2(conv6, 16*numFilters)
    #pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    #pool6 = Dropout(dropout)(pool6)

    x = tf.keras.layers.Flatten()(conv6)

    x = tf.keras.layers.Dense(100,activation='relu')(x)
    x = tf.keras.layers.Dense(100,activation='relu')(x)
    x = tf.keras.layers.Dense(80,activation='relu')(x)
    x = tf.keras.layers.Dense(100,activation='relu')(x)
    outputs = tf.keras.layers.Dense(1,activation='linear')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


####### INCEPTION VGG-BASED REGRESION MODELS
def regVGG(input_shape = (200,200,1), numFilters = 6, dropout = 0.15):

    inputs = Input(input_shape)
    
    conv1 = InceptionModule2(inputs, numFilters)
    conv1 = InceptionModule2(conv1, numFilters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = InceptionModule2(pool1, 2*numFilters)
    conv2 = InceptionModule2(conv2, 2*numFilters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = InceptionModule2(pool2, 4*numFilters)
    conv3 = InceptionModule2(conv3, 4*numFilters) 
    conv3 = InceptionModule2(conv3, 4*numFilters)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = InceptionModule2(pool3, 8*numFilters)
    conv4 = InceptionModule2(conv4, 8*numFilters)
    conv4 = InceptionModule2(conv4, 8*numFilters)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = InceptionModule2(pool4,8*numFilters)
    conv5 = InceptionModule2(conv5, 8*numFilters)
    conv5 = InceptionModule2(conv5, 8*numFilters)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(dropout)(pool5)

    x = tf.keras.layers.Flatten()(pool5)
    x = tf.keras.layers.Dense(64*numFilters,activation='relu')(x)
    x = tf.keras.layers.Dense(64*numFilters,activation='relu')(x)
    outputs = tf.keras.layers.Dense(1,activation='linear')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


####### RESIDUAL INCEPTION REGRESSION
# function for creating an identity or projection residual module
def residualInception_module(layer_in, n_filters):
    merge_input = layer_in
    
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:        
        merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
        
    tower_1 = Convolution2D(n_filters, (3,3), padding='same',kernel_initializer = 'he_normal')(layer_in)
    tower_1 = Convolution2D(n_filters, (3,3), padding='same',kernel_initializer = 'he_normal')(tower_1)
    tower_1 = BatchNormalization()(tower_1)
    layer_out1 = add([tower_1, merge_input])
        
    tower_2 = Convolution2D(n_filters, (5,5), padding='same',kernel_initializer = 'he_normal')(layer_in)
    tower_2 = Convolution2D(n_filters, (5,5), padding='same',kernel_initializer = 'he_normal')(tower_2)
    tower_2 = BatchNormalization()(tower_2)
    layer_out2 = add([tower_2, merge_input])
    
    tower_3 = Convolution2D(n_filters, (7,7), padding='same',kernel_initializer = 'he_normal')(layer_in)
    tower_3 = Convolution2D(n_filters, (7,7), padding='same',kernel_initializer = 'he_normal')(tower_3)
    tower_3 = BatchNormalization()(tower_3)
    layer_out3 = add([tower_3, merge_input])
    
    layer_out = concatenate([layer_out1, layer_out2, layer_out3], axis = 3)
    layer_out = Activation('relu')(layer_out)
    return layer_out


def inceptionResNet(input_shape = (200,200,1), numFilters = 6, dropout = 0.15):

    inputs = Input(input_shape)
    
    conv1 = residualInception_module(inputs, numFilters)
    conv1 = residualInception_module(conv1, numFilters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = residualInception_module(pool1, 2*numFilters)
    conv2 = residualInception_module(conv2, 2*numFilters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = residualInception_module(pool2, 4*numFilters)
    conv3 = residualInception_module(conv3, 4*numFilters)    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = residualInception_module(pool3, 8*numFilters)
    conv4 = residualInception_module(conv4, 8*numFilters)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = residualInception_module(pool4,12*numFilters)
    conv5 = residualInception_module(conv5, 12*numFilters)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(dropout)(pool5)

    conv6 = residualInception_module(pool5,14*numFilters)
    conv6 = residualInception_module(conv6, 14*numFilters)
    #pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    pool6 = Dropout(dropout)(conv6) #pool6 = Dropout(dropout)(pool6)

    x = tf.keras.layers.Flatten()(conv6)

    x = tf.keras.layers.Dense(100,activation='relu')(x)
    x = tf.keras.layers.Dense(100,activation='relu')(x)
    x = tf.keras.layers.Dense(80,activation='relu')(x)
    x = tf.keras.layers.Dense(100,activation='relu')(x)
    outputs = tf.keras.layers.Dense(1,activation='linear')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model


