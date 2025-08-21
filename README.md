# DL4ART
Deep learning models for Art

In this repository we include some code used in our latest results in the application of artificial intelligence to the analysis of fabrics in canvases. With these analyses we provide the curator with tools to much improve the historical description of masterpieces. 

## Thread counting with CNN models for regression

We include the code to train the VGG model in the manuscript "Thread Counting in Plain Weave for Old Paintings Using Semi-Supervised Regression Deep Learning Models"

We have no permisson to share data. Therefore the code is given for those interested in trying it on their labeled samples.

Samples were generated as 1,5 x 1,5 cm square, 200 pixel per cm, 8 bits grayscale random samples images of X-rays plates of canvases. We labeled the image using labelme and saving the result in a json. 

Here you will find:

1.- A labeled sample, Crops0139.tif so you can see what the input to the training of the model would be.

2.- Code to learn the weights of the VGG regression model see Train-Regression-VGGEq-NMAE-BS32.ipynb

You should gather your own images, generate samples, label them, and generate the .npz files. The input to the learning are the preprocessed samples and the npz with the labels for the horizontal and vertical threads.


