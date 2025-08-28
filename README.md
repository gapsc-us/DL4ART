# DL4ART
Deep learning models for Art

In this repository we include some code used in our latest results in the application of artificial intelligence to the analysis of fabrics in canvases. With these analyses we provide the curator with tools to much improve the historical description of masterpieces. 

## Thread Counting with CNN Models for Regression

In folder RegressionNMAE we include a notebook with the code to train the VGG model in the manuscript

Delgado, A., Murillo-Fuentes, J.J. & Alba-Carcelén, L. Thread Counting in Plain Weave for Old Paintings Using Regression Deep Learning Models. Int J Comput Vis (2025). https://doi.org/10.1007/s11263-025-02473-9 

We uploaded the code to learn the weights of the VGG regression model see Train-Regression-VGGEq-NMAE-BS32.ipynb. It will need the model itsel, provided as RegressionModelsV5gh.py in Utils folder.

We have no permisson to share data. Therefore the code is given for those interested in trying it on their labeled samples.

Samples were generated as 1,5 x 1,5 cm square, 200 pixel per cm, 8 bits grayscale random samples images of X-rays plates of canvases. We labeled the image using labelme and saving the result in a json. 

We include in folder cropping with a labeled sample, Crops0139.tif, so you can see what the input to the training of the model would be.

You should gather your own images, generate samples, label them, and generate the .npz files. The input to the learning are the preprocessed samples and the npz with the labels for the horizontal and vertical threads.

## Forensic Analysis of Fabrics based on Siamese Networks

In Forensic Study of Paintings Through the Comparison of Fabrics
Juan José Murillo-Fuentes, Pablo M. Olmos, Laura Alba-Carcelén
[arXiv:2506.20272 [cs.CV]](https://arxiv.org/abs/2506.20272)

we report recent results on the comparison of fabrics not based on thread counting matching. Traditional methods are based on thread density map matching, which cannot be applied when canvases do not come from contiguous positions on a roll. This paper presents a novel approach based on deep learning to assess the similarity of textiles. We introduce an automatic tool that evaluates the similarity between canvases without relying on thread density maps. A Siamese deep learning model is designed and trained to compare pairs of images by exploiting the feature representations learned from the scans. In addition, a similarity estimation method is proposed, aggregating predictions from multiple pairs of cloth samples to provide a robust similarity score. Our approach is applied to canvases from the Museo Nacional del Prado, corroborating the hypothesis that plain weave canvases, widely used in painting, can be effectively compared even when their thread densities are similar. The results demonstrate the feasibility and accuracy of the proposed method, opening new avenues for the analysis of masterpieces.

In the folder Siamese we will upload the models used once the manuscript is accepted. We have no permisson to share data. Therefore the code is given for those interested in trying it on their samples. We include in folder cropping a sample Crops0139.tif, so you can see what the input to the training of the model would be.



