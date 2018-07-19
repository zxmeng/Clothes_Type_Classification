# Clothes_Type_Classification

Fashion MNIST dataset [1] is a newly collected MNIST-like dataset which is expected to help quickly check and prototype Machine Learning algorithms as a benchmark dataset. 

In this project, we apply several classical classification methods, including Perceptron, Ridge Regression, LASSO, Logistic Regression, K-Nearest Neighbours, Support Vector Machine, and Convolutional Neural Network, to this Fashion MNIST dataset to do the clothing type classification and compare their performances. We report the best test accuracy to be 94.67% by using a simple Convolutional Neural Network model. 

Inspired by the data preparation procedure of Fashion MNIST dataset, we also process another two datasets of fashion products images, the Apparel Classification with Style (ACS) dataset created by Bossard et al. [2] and the Fashionwear (FW) dataset prepared by Subramanya et al. [3], in a similar way, and do the classification by using AlexNet[4] and VGG11 [5] with modified structures. We report the best test accuracy to be 83.24% by using modified VGG11 on FW dataset. By converting sample images into smaller grayscale images, the training time can be largely reduced while similar classification accuracy can still be achieved.

[1] H. Xiao, K. Rasul, and R. Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. 2017.

[2] C. Leistner C. Wengert T. Quack L. Bossard, M. Dantone and L. V. Gool. Apparel Classification with Style, pages 321–335. Springer Berlin Heidelberg, Berlin, Heidelberg, 2013.

[3] V. M. Pramod A. Subramanya, P. Srinivas and S. S. Shylaja. Classification of Fashionwear Using Deep Learning, pages 605–611. Springer Singapore, Singapore, 2018.

[4] I. Sutskever K. Alex and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012.

[5] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014.
