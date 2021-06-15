# saeClassify
# Supervised cell type identification of single-cell transcriptomes with deep sparse autoencoder and multiclass classifier   
Mingguang Shi*, Yachen Tang, Zhou Sheng

This folder contains the program and the data we used for supervised cell type recognition of single-cell transcriptome using deep sparse autoencoders and multiple classifiers.


"initializeParameters.m"  A function that randomly initializes parameters based on the size of the layer.

"sparseAutoencoderCost.m"  A function we use to construct a sparse autoencoder model.

"pancreas.m"	The main program for supervised cell type recognition of single-cell transcriptome data contains the main methods we used and the multi-classification SVM model we constructed.

"pancreas.train.cbind.mat" The data and labels we collected for training the model included 1,802 samples and 16,669 features.

"pancreas.test.cbind.mat" The data and labels we collected for testing the model included 1,802 samples and 16,669 features.

See our code for more details. We have detailed comments on the code or contact us at mingguang.shi@hfut.edu.cn

Note: This code is built to run on a CPU.
