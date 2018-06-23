# CC_Project
Security Project for CC

## Used Correlation Matrices to gain insights on Phishing Emails 
Visual - /CC_Project/Correlation\ Matrix/corr_wholeData.png
Script - /CC_Project/Correlation\ Matrix/CorrMatrix.py

#### Phishing Emails have strong Relationships to
1. domain_SenderID
2. Email_has_attachment

## Used FeedFoward Neural Networks (FFNN) to predict phishing emails
Output Report - /CC_Project/FFNN/FFNN_final_pred.csv
Trained Model - /CC_Project/FFNN/FFNN_model.h5
Keras model - /CC_Project/FFNN/keras_FFNN.py
TensorFlow model - /CC_Project/FFNN/tf_FFNN.py

#### FFNN algorithm used for non-linear patterns in the dataset
57% success rate with predicting a phishing email
87% accuracy rate for all predictions
90% evaluation rate on training

## Density-based spatial clustering of applications with noise (DBSCAN) 
## Used to classify the dataset

