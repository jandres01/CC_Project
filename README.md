# CC_Project
Security Project for CC

## Used Correlation Matrices to gain insights on Phishing Emails 
Visual - /CC_Project/Correlation\ Matrix/corr_wholeData.png <br/>
Script - /CC_Project/Correlation\ Matrix/CorrMatrix.py

#### Phishing Emails have strong Relationships to
1. domain_SenderID <br/>
2. Email_has_attachment

## Used FeedFoward Neural Networks (FFNN) to predict phishing emails
Output Report - /CC_Project/FFNN/FFNN_final_pred.csv <br/>
Trained Model - /CC_Project/FFNN/FFNN_model.h5 <br/>
Keras model - /CC_Project/FFNN/keras_FFNN.py <br/>
TensorFlow model - /CC_Project/FFNN/tf_FFNN.py

#### FFNN algorithm used for non-linear patterns in the dataset
57% success rate with predicting a phishing email <br/>
87% accuracy rate for all predictions <br/>
90% evaluation rate on training <br/>

## Density-based spatial clustering of applications with noise (DBSCAN) 
#### Used to classify the dataset

