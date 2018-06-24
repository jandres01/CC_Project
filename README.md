# CC_Project
Security Project for CC

## Data
- Num_Phish_Emails = 1948   
- Size of Dataset = 10799

## Correlation Matrices for insights on Email dataset 
**Actual numbers of Whole Data**   - /CC_Project/Correlation\ Matrix/corr_wholeData.png <br/>
**Correlation of Regular Emails**  - /CC_Project/Correlation\ Matrix/Corr_Reg_Email.png <br/>
**Correlation of Whole Data**  - /CC_Project/Correlation\ Matrix/Corr Whole Data.png <br/>
**Correlation of Phishing Emails** -/CC_Project/Correlation\ Matrix/CCorr_Phish_Data <br/>
**Script** - /CC_Project/Correlation\ Matrix/CorrMatrix.py

#### Phishing Emails have strong Relationships to
1. domain_SenderID <br/>
2. Email_has_attachment

## Used FeedFoward Neural Networks (FFNN) to predict phishing emails
**Output Results** - /CC_Project/FFNN/FFNN_final_test_model_30perc.csv <br/>
**Trained Model** - /CC_Project/FFNN/FFNN_final_test_model_30perc.h5 <br/>
**Keras model** - /CC_Project/FFNN/keras_FFNN.py <br/>
**TensorFlow model** - /CC_Project/FFNN/tf_FFNN.py

#### FFNN Algorithm Best Testing Results Report
- /CC_Project/FFNN/FFNN_final_test_model_30perc.csv
- /CC_Project/FFNN/Count_final_test_model_30perc.png
- /CC_Project/FFNN/Norm_final_test_model_30perc.png
**81% success rate** with discovering a phishing email <br/>
**88% accuracy rate** for all predictions <br/>
**90% evaluation rate** on training <br/>

## Density-based spatial clustering of applications with noise (DBSCAN) 
**Script** - /CC_Project/DBSCAN/dbscan.py
**Results** - /CC_Project/DBSCAN/DBSCAN_Clusters.png
Discovered **899 clusters** in the dataset

