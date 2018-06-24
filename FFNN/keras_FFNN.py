#Keras Feed Forward Neural Network for predicting Phish Emails

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import cv2
import os

def read_data():
    file = '../Phish_Dataset.csv'
    #drop id & temporarily tld  <---
    df = pd.read_csv(file)#.drop(['id','senders_email_tld'], axis=1)
    df = df.replace({'target_phishYN':{'Y':1, 'N':0}})
    #fix domain names
    #send = df[['senders_email_tld']].applymap(lambda x: x.split('.')[0])
    return df

def splitData(data):
    #Split Data testing & training
    perc = np.random.rand(len(data)) < 0.9
    training = data#[perc]
    testing = data[~perc]
    gid = testing.groupby('id')
    id_test = []
    for group in gid.groups:
      id_test.append(group)
    output = pd.DataFrame(columns=["id"])
    output["id"] = id_test
    #Split data x and y
    x_train = training.loc[:, data.columns != 'target_phishYN'].drop(['id','senders_email_tld'], axis=1)
    y_train = training[['target_phishYN']]
    x_test = testing.loc[:, data.columns != 'target_phishYN'].drop(['id','senders_email_tld'], axis=1)
    y_test = testing[['target_phishYN']]
    return testing,training,x_train,y_train,x_test,y_test,output

def main():
    #read Data
    df = read_data()
    test_data,train_data,x_train,y_train,x_test,y_test,output = splitData(df)

    # define the architecture of the network
    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_dim=16))
    model.add(Dropout(0.2))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    #Train Model
    print("[INFO] compiling model...")
    model.fit(x_train, y_train, epochs=50, batch_size=128, verbose=1)

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(x_test, y_test,
	batch_size=128, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
        accuracy * 100))

    print("Model Predictions")
    pred = model.predict(x_test,batch_size=128)
    
    #Analyze output & put into csv
    output = y_test    
    output['prediction'] = pred
    y =y_test['target_phishYN'].values    

    #add rounded colum
    ct_file = 0
    ct_phish = 0
    phish_total = 0
    round = pred
    for i in range(len(round)):
      if pred[i][0] <=0.3:
          round[i][0] = 0
      else:
          round[i][0] = 1
      if round[i][0] == y[i]:
          ct_file += 1
          if round[i][0] == 1:
              ct_phish += 1
 	      print i
      if y[i] == 1:
          phish_total += 1

    output["rounded"] = round

    #accuracy
    acc_file = float(ct_file)/len(y) 
    acc_phish = float(ct_phish)/phish_total
    output["acc_phish"] = acc_phish
    output["acc_file"] = acc_file
    
    output.to_csv('FFNN_whole_pred_3.csv')
    cm_pred = pred.flatten()

    #Display Confusion Matrix
    df_confusion = pd.crosstab(y, cm_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Blues):
      plt.matshow(df_confusion, cmap=cmap)
      plt.title(title)
      plt.colorbar()
      tick_marks = np.arange(len(df_confusion.columns))
      plt.xticks(tick_marks, df_confusion.columns, rotation=45)
      plt.yticks(tick_marks, df_confusion.index)
      #plt.tight_layout()
      plt.ylabel(df_confusion.index.name)
      plt.xlabel(df_confusion.columns.name)
      plt.subplots(figsize=(9, 7))
     

    fig = plot_confusion_matrix(df_confusion)
    norm_fig = plot_confusion_matrix(df_conf_norm)   
    plt.show()

    # dump the network architecture and weights to file
    print("[INFO] dumping architecture and weights to file...")
    model.save("FFNN_whole_model_3.h5")

if __name__ == "__main__":
    main()
