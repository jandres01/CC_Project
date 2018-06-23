#Keras Feed Forward Neural Network for predicting Phish Emails
#Best result so far is 90% accuracy rate

from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import pandas as pd
import numpy as np
import argparse
import cv2
import os

def read_data():
    file = 'Phish_Dataset.csv'
    #drop id & temporarily tld  <---
    df = pd.read_csv(file)#.drop(['id','senders_email_tld'], axis=1)
    df = df.replace({'target_phishYN':{'Y':1, 'N':0}})
    #fix domain names
    #send = df[['senders_email_tld']].applymap(lambda x: x.split('.')[0])
    return df

def splitData(data):
    #Split Data testing & training
    perc = np.random.rand(len(data)) < 0.9
    training = data[perc]
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
 
    # dump the network architecture and weights to file
    print("[INFO] dumping architecture and weights to file...")
    model.save(args["model_whole_train"])

if __name__ == "__main__":
    main()
