#Neural Network to predict phishing Emails
#TensorFlow Model
#Written by Robbie Andres

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from sqlite3 import Error
from sklearn.preprocessing import MinMaxScaler

#Read data & convert Y-1, N-0
def read_data():
    file = 'Phish_Dataset.csv'
    #drop id & temporarily tld  <---
    df = pd.read_csv(file)#.drop(['id','senders_email_tld'], axis=1)
    df = df.replace({'target_phishYN':{'Y':1, 'N':0}})
    #fix domain names
    #send = df[['senders_email_tld']].applymap(lambda x: x.split('.')[0])
    return df

def calcLossNN():
    #First shuffle data
    batch_size = 246
    for j in range(0, epochs):
      shuffle_indices = np.random.permutation(np.arange(len(y_train)))
      X_train = x_train.values[shuffle_indices]
      Y_train = y_train.values[shuffle_indices]
      #predict loss by grabbing batches
      #epoch is finished cycle of grabbing all batches
      for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = Y_train[start:start + batch_size]
        train_data = {X:batch_x,Y_:batch_y}
        _,y = sess.run([train_step,Y],feed_dict=train_data)
      c1 =sess.run([loss],feed_dict={X: x_train, Y_: y_train})
      c2 =sess.run([loss],feed_dict={X: x_test, Y_: y_test})
      print "Epoch: ",j,"\t Train Loss: ", c1, "\t Test Loss: ",c2

def splitData(data):
    #Split Data testing & training
    #perc = np.random.rand(len(data)) < 0.9
    training = data #[perc]
    testing = data #[~perc]
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

def calcNN_Pred(testData,xTest):
    calcLossNN()
    #Create predictions with unshuffled test data
    y = sess.run(Y, feed_dict={X:xTest.values})
    return y

def binary_crossentropy(y_true, y_pred):
    result = []
    for i in range(len(y_pred)):
      y_pred[i] = [max(min(x, 1 - K.epsilon()), K.epsilon()) for x in y_pred[i]]
      result.append(-np.mean([y_true[i][j] * math.log(y_pred[i][j]) + (1 - y_true[i][j]) * math.log(1 - y_pred[i][j]) for j in range(len(y_pred[i]))]))
    return np.mean(result)

if __name__ == "__main__":
  df = read_data()
#  df = df[['id','target_phishYN','domain_SenderID','email_has_attachment','num_attachment_zip','num_attachment_othertype','num_recipients','num_attachment_msoffice','num_attachment_txt_csv']]
  print("hi")
  test_data,train_data,x_train,y_train,x_test,y_test,output = splitData(df)

  epochs = 75
  n_passed_values = 16
  n_neurons_1 = 128 #512
  n_neurons_2 = 64 #256
  n_neurons_3 = 32 #128
  n_target = 1

  X = tf.placeholder(tf.float32,[None,n_passed_values],name="X")
  Y_ = tf.placeholder(tf.float32,[None,1],name="Ylabel")

  weight_initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True, factor=1.0, seed=None)
  bias_initializer = tf.zeros_initializer()

  #Variables for Hidden layers: [Input,Output]
  W_hidden_1 = tf.Variable(weight_initializer([n_passed_values, n_neurons_1]))
  bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

  W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
  bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

  W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
  bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

  # Output layer: Variables for output weights and biases
  W_out = tf.Variable(weight_initializer([n_neurons_3, n_target]))
  bias_out = tf.Variable(bias_initializer([n_target]))

  # Hidden layer
  #test Relu - rectified linear unit - helps decrease vanishing gradient
#  hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
#  hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
#  hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

  # Hidden layer
  #test Relu - rectified linear unit - helps decrease vanishing gradient
  hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
  hidden_1 = tf.nn.dropout(hidden_1, keep_prob=0.8)
  hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
  hidden_2 = tf.nn.dropout(hidden_2, keep_prob=0.85)
  hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
  hidden_3 = tf.nn.dropout(hidden_3, keep_prob=0.9)

  #feedforward network
  Y = tf.nn.sigmoid(tf.add(tf.matmul(hidden_3, W_out), bias_out)) 

  loss = K.binary_crossentropy(Y,Y_)
  train_step = tf.train.RMSPropOptimizer(0.001).minimize(loss)

  #feedforward network
  #Y = tf.add(tf.matmul(hidden_3, W_out), bias_out)
  #loss = tf.reduce_mean(tf.squared_difference(Y,Y_))
  #train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  sess = tf.Session()
  sess.run(init)

  output["nn_pred"] = calcNN_Pred(test_data,x_test)
  add_data = test_data[['id','target_phishYN']]
  output = output.set_index('id').join(add_data.set_index('id'))

  output.to_csv("FFNN_whole_results.csv")
  # Add ops to save and restore all the variables.

  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)

