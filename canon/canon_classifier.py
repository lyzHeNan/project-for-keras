# coding:utf-8
# lyz
# 10.11.2017
# implement a classifier for pictures taken by canon

# import package
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input,Dense,Activation
from keras.optimizers import Adam,RMSprop
from sklearn.model_selection import train_test_split

import sys,os
import time
import numpy as np
#

def load_dataset(file_dir):
  dataset = list()
  labels = list()
  with open(file_dir,'r') as f:
    for line in f.readlines():
      line = line.strip().split('\t')
      dataset.append(line[:-1])
      labels.append(line[-1])
  return dataset,labels
  
#
def NN_Canon(dataset,labels):
  x_train,x_test,y_train,y_test = train_test_split(dataset,labels,test_size=0.25,random_state=30)
  print(len(x_train),len(y_train),len(x_test),len(y_test))
  y_train = np_utils.to_categorical(y_train,4)[:,1:]
  y_test = np_utils.to_categorical(y_test,4)[:,1:]
  x_train = np.array(x_train).reshape(204,6)
  y_train = np.array(y_train).reshape(204,3)
  # 
  input_dataset = Input(shape=(6,))
  layer_01 = Dense(64,activation='relu',name='layer_01')(input_dataset)
  
  layer_02 = Dense(32,activation='relu',name='layer_02')(layer_01)
  
  predictions = Dense(3,activation='relu')(layer_02)
  
  #
  rmsprop = RMSprop(lr=0.001,rho=0.9,decay=0.0)
  # 
  model = Model(inputs=input_dataset,outputs=predictions)
  model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
  # fitting
  model.fit(x_train,y_train,epochs=10,batch_size=10)
  # 01 data & parameter
  dense_layer_01 = Model(inputs=input_dataset,outputs=model.get_layer('layer_01').output)
  data_01 = dense_layer_01.predict(x_train)
  print('layer_01 parameter:',dense_layer_01.get_weights())
  print('layer_01:',data_01)
  # 02 data & parameter
  dense_layer_02 = Model(inputs=input_dataset,outputs=model.get_layer('layer_02').output)
  data_02 = dense_layer_02.predict(x_train)
  print('layer_02 parameter:',dense_layer_02.get_weights())
  print('layer_02:',data_02)
  # testing
  loss, accuracy = model.evaluate(x_test,y_test)
  print('loss:',loss)
  print ('accuracy:',accuracy)
  
  
if __name__ == '__main__':
  print('this script is:',sys.argv[0])
  file_dir = sys.argv[1]
  print(file_dir)
  dataset,labels = load_dataset(file_dir)
  NN_Canon(dataset,labels)
