from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, dot, multiply, concatenate, add, Activation
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.normalization import BatchNormalization as BN
from keras.layers.core     import Dropout 
from keras.layers.embeddings import Embedding
from keras.models import Model
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import json
import glob
import copy
import os
import re
import time
import concurrent.futures
import threading 

inputs      = Input( shape=(100,256) ) 
encoded     = Bi( LSTM(200, return_sequences=False) )( inputs )
#encoded     = BN()(encoded)
encoded     = Dense(1012, activation='relu')( encoded )
encoded     = Dense(512, activation='relu')( encoded )
encoder     = Model(inputs, encoded)

decoded_1   = Bi( LSTM(200, dropout=0.0, recurrent_dropout=0.0, return_sequences=True) )( RepeatVector(100)( encoded ) )
decoded_1   = TD( Dense(1024, activation='relu') )( decoded_1 )
decoded_1   = TD( Dense(256, activation='linear') )( decoded_1 )

decoded_2   = Bi( LSTM(200, dropout=0.0, recurrent_dropout=0.0, return_sequences=True) )( RepeatVector(100)( encoded ) )
decoded_2   = TD( Dense(1024, activation='relu') )( decoded_1 )
decoded_2   = TD( Dense(256, activation='linear') )( decoded_1 )

skipthought = Model( inputs, [decoded_1, decoded_2] )
skipthought.compile( optimizer=RMSprop(), loss='mae' )
  
buff = None
now  = time.strftime("%H_%M_%S")
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
  # with open('../logs/loss_%s.log'%now, 'a+') as f:
  #   f.write('%s\n'%str(buff))
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )

def train():
  triples = pickle.load( open('triples.pkl','rb') )
  Xs, ys1, ys2 = [], [], []
  for x, y1, y2 in triples[:30000]:
    Xs.append(x)
    ys1.append(y1)
    ys2.append(y2)
  Xs, ys1, ys2 = map(np.array, [Xs, ys1, ys2])
  
  for count in range(100):
    skipthought.fit( Xs, [ys1, ys2], \
                          epochs=5,\
                          batch_size=128,
                          validation_split=0.02, \
                          callbacks=[batch_callback] )
    skipthought.save_weights('models/%09d.h5'%count)

def predict():
  to_load = sorted( glob.glob('../models/*.h5') ).pop() 
  skipthought.load_weights( to_load )
  t = threading.Thread(target=loader, args=())
  t.start()
  while True:
    if DATASET_POOL == []:
      print('no buffers so delay some seconds')
      time.sleep(1.)
      continue

    x, y1, y2, name = DATASET_POOL.pop(0)
    
    vecs = encoder.predict( x )
    for v in vecs.tolist():
      print( v )

  
  
if __name__ == '__main__':
  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
