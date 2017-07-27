from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model, Sequential
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

WIDTH       = 256
ACTIVATOR   = 'tanh'
DO          = Dropout(0.1)
inputs      = Input( shape=(20, WIDTH) ) 
encoded     = Bi( LSTM(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )(inputs)
encoded     = TD( Dense(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR) )( encoded )
encoded     = Flatten()( encoded )
encoded     = Dense(512, kernel_initializer='lecun_uniform', activation='linear')( encoded )
encoder     = Model(inputs, encoded)

decoded_1   = Bi( LSTM(512, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )( RepeatVector(20)( encoded ) )
decoded_1   = TD( Dense(256) )( decoded_1 )

decoded_2   = Bi( LSTM(511, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )( RepeatVector(20)( encoded ) )
decoded_2   = TD( Dense(256) )( decoded_1 )

skipthought = Model( inputs, [decoded_1, decoded_2] )
skipthought.compile( optimizer=Adam(), loss='mean_squared_logarithmic_error' )
  
buff = None
now  = time.strftime("%H_%M_%S")
def callback(epoch, logs):
  global buff
  buff = copy.copy(logs)
  with open('../logs/loss_%s.log'%now, 'a+') as f:
    f.write('%s\n'%str(buff))
batch_callback = LambdaCallback(on_epoch_end=lambda batch,logs: callback(batch,logs) )

DATASET_POOL = []
def loader():
  while True:
    for name in glob.glob('../bin/fastvec_data_*.pkl'):
      while True:
        if len( DATASET_POOL ) >= 1: 
          time.sleep(1.0)
        else:
          break
      print('loading data...', name)
      x, y1, y2 = pickle.loads( open(name, 'rb').read() ) 
      print( x.shape, y1.shape, y2.shape )
      DATASET_POOL.append( (x, y1, y2, name) )
      print('finish recover from sparse...', name)

class E:
  l = [12, 10, 5, 3]
  cn = 0
  @staticmethod
  def G():
    E.cn += 1
    return E.l[E.cn%len(E.l)]

def train():
  try:
    to_load = sorted( glob.glob('../models/*.h5') ).pop() 
    skipthought.load_weights( to_load )
    count = int( re.search(r'(\d{1,})', to_load).group(1) )
  except IndexError as e:
    print(e)
  except Exception as e:
    print(e)
    sys.exit()
  t = threading.Thread(target=loader, args=())
  t.start()
  count = 0
  while True:
    if DATASET_POOL == []:
      print('no buffers so delay some seconds')
      time.sleep(1.)
    else:
      x, y1, y2, name = DATASET_POOL.pop(0)
      print('will deal this data', name)
      print('now count is', count)
      inner_loop = 0
      skipthought.fit( x, [y1, y2], \
                            epochs=E.G(),\
                            validation_split=0.02, \
                            callbacks=[batch_callback] )
      count += 1
      if count%1 == 0:
         skipthought.save_weights('../models/%09d.h5'%count)

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
