from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
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
WIDTH       = 512
ACTIVATOR   = 'selu'
DO          = Dropout(0.1)
inputs      = Input( shape=(20, WIDTH) ) 
encoded     = Bi( GRU(256, kernel_initializer='lecun_uniform', activation=ACTIVATOR, return_sequences=True) )(inputs)
encoded     = TD( Dense(256, kernel_initializer='lecun_uniform', activation=ACTIVATOR) )( encoded )
encoded     = Flatten()( encoded )
encoded     = Dense(256, kernel_initializer='lecun_uniform', activation='sigmoid')( encoded )
encoder     = Model(inputs, encoded)

decoded_1   = Dense(1024, kernel_initializer='lecun_uniform', activation=ACTIVATOR)( encoded )
decoded_1   = Dense(512, activation='linear')( decoded_1 )
decoded_2   = Dense(1024, kernel_initializer='lecun_uniform', activation=ACTIVATOR)( encoded )
decoded_2   = Dense(512, activation='linear')( decoded_2 )

skipthought = Model( inputs, [decoded_1, decoded_2] )
skipthought.compile( optimizer=Adam(), loss='mse' )

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
    for name in sorted( glob.glob('../data/triples_*.pkl') ):
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

def train():
  t = threading.Thread(target=loader, args=())
  t.start()
  count = 0
  try:
    to_load = sorted( glob.glob('models/*.h5') ).pop() 
    in2de.load_weights( to_load )
    count = int( re.search(r'(\d{1,})', to_load).group(1) )
  except Exception as e:
    print( e )
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
                            epochs=1,\
                            validation_split=0.02, \
                            callbacks=[batch_callback] )
      count += 1

if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
