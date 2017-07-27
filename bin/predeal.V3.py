import os
import sys
import glob
import re
import MeCab
import pickle
from gensim.models import KeyedVectors
import numpy as np
def flatten():
  with open('../data/flatten.txt', 'w') as w:
    m = MeCab.Tagger('-Owakati')
    for name in glob.glob('../download/*'):
      print(name)
      text = re.sub(r'\n{1,}', '\n', open(name).read() ) 
      for sen in m.parse( text ).strip().split('。'):
        senw = m.parse( sen ).strip()
        w.write( '%s\n'%senw ) 

from sklearn.cluster import KMeans
def to_one1():
  #os.system('../bin/fasttext skipgram -dim 256 -input ../data/flatten.txt -output ../data/model -minCount 1 ') 
  model = KeyedVectors.load_word2vec_format('../data/model.vec', binary=False)
  terms = set()
  with open('../data/flatten.txt', 'r') as f:
    for line in f:
      line = line.strip()
      for term in line.split():
        terms.add(term)
      if len(terms) > 50000:
        break
 
  vs = []
  for term in terms:
    try:
      v = np.array( model[term] )
    except KeyError as e:
      continue
    if v.shape == (256,):
      print("ok")
      vs.append( v )

  vs = np.array( vs )
  kmeans = KMeans(n_clusters=128, random_state=0)
  print('now fitting...')
  kmeans.fit(vs)
  print('fit was finished')

  open('../data/kmeans.pkl', 'wb').write( pickle.dumps(kmeans) )

def to_one2():
  kmeans = pickle.loads( open('../data/kmeans.pkl', 'rb').read() )
  model  = KeyedVectors.load_word2vec_format('../data/model.vec', binary=False)
  terms  = set()
  print('create term sets')
  with open('../data/flatten.txt', 'r') as f:
    for line in f:
      line = line.strip()
      for term in line.split():
        terms.add(term)
  print('finish term sets')
  
  print('creating term -> onehot')
  term_cls = {}
  for term in terms:
    try:
      v = np.array( model[term] )
    except KeyError as e:
      continue
    if v.shape != (256,):
      continue

    cls = kmeans.predict( v ).tolist().pop()
    term_cls[term] = cls
    print( term, cls )
  open('../data/term_cls.pkl', 'wb').write( pickle.dumps(term_cls) )

def make_triple():
  term_cls = pickle.loads( open('../data/term_cls.pkl', 'rb').read() )
  with open('../data/flatten.txt', 'r') as f:
    buffs  = []
    for line in f:
      line = line.strip()
      onehots = []
      for term in line.split():
        try:
          cls = term_cls[term]
          onehots.append( cls )
        except KeyError as e:
          continue
      buffs.append( onehots )
  triples = []
  for i in range(1, len(buffs)-1):
      head = buffs[i-1]
      mid  = buffs[i]
      tail = buffs[i+1]
      triple = (head, mid, tail)
      triples.append( triple )
  print( len(triples) )
  open('../data/triples.pkl', 'wb').write( pickle.dumps(triples) )

def enough_dataset():
  triples = pickle.loads( open('../data/triples.pkl', 'rb').read() )

  enoughs = []
  for e, triple in enumerate(triples):
    h,m,t = triple 
    if len(h) >= 50 or len(m) >= 50 or len(t) >= 50:
      continue
    hs = [ [0]*129 for i in range(50) ]
    for i in range(50):
      if i < len(h) - 1:
        hs[i][ h[i] ] = 1
      else:
        hs[i][ 128 ]  = 1
    ms = [ [0]*129 for i in range(50) ]
    for i in range(50):
      if i < len(m) - 1:
        ms[i][ m[i] ] = 1
      else:
        ms[i][ 128 ]  = 1
    ts = [ [0]*129 for i in range(50) ]
    for i in range(50):
      if i < len(t) - 1:
        ts[i][ t[i] ] = 1
      else:
        ts[i][ 128 ]  = 1
    enough = (hs, ms, ts)
    enoughs.append( enough )
    print( e, len(h), len(m), len(t) )

    if e != 0 and e%5000 == 0:
      open('../data/enoughs_%09d.pkl'%e, 'wb').write( pickle.dumps(enoughs) )
      enoughs = []

def make_batch():
  for e, name in enumerate( glob.glob('../data/enoughs_*.pkl') ):
    print( name )
    triples = pickle.loads( open(name, 'rb').read() )
    hs = list( map(lambda x:x[0], triples) )
    ms = list( map(lambda x:x[1], triples) )
    ts = list( map(lambda x:x[2], triples) )
    hs = np.array(hs)
    ms = np.array(ms)
    ts = np.array(ts)
    #print(hs.shape, ms.shape, ts.shape)
    open('../data/triple_%09d.pkl'%e, 'wb').write( pickle.dumps( (hs, ms, ts) ) )

if __name__ == '__main__':
  if '--step1' in sys.argv:
    flatten()

  if '--step2' in sys.argv:
    to_one1()

  if '--step3' in sys.argv:
    to_one2()
  
  if '--step4' in sys.argv:
    make_triple()

  if '--step5' in sys.argv:
    enough_dataset()

  if '--step6' in sys.argv:
    make_batch()
