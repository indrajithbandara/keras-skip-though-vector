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

def vectorize():
  #os.system('../bin/fasttext skipgram -dim 256 -input ../data/flatten.txt -output ../data/model -minCount 1 ') 
  model = KeyedVectors.load_word2vec_format('../data/model.vec', binary=False)
  with open('../data/flatten.txt', 'r') as f:
    tosave = []
    for e, line in enumerate(f):
      line = line.strip()
      base = np.zeros((20, 256), dtype=np.float)
      for i,term in enumerate(line.split()[:20]):
        try:
          base[i,:] = model[term]
        except KeyError as e:
          continue
      #print(base)
      tosave.append( base )
  open('fastvec.pkl', 'wb').write( pickle.dumps(tosave) )

def make_triple():
  tosave = pickle.loads( open('fastvec.pkl', 'rb').read() )
  buff   = []
  for i in range(1, len(tosave)-2, 1):
    mid =   tosave[i]
    pre = tosave[i-1]
    nex = tosave[i+1]
    buff.append( ( mid, pre, nex ) )
    if i != 0 and i%3000 == 0:
      mids = np.array( [t[0] for t in buff] )
      pres = np.array( [t[1] for t in buff] )
      nexs = np.array( [t[2] for t in buff] )
      open('fastvec_data_%09d.pkl'%i, 'wb').write( pickle.dumps( (mids,pres,nexs) ) )
      buff = []


if __name__ == '__main__':
  if '--step1' in sys.argv:
    flatten()

  if '--step2' in sys.argv:
    vectorize()

  if '--step3' in sys.argv:
    make_triple()
