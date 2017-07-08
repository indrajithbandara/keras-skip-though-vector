import os
import sys
import glob
import re
import MeCab
import pickle
from gensim.models import KeyedVectors

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
  os.system('../bin/fasttext skipgram -dim 512 -input ../data/flatten.txt -output ../data/model -minCount 1 ') 
  buff = os.popen('../bin/fasttext print-sentence-vectors ../data/model.bin < ../data/flatten.txt').read()
  
  tosave = []
  for text in buff.split('\n'):
    terms = text.split()
    vecs  = terms[-512:]
    terms = terms[:-512]
    vecs  = [ float(v) for v in vecs ]
    tosave.append( (terms, vecs) )

  open('../data/terms_vecs.pkl', 'wb').write( pickle.dumps(tosave) )

def make_triple():
  model = KeyedVectors.load_word2vec_format('../data/model.vec', binary=False)
  terms_vecs = pickle.loads( open('../data/terms_vecs.pkl', 'rb').read() )   
  term_index = {}
  for (terms, vecs) in terms_vecs:
    #print( terms )
    for term in terms:
      if term_index.get(term) is None:
        term_index[term] = len(term_index)   
  
  print( max( term_index.values() ) )
  triples = []
  for i in range(1, len(terms_vecs)-2, 1):
      
    prevec  = terms_vecs[i-1][1]
    nextvec = terms_vecs[i+1][1]
    nowvec  = []
    for term in terms_vecs[i][0]:
      try:
        nowvec.append( model[term] )  
      except KeyError as e:
        print( e )
    # print( nowvec )
    triple  = (nowvec, prevec, nextvec)
    triples.append( triple )
    
    if i%3000 == 0: 
      open('../data/triples_%09d.pkl'%i, 'wb').write( pickle.dumps(triples) ) 
      triples = []


if __name__ == '__main__':
  if '--step1' in sys.argv:
    flatten()

  if '--step2' in sys.argv:
    vectorize()

  if '--step3' in sys.argv:
    make_triple()
