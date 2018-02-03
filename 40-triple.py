import json
import pickle
vector = pickle.load(open('vector.pkl', 'rb'))

triples = []
for body in vector:
  for i in range(1,len(body)-1):
    prev = body[i-1]
    nxt  = body[i+1]
    
    inpt = body[i][::-1]

    triple = (inpt, prev, nxt)
    triples.append( triple )

open('triples.pkl', 'wb').write(pickle.dumps(triples))
