import json
import pickle

term_vector = pickle.loads(open('./embedding/term_vec.pkl', 'rb').read())

bodies = json.load(open('bodies.json'))

vector = []
for it, body in enumerate(bodies):
  bases = []
  for terms in body:
    print(it, terms)
    base = [ [0.0]*256 ]*100
    for index, term in enumerate(terms[:100]):
      if term_vector.get(term):
        base[index] = term_vector[term]
      
    bases.append( base )
  vector.append( bases )

open('vector.pkl', 'wb').write( pickle.dumps( vector ) )
    

