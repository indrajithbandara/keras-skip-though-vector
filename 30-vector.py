import json

term_index = json.load(open('term_index.json'))
term_index['<EOS>'] = len(term_index)

bodies = json.load(open('bodies.json'))

vector = []
for body in bodies:
  bases = []
  for terms in body:
    base = [ term_index['<EOS>'] ]*100
    for index, term in enumerate(terms[:100]):
      base[index] = term_index[ term ]
    bases.append( base )
  vector.append( bases )

json.dump( vector, fp=open('vector.json', 'w'), ensure_ascii=False, indent=2)
    

