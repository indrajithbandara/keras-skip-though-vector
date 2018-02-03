import json

term_index = {}

obj = json.load(open('bodies.json'))

for contents in obj:
  for terms in contents:
    for term in terms:
      if term_index.get(term) is None:
        term_index[term] = len(term_index)
json.dump(term_index, fp=open('term_index.json', 'w'), ensure_ascii=False, indent=2)
