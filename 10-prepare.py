import glob

import json

import re

import MeCab

bodies = []
m = MeCab.Tagger('-Owakati')
for name in glob.glob('../contents/*')[:10000]:
  o = json.load( open(name))
  body = (re.sub(r'\s{1,}', ' ', o['bodies']))
  body = body.replace('。', '。\n')
  body = [m.parse(x).strip().split() for x in list(filter(lambda x:re.search(r'。$', x), body.split('\n'))) ]
  bodies.append( body )

open('bodies.json', 'w').write( json.dumps(bodies, indent=2, ensure_ascii=False) )
