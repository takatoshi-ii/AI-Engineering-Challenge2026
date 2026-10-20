"""Search the extracted text artifacts.  Usage: grepdocs.py PATTERN [pathfilter...]"""
import json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import config
from dax.util import nkey

pat = re.compile(sys.argv[1])
filters = sys.argv[2:]
man = json.load(open(os.path.join(config.ARTIFACTS, 'manifest.json'), encoding='utf-8'))
for d in man['documents']:
    if filters and not all(nkey(f) in nkey(d['rel_path']) for f in filters):
        continue
    tp = os.path.join(config.TEXT_DIR, f"{d['id']}.md")
    if not os.path.exists(tp):
        continue
    txt = open(tp, encoding='utf-8').read()
    hits = [(m.start(), m.group(0)) for m in pat.finditer(txt)]
    if not hits:
        continue
    print('=' * 90)
    print(d['rel_path'])
    for pos, _ in hits[:12]:
        s = max(0, pos - 200)
        print('  ...' + txt[s:pos + 300].replace('\n', '\n  ') + '...')
        print('  ' + '-' * 60)
