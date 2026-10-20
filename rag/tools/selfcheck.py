"""Pipeline health check that needs no model access.

Verifies that every artifact stage exists and that the evidence types the
questions depend on (formatting, colours, formulas, pivots, filters, charts,
pagination, decryption, images) are actually present in the corpus.
"""
import json, os, re, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import config
import agent as agent_mod
from dax.util import strip_fmt

FAIL = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAIL.append(name)


print('=== artifacts ===')
man_path = os.path.join(config.ARTIFACTS, 'manifest.json')
check('manifest.json', os.path.exists(man_path))
man = json.load(open(man_path, encoding='utf-8'))
docs = man['documents']
check('403 files extracted', len(docs) >= 400, f'{len(docs)} docs')
check('no extraction errors', not man['errors'], f"{len(man['errors'])} errors")
check('10 projects resolved', len(man['projects']) >= 10, f"{len(man['projects'])} projects")
check('glossary terms parsed', len(man['glossary_terms']) > 100,
      f"{len(man['glossary_terms'])} terms")

n_text = sum(1 for d in docs if os.path.exists(os.path.join(config.TEXT_DIR, d['id'] + '.md')))
check('text renderings', n_text == len(docs), f'{n_text}/{len(docs)}')

img_path = os.path.join(config.ARTIFACTS, 'images.json')
check('images.json', os.path.exists(img_path))
imgs = json.load(open(img_path, encoding='utf-8')) if os.path.exists(img_path) else []
by_origin = collections.Counter(e['origin'] for e in imgs)
check('image registry', len(imgs) > 500, dict(by_origin))
check('image files on disk',
      all(os.path.exists(e['path']) for e in imgs[:50]), 'sampled 50')

check('bm25 index', os.path.exists(os.path.join(config.INDEX_DIR, 'bm25.pkl')))

print('\n=== evidence coverage ===')
corpus = {}
for d in docs:
    p = os.path.join(config.TEXT_DIR, d['id'] + '.md')
    if os.path.exists(p):
        corpus[d['rel_path']] = open(p, encoding='utf-8').read()
allt = '\n'.join(corpus.values())

check('bold spans', '[[bold]]' in allt, f"{allt.count('[[bold]]')} runs")
check('yellow highlights', 'hl:yellow' in allt, f"{allt.count('hl:yellow')} runs")
check('red text', 'color:red' in allt, f"{allt.count('color:red')} runs")
check('underline+italic', 'underline' in allt and 'italic' in allt)
check('word comments', 'コメントが付いている箇所' in allt)
check('excel fills', '塗りつぶし色の使用状況' in allt)
check('excel formulas', '数式セル一覧' in allt)
check('autofilter', 'フィルター条件' in allt)
check('pivot tables', 'ピボットテーブル' in allt)
check('charts with values', 'データ: (1,' in allt or 'データ: (' in allt)
check('extended charts (histogram)', '拡張グラフ' in allt)
check('layout pagination', 'ページ割り（レイアウト実測）' in allt,
      f"{sum(1 for d in docs if d.get('n_pages'))} docs")
check('decrypted files', sum(1 for d in docs if d.get('encrypted')) == 2,
      [d['name'] for d in docs if d.get('encrypted')])
check('notebook outputs', '[出力:' in allt)
check('csv profiles', '列プロファイル' in allt)
check('pdf formatting spans', '書式が付いている箇所の一覧（PDF）' in allt)
check('conditional formatting with colours',
      '条件付き書式（この条件に合致' in allt and re.search(r'条件=cellIs.*→ 塗り=', allt) is not None)
check('highlight+font-colour combinations', 'hl:yellow,color:red' in allt)
triple = [l for l in allt.split('\n')
          if l.startswith('- p.') and all(k in l.split(']')[0] for k in ('bold', 'italic', 'underline'))]
check('pdf bold+italic+underline detected', bool(triple), triple[:2])

# Tool checks assert only structural properties (a lookup resolves, a search
# returns hits, a page number is produced).  No expected answer or data value is
# written here — this file is a diagnostic and is not part of answer generation.
print('\n=== tools ===')
tb = agent_mod.Toolbox()

abbrev = next((p['abbrev'] for p in man['projects'] if p.get('abbrev')), None)
some_term = next(iter(man['glossary_terms']))
r = tb.lookup_glossary(some_term)
check('glossary lookup', some_term in r, r.strip()[:50])
r = tb.list_files(project=abbrev)
check('project routing', '件' in r and 'プロジェクト' in r, r.split('\n')[0])
r = tb.search_corpus(query='契約 金額 支払', top_k=3)
check('search returns hits', r.count('---') >= 2)

contract = next((d for d in docs if d['ext'] == '.docx' and '契約' in d['rel_path']), None)
r = tb.read_document(rel_path=contract['rel_path'])
check('read_document', len(r) > 500)
r_plain = tb.read_document(rel_path=contract['rel_path'], plain=True)
check('plain mode strips tags', '[[' not in r_plain and len(r_plain) > 500)

paged = next(d for d in docs if d.get('n_pages'))
rec = json.load(open(os.path.join(config.DOCS_DIR, paged['id'] + '.json'), encoding='utf-8'))
last_page = rec['page_text'][-1]
probe = ''.join(strip_fmt(last_page['text'] or '').split())[:14]
r = tb.locate_in_pages(rel_path=paged['rel_path'], text=probe)
found = re.findall(r'p\.(\d+)', r.split('---')[0])
check('page locator resolves a unique page',
      found and found[0] == str(last_page['page']) and len(set(found)) == 1,
      f"probe={probe!r} -> {found[:4]} (expected {last_page['page']})")

withimg = next(e for e in imgs if e['origin'] == 'embedded_media')
r = tb.list_images(rel_path=withimg['rel_path'])
check('image listing', 'image_id=' in r)
r = tb.view_image(image_id=withimg['image_id'])
check('image loading', isinstance(r, dict) and 'image' in r)

r = tb.run_python(code='''
paths = find(".csv")
print("csv files found:", len(paths) > 0)
df = pd.read_csv(paths[0], encoding="utf-8-sig", low_memory=False)
print("shape ok:", df.shape[0] > 0 and df.shape[1] > 0)
''')
check('python sandbox', 'csv files found: True' in r and 'shape ok: True' in r, r.strip()[:60])

r = tb.run_python(code='''
n = sum(1 for rel, t in each_text("01.契約") if len(t) > 200)
print("contract docs readable:", n)
''')
check('cross-project sandbox helpers',
      'contract docs readable:' in r and '0' != r.strip().split(':')[-1].strip(),
      r.strip()[:45])

print('\n' + ('ALL CHECKS PASSED' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}'))
sys.exit(1 if FAIL else 0)
