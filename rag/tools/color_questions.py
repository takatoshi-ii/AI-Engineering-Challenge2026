"""Which colours actually occur in the files the colour questions point at."""
import collections, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import config
from dax.util import color_info, nkey

TARGETS = [
    ('Q2  オレンジ行', ['青嶺', 'スケジュール_r2.xlsx']),
    ('Q7  黄色', ['青潮', '基礎分析.pptx']),
    ('Q15 黄色', ['東都', '03.データ', 'train.xlsx']),
    ('Q16 黄色+赤字', ['青葉与信', '報告資料']),
    ('Q17 黄色+RED', ['青葉与信', '会議録']),
    ('Q25 青色', ['白峰', '03.データ', 'train.xlsx']),
    ('Q33 青の折れ線', ['青潮', '基礎分析.docx']),
    ('Q42/77 黄色', ['ひがし丘', '03.データ', 'train.xlsx']),
    ('Q47 黄色', ['青嶺', '03.データ', 'train.xlsx']),
    ('Q65 黄色', ['白峰', '03.データ', 'train.xlsx']),
    ('Q80 黄色', ['東都', '03.データ', 'train.xlsx']),
    ('Q82 オレンジ行', ['ひがし丘', 'スケジュール.xlsx']),
    ('Q97 黄色', ['青葉バイオ', 'train.xlsx']),
]

man = json.load(open(os.path.join(config.ARTIFACTS, 'manifest.json'), encoding='utf-8'))

for label, frags in TARGETS:
    hits = [d for d in man['documents']
            if all(nkey(f) in nkey(d['rel_path']) for f in frags)]
    if not hits:
        print(f'{label:20s} -> ファイル未特定 {frags}')
        continue
    d = hits[0]
    t = open(os.path.join(config.TEXT_DIR, d['id'] + '.md'), encoding='utf-8').read()
    cols = collections.Counter(re.findall(r'\(([0-9A-F]{6})\)', t))
    # keep only colours that are not plain background/neutral
    interesting = [(c, n) for c, n in cols.most_common()
                   if color_info(c)[0] not in ('white', 'gray', 'black')]
    desc = ', '.join(f'{c}={color_info(c)[0]}/{color_info(c)[1]}({n})'
                     for c, n in interesting[:6]) or '(有彩色なし)'
    print(f'{label:20s} {d["name"][:26]:28s} {desc}')
