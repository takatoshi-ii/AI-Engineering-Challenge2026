"""Estimate token consumption from the pipeline's real payload sizes."""
import json, os, sys, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import tiktoken
import config
import agent as agent_mod

enc = tiktoken.get_encoding('o200k_base')
T = lambda s: len(enc.encode(s))

print('=== 固定オーバーヘッド（毎回のLLM呼び出しで送るもの）===')
sys_t = T(agent_mod.SYSTEM)
tools_t = T(json.dumps(agent_mod.TOOLS, ensure_ascii=False))
print(f'  システムプロンプト : {sys_t:>7,} tokens')
print(f'  ツール定義         : {tools_t:>7,} tokens')
fixed = sys_t + tools_t
print(f'  小計               : {fixed:>7,} tokens / 呼び出し')

print('\n=== ツール出力の実サイズ ===')
tb = agent_mod.Toolbox()
man = json.load(open(os.path.join(config.ARTIFACTS, 'manifest.json'), encoding='utf-8'))

samples = {}
samples['list_files(案件指定)'] = tb.list_files(
    project=next(p['abbrev'] for p in man['projects'] if p.get('abbrev')))
samples['search_corpus'] = tb.search_corpus(query='契約 金額 支払 条件', top_k=12)
samples['lookup_glossary(全件)'] = tb.lookup_glossary()

doc_sizes = []
for d in man['documents']:
    p = os.path.join(config.TEXT_DIR, d['id'] + '.md')
    if os.path.exists(p):
        doc_sizes.append(T(open(p, encoding='utf-8').read()))
doc_sizes.sort()

for k, v in samples.items():
    print(f'  {k:24s} {T(v):>7,} tokens')
print(f'  read_document(既定18000字上限) 最大 {T("あ" * 18000):>7,} tokens 相当')
print(f'\n  文書1件あたりの抽出テキスト: 中央値 {statistics.median(doc_sizes):>7,.0f} / '
      f'平均 {statistics.mean(doc_sizes):>7,.0f} / 最大 {doc_sizes[-1]:>7,} tokens')
big = [s for s in doc_sizes if s > 20000]
print(f'  20k tokens 超の文書: {len(big)} 件 / {len(doc_sizes)} 件')

print('\n=== 画像（書き起こし対象）===')
imgs = json.load(open(os.path.join(config.ARTIFACTS, 'images.json'), encoding='utf-8'))
auto = [e for e in imgs if e['origin'] in ('pdf_page', 'notebook_output', 'file', 'embedded_media')]
from PIL import Image
dims = []
for e in auto[:40]:
    try:
        with Image.open(e['path']) as im:
            dims.append(im.size)
    except Exception:
        pass


def openai_image_tokens(w, h):
    """OpenAI high-detail tiling: 85 base + 170 per 512px tile after fitting."""
    scale = min(2048 / max(w, h), 1.0)
    w, h = w * scale, h * scale
    scale = 768 / min(w, h) if min(w, h) > 768 else 1.0
    w, h = w * scale, h * scale
    tiles = (int(w + 511) // 512) * (int(h + 511) // 512)
    return 85 + 170 * tiles


img_t = [openai_image_tokens(w, h) for w, h in dims]
avg_img = statistics.mean(img_t) if img_t else 1500
print(f'  対象枚数           : {len(auto)}')
print(f'  1枚あたり入力      : 約 {avg_img:,.0f} tokens (サンプル{len(dims)}枚の実寸から)')
vis_in = len(auto) * (avg_img + T(__import__("vision").PROMPT))
vis_out = len(auto) * 1200
print(f'  書き起こし合計     : 入力 約 {vis_in:>10,.0f} / 出力 約 {vis_out:>9,} tokens')

print('\n=== 回答生成（100問）===')


import agent as A


def estimate(k_calls, avg_result, fixed_t, pruned=True):
    """The transcript is resent every round, so results accumulate.

    Without pruning that is quadratic in the number of steps; with pruning only
    the most recent KEEP_FULL_RESULTS stay full-size, so it is ~linear.
    """
    total = 0
    for i in range(1, k_calls + 1):
        if pruned:
            full = min(i, A.KEEP_FULL_RESULTS)
            stubs = max(0, i - A.KEEP_FULL_RESULTS)
            carried = full * avg_result + stubs * (A.STUB_CHARS // 2)
        else:
            carried = i * avg_result
        total += fixed_t + carried
    return total


for label, k, r in (('楽観 (10回・軽い出力)', 10, 1500),
                    ('標準 (18回・中程度)', 18, 3000),
                    ('悲観 (26回・重い出力)', 26, 6000)):
    raw = estimate(k, r, fixed, pruned=False)
    cut = estimate(k, r, fixed, pruned=True)
    print(f'  {label:24s} 履歴削減なし 100問 {raw * 100:>12,.0f} → '
          f'あり {cut * 100:>11,.0f} 入力tokens')
print(f'  ※ 出力は1問あたり概ね {18 * 300:,}〜{26 * 600:,} tokens')
print(f'  ※ 履歴削減: 直近{A.KEEP_FULL_RESULTS}件は全文、それ以前は{A.STUB_CHARS}文字に圧縮')
