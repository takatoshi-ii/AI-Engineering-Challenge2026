"""Transcribe the same images with several models and lay the results side by side.

Speed is easy to measure; quality is not.  This runs a small, deliberately
varied sample (a scanned page, a chart, a diagram, a notebook figure) through
each candidate model and writes one page showing the picture next to every
model's reading of it, so the trade-off can be judged rather than guessed.

  python rag/tools/ocr_compare.py --models qwen2.5vl:7b,qwen3-vl:8b --n 4
"""
import argparse, html, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import config
import llm as llm_mod
import vision

PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>OCR モデル比較</title>
<style>
 body {{ font-family: "Segoe UI","Yu Gothic UI",sans-serif; margin:0; background:#f6f7f9; }}
 header {{ background:#fff; border-bottom:1px solid #d8dbe0; padding:14px 22px; }}
 h1 {{ margin:0; font-size:17px; }}
 .sum {{ margin-top:6px; font-size:13px; color:#555; }}
 .sum b {{ color:#111; }}
 .item {{ background:#fff; margin:16px 22px; border:1px solid #dfe2e7; border-radius:8px; }}
 .item h2 {{ font-size:13px; margin:0; padding:9px 14px; background:#eef1f5;
             border-bottom:1px solid #dfe2e7; word-break:break-all; }}
 .grid {{ display:grid; grid-template-columns: 300px repeat({ncol}, 1fr); }}
 .grid > div {{ padding:12px; border-right:1px solid #eceef1; }}
 .grid > div:last-child {{ border-right:none; }}
 .pic img {{ max-width:100%; border:1px solid #dde; border-radius:4px; cursor:zoom-in; }}
 .mdl {{ font-size:12px; font-weight:600; color:#334; margin-bottom:6px;
         padding-bottom:5px; border-bottom:1px solid #eceef1; }}
 .mdl span {{ font-weight:400; color:#778; }}
 pre {{ margin:0; white-space:pre-wrap; word-wrap:break-word; font-size:12px;
        line-height:1.6; font-family:Consolas,monospace; }}
 @media (prefers-color-scheme: dark) {{
   body {{ background:#16181c; color:#e6e6e6; }}
   header,.item {{ background:#1e2126; border-color:#333840; }}
   .item h2 {{ background:#262a31; border-color:#333840; }}
   .grid > div {{ border-color:#2b3038; }}
   .mdl {{ color:#cdd3dc; border-color:#2b3038; }}
 }}
</style></head><body>
<header><h1>OCR モデル比較</h1><div class="sum">{summary}</div></header>
{items}
<script>document.querySelectorAll('.pic img').forEach(i=>i.onclick=()=>window.open(i.src,'_blank'));</script>
</body></html>"""


def pick_sample(entries, n):
    """A spread across the kinds of image the corpus contains."""
    out, seen = [], set()
    for origin in ('pdf_page', 'file', 'notebook_output', 'embedded_media'):
        for e in entries:
            if e['origin'] != origin or e['image_id'] in seen:
                continue
            out.append(e)
            seen.add(e['image_id'])
            break
    for e in entries:
        if len(out) >= n:
            break
        if e['image_id'] not in seen and e['origin'] != 'slide_render':
            out.append(e)
            seen.add(e['image_id'])
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='qwen2.5vl:7b')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--out', default=os.path.join(config.ARTIFACTS, 'ocr_compare.html'))
    a = ap.parse_args()

    models = [m.strip() for m in a.models.split(',') if m.strip()]
    entries = json.load(open(os.path.join(config.ARTIFACTS, 'images.json'), encoding='utf-8'))
    entries = [e for e in entries if e['origin'] != 'slide_render']
    sample = pick_sample(entries, a.n)
    print(f'{len(sample)} 枚 x {len(models)} モデル')

    results, timing = {}, {m: [] for m in models}
    for m in models:
        client = llm_mod.Client(model=m, use_cache=False)
        for e in sample:
            t0 = time.time()
            try:
                _, txt = vision.transcribe_one(client, e)
            except Exception as exc:
                txt = f'[失敗] {type(exc).__name__}: {exc}'
            dt = time.time() - t0
            timing[m].append(dt)
            results[(m, e['image_id'])] = (txt, dt)
            print(f'  {m:18s} {dt:6.1f}s  {len(txt):5d}文字  {e["locator"]}')

    summary = ' / '.join(
        f'<b>{html.escape(m)}</b>: 平均 {sum(timing[m]) / max(len(timing[m]), 1):.1f}s，'
        f'253枚換算 {sum(timing[m]) / max(len(timing[m]), 1) * 253 / 60:.0f}分'
        for m in models)

    items = []
    for e in sample:
        src = os.path.relpath(e['path'], config.ARTIFACTS).replace('\\', '/')
        cells = [f'<div class="pic"><img loading="lazy" src="{html.escape(src)}"></div>']
        for m in models:
            txt, dt = results[(m, e['image_id'])]
            cells.append(
                f'<div><div class="mdl">{html.escape(m)} '
                f'<span>{dt:.1f}s / {len(txt)}文字</span></div>'
                f'<pre>{html.escape(txt)}</pre></div>')
        items.append(
            f'<div class="item"><h2>{html.escape(e["rel_path"])} '
            f'— {html.escape(e["origin"])} {html.escape(str(e["locator"]))}</h2>'
            f'<div class="grid">{"".join(cells)}</div></div>')

    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(PAGE.format(ncol=len(models), summary=summary, items='\n'.join(items)))
    print(f'\n-> {a.out}')


if __name__ == '__main__':
    main()
