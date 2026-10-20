"""Build a side-by-side review page for the image transcriptions.

The OCR stage is the one place where a silent quality problem would be hard to
notice — the text looks plausible whether or not it matches the picture.  This
writes an HTML page pairing every image with what was read out of it, so the
result can be checked by eye.

  python rag/tools/ocr_review.py                      # everything transcribed
  python rag/tools/ocr_review.py --filter 白峰        # only matching paths
  python rag/tools/ocr_review.py --origin pdf_page    # only scanned pages
  python rag/tools/ocr_review.py --dump               # plain text instead
"""
import argparse, html, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import config
from dax.util import nkey

PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>OCR 品質レビュー</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font-family: "Segoe UI", "Yu Gothic UI", sans-serif; margin: 0;
        background: #f6f7f9; color: #1a1a1a; }}
 header {{ position: sticky; top: 0; background: #fff; border-bottom: 1px solid #d8dbe0;
          padding: 14px 22px; z-index: 10; }}
 h1 {{ margin: 0 0 4px; font-size: 17px; }}
 .meta {{ color: #666; font-size: 13px; }}
 .filter {{ margin-top: 10px; }}
 .filter input {{ width: 320px; padding: 6px 9px; border: 1px solid #c8ccd2;
                 border-radius: 5px; font-size: 13px; }}
 .item {{ background: #fff; margin: 16px 22px; border: 1px solid #dfe2e7;
         border-radius: 8px; overflow: hidden; }}
 .item h2 {{ font-size: 13px; margin: 0; padding: 9px 14px; background: #eef1f5;
            border-bottom: 1px solid #dfe2e7; font-weight: 600; word-break: break-all; }}
 .item h2 span {{ color: #667; font-weight: 400; }}
 .body {{ display: grid; grid-template-columns: minmax(280px, 42%) 1fr; }}
 .body > div {{ padding: 14px; }}
 .pic {{ border-right: 1px solid #eceef1; background: #fafbfc; text-align: center; }}
 .pic img {{ max-width: 100%; border: 1px solid #dde; border-radius: 4px; cursor: zoom-in; }}
 pre {{ margin: 0; white-space: pre-wrap; word-wrap: break-word; font-size: 12.5px;
       line-height: 1.65; font-family: "Consolas", monospace; }}
 @media (max-width: 900px) {{ .body {{ grid-template-columns: 1fr; }}
                             .pic {{ border-right: none; }} }}
 @media (prefers-color-scheme: dark) {{
   body {{ background: #16181c; color: #e6e6e6; }}
   header, .item {{ background: #1e2126; border-color: #333840; }}
   .item h2 {{ background: #262a31; border-color: #333840; }}
   .pic {{ background: #202329; border-color: #333840; }}
   .filter input {{ background: #262a31; color: #e6e6e6; border-color: #3a404a; }}
 }}
</style></head><body>
<header>
  <h1>OCR 品質レビュー</h1>
  <div class="meta">{n} 件 / 書き起こし済み {done} 件・未処理 {todo} 件</div>
  <div class="filter"><input id="q" placeholder="パスや本文で絞り込み（入力すると即時フィルタ）"></div>
</header>
{items}
<script>
const q = document.getElementById('q');
q.addEventListener('input', () => {{
  const v = q.value.toLowerCase();
  document.querySelectorAll('.item').forEach(el => {{
    el.style.display = el.innerText.toLowerCase().includes(v) ? '' : 'none';
  }});
}});
document.querySelectorAll('.pic img').forEach(img => {{
  img.addEventListener('click', () => window.open(img.src, '_blank'));
}});
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--filter', default='', help='パスの部分一致で絞り込む')
    ap.add_argument('--origin', default='', help='pdf_page / file / notebook_output / embedded_media')
    ap.add_argument('--dump', action='store_true', help='HTMLでなくテキストで出力')
    ap.add_argument('--out', default=os.path.join(config.ARTIFACTS, 'ocr_review.html'))
    a = ap.parse_args()

    entries = json.load(open(os.path.join(config.ARTIFACTS, 'images.json'), encoding='utf-8'))
    vpath = os.path.join(config.ARTIFACTS, 'vision.json')
    notes = json.load(open(vpath, encoding='utf-8')) if os.path.exists(vpath) else {}

    rows = [e for e in entries if e['origin'] != 'slide_render']
    if a.filter:
        rows = [e for e in rows if nkey(a.filter) in nkey(e['rel_path'])]
    if a.origin:
        rows = [e for e in rows if e['origin'] == a.origin]
    rows.sort(key=lambda e: (e['rel_path'], e.get('page') or 0, e['locator']))

    done = sum(1 for e in rows if notes.get(e['image_id']))
    if a.dump:
        for e in rows:
            print('=' * 100)
            print(f"{e['rel_path']}  [{e['origin']} {e['locator']}]")
            print('=' * 100)
            print(notes.get(e['image_id']) or '(未処理)')
            print()
        return

    items = []
    for e in rows:
        note = notes.get(e['image_id'])
        # the page lives in artifacts/, so images/ resolves relative to it
        src = os.path.relpath(e['path'], config.ARTIFACTS).replace('\\', '/')
        items.append(
            '<div class="item">'
            f'<h2>{html.escape(e["rel_path"])} '
            f'<span>— {html.escape(e["origin"])} {html.escape(str(e["locator"]))}</span></h2>'
            '<div class="body">'
            f'<div class="pic"><img loading="lazy" src="{html.escape(src)}"></div>'
            f'<div><pre>{html.escape(note) if note else "（未処理）"}</pre></div>'
            '</div></div>')

    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(PAGE.format(n=len(rows), done=done, todo=len(rows) - done,
                             items='\n'.join(items)))
    print(f'{len(rows)} 件（書き起こし済み {done}）-> {a.out}')
    print('ブラウザで開いてください:')
    print(f'  start "" "{a.out}"')


if __name__ == '__main__':
    main()
