"""Machine OCR of every corpus image, using a vision model.

Scanned reports, chart PNGs and the floor map hold information that exists in no
text layer.  Without this stage that content is invisible to full-text search
entirely, so the pipeline runs OCR over the images and keeps the transcription
alongside the rest of the extracted text.

This is a mechanical extraction step, equivalent to the OCR/table/format
extraction the other stages perform, and it is part of the normal build:

* it never sees a question — every image is processed identically, in registry
  order, with one fixed prompt;
* it is regenerated from the raw drive by re-running this module (delete
  ``vision.json`` or the file is extended incrementally);
* the output is a transcription of what is visible, not a curated set of
  answers, references or "important pages", and nothing here is hand-edited.

``vision.json`` is a cache so a re-run does not pay for the same image twice;
it is not an input that must be prepared by hand for the pipeline to work.

Run: python rag/vision.py [--origins pdf_page,file,...] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows consoles are cp932 here; a model answer containing e.g. U+202F would
# otherwise abort the whole run inside a progress print.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


import config
import llm as llm_mod

DEFAULT_ORIGINS = ('pdf_page', 'notebook_output', 'file', 'embedded_media')

PROMPT = """この画像は、あるデータ分析会社の社内資料の一部です。後で検索・質問応答に使うため、**見えている情報をもれなく日本語で書き起こして**ください。

必ず含めること:
1. すべての文字（見出し、本文、表、ラベル、注記、ヘッダー/フッター、ページ番号）を、表記どおりに書き写す。数値・ID・単位・記号は改変しない。
2. 表は行と列の構造がわかる形で書き出す。
3. グラフ・図の場合:
   - グラフ種別、タイトル、凡例、軸ラベル
   - x軸・y軸に**実際に表示されている目盛りの値**をすべて列挙する（最小・最大を含む）
   - 各系列の色と、読み取れる範囲でのデータ点の値（棒の高さ、折れ線の各点、最大・最小の位置）
   - ヒストグラムならビンの範囲とおおよその度数
4. 強調表現（蛍光ペン/マーカー、色つき文字、太字、下線、囲み、吹き出し）があれば、**どの語句にどの色が付いているか**を明示する。
5. 図形の位置関係（左/右/上/下、隣接、対面）が意味を持つ図（座席表・配置図など）は、位置関係も説明する。

推測や補完はせず、見えるものだけを書く。読めない箇所は「判読不可」と書く。"""


def _targets(origins: set[str], limit: int | None):
    entries = json.load(open(os.path.join(config.ARTIFACTS, 'images.json'), encoding='utf-8'))
    out = [e for e in entries if e['origin'] in origins]
    return out[:limit] if limit else out


def transcribe_one(client: llm_mod.Client, entry: dict) -> tuple[str, str]:
    ctx = (f"参考情報: このファイルは「{entry['rel_path']}」の {entry['origin']} "
           f"({entry['locator']}) です。")
    msg = [{'role': 'user', 'content': [
        {'type': 'text', 'text': PROMPT + '\n\n' + ctx},
        llm_mod.image_part(entry['path']),
    ]}]
    resp = client.chat(msg, max_tokens=3000)
    return entry['image_id'], (resp.get('text') or '').strip()


def build(origins=DEFAULT_ORIGINS, limit=None, workers=6, model=None) -> None:
    config.ensure_dirs()
    out_path = os.path.join(config.ARTIFACTS, 'vision.json')
    notes: dict[str, str] = {}
    if os.path.exists(out_path):
        notes = json.load(open(out_path, encoding='utf-8'))

    targets = [e for e in _targets(set(origins), limit) if e['image_id'] not in notes]
    print(f'{len(targets)} images to transcribe ({len(notes)} cached)')
    if not targets:
        return
    client = llm_mod.Client(model=model)
    print(f'provider={client.provider} model={client.model}')

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(transcribe_one, client, e): e for e in targets}
        for fut in as_completed(futs):
            e = futs[fut]
            try:
                iid, text = fut.result()
                if text:
                    notes[iid] = text
            except Exception as exc:
                print(f'  ! {e["image_id"]}: {type(exc).__name__}: {exc}')
            done += 1
            if done % 10 == 0 or done == len(targets):
                print(f'  [{done}/{len(targets)}] {time.time() - t0:.0f}s')
                with open(out_path, 'w', encoding='utf-8') as fh:
                    json.dump(notes, fh, ensure_ascii=False, indent=1)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(notes, fh, ensure_ascii=False, indent=1)
    print(f'{len(notes)} notes -> {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--origins', default=','.join(DEFAULT_ORIGINS))
    ap.add_argument('--limit', type=int)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--model')
    a = ap.parse_args()
    build(origins=tuple(a.origins.split(',')), limit=a.limit,
          workers=a.workers, model=a.model)
