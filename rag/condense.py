"""Reduce each stored answer to just the value the question asked for.

The evaluator judges an enumeration answer Incorrect if it carries elements the
expected answer does not, and a wordy reply is exactly that: the value plus
several sentences of justification.  The agent needs its reasoning while it
works, so the trimming happens here, after the fact, as a separate pass over the
stored answers.

Nothing question-specific is applied — one fixed instruction runs over every
answer, and the source is the agent's own text.

Run: python rag/condense.py --tag final [--questions test]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import config
import llm as llm_mod
import run as run_mod

SYSTEM = """あなたは、すでに作成された回答文から「質問が求めた答えそのもの」だけを取り出す編集者です。

出力の規則:
- 質問が尋ねている値・名称・IDだけを出力する。説明、理由、根拠、出典、前置き、言い換えは一切書かない。
- 元の回答に含まれる数値・固有名詞・IDは、表記を一字も変えずにそのまま使う。丸めたり単位を足し引きしたりしない。
- 質問が単位や桁数を指定している場合は、その形式を保つ（例:「43日」「5,775,000円」「0.50534」）。
- 複数の要素を答える問題は、要素だけを「、」で区切って並べる。番号、箇条書き記号、補足は付けない。
- 括弧書きの補足は削る。ただし答えそのものが括弧を含む場合（例:「対象外（契約明記）」）は残す。
- 元の回答が「確認できませんでした」等で答えを持たない場合は、「資料からは確認できませんでした。」とだけ出力する。
- 元の回答に無い情報を足さない。判断に迷う場合は、元の回答から答えに当たる部分を最短で抜き出す。

答えだけを出力し、それ以外は何も書かないこと。"""

FALLBACK = run_mod.FALLBACK


def condense_one(client: llm_mod.Client, question: str, answer: str) -> str:
    if not answer.strip():
        return FALLBACK
    msg = [{'role': 'user', 'content':
            f'質問:\n{question}\n\n作成された回答:\n{answer}\n\n答えだけを出力してください。'}]
    out = (client.chat(msg, system=SYSTEM, max_tokens=400).get('text') or '').strip()
    # a condenser that returns nothing, or balloons the text, is not trusted
    if not out or len(out) > max(len(answer), 200):
        return answer
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--questions', default='test')
    ap.add_argument('--model')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out-tag', help='書き出し先タグ（既定は <tag>_short）')
    a = ap.parse_args()

    questions = {q['index']: q['question'] for q in run_mod.load_questions(a.questions)}
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', a.tag)
    src = os.path.join(config.ANSWER_DIR, f'{a.questions}__{slug}.json')
    store = json.load(open(src, encoding='utf-8'))

    out_slug = re.sub(r'[^A-Za-z0-9._-]+', '_', a.out_tag or f'{a.tag}_short')
    dst = os.path.join(config.ANSWER_DIR, f'{a.questions}__{out_slug}.json')
    result = json.load(open(dst, encoding='utf-8')) if os.path.exists(dst) else {}

    client = llm_mod.Client(model=a.model)
    todo = [k for k in store if k not in result]
    print(f'{len(todo)} 件を圧縮 (model={client.model})')

    def work(k):
        return k, condense_one(client, questions[int(k)], store[k]['answer'])

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, k): k for k in todo}
        for fut in as_completed(futs):
            k, short = fut.result()
            result[k] = {**store[k], 'answer': short, 'answer_full': store[k]['answer']}
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f'  [{done}/{len(todo)}]')
                json.dump(result, open(dst, 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=1)
    json.dump(result, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    before = sum(len(v['answer_full']) for v in result.values()) / max(len(result), 1)
    after = sum(len(v['answer']) for v in result.values()) / max(len(result), 1)
    print(f'\n平均文字数: {before:.0f} -> {after:.0f}')
    print(f'-> {dst}')


if __name__ == '__main__':
    main()
