"""Scores a validation run with the competition's own CRAG rubric.

Mirrors evaluation/readme.md: Perfect=1, Acceptable=0.5, Missing=0,
Incorrect=-1, including the numeric / enumeration / extraction-condition rules.

Run: python rag/score.py [--questions valid]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
import run as run_mod

RUBRIC = """あなたは質問応答の採点者です。予測回答を模範解答と比較し、次の基準で判定してください。

- Perfect(1): 質問に対して正確に答えており、虚偽の内容が含まれていない。
- Acceptable(0.5): 有用な答えを提供しているが、有用性を損なわない軽微な誤りがある。
- Missing(0): 「わかりません」「確認できませんでした」など、具体的な答えがない。
- Incorrect(-1): 間違っている、または無関係。

判定規則:
【数値問題】
- 模範解答に数値がある場合のみ、数値の一致を評価する。完全一致のみ Perfect。
- Acceptable は、所定の桁数で四捨五入した結果が一致する場合に限る。
- 単位の有無や接尾辞の違いは、意味が変わらなければ同一とみなす（例:「5」と「5ページ」）。
- 範囲は、対応関係が一致していれば区切り記号・括弧・接続表現の違いだけで Incorrect にしない。
- 模範解答に数値がない場合、予測中の数値の正しさを推測してはならない。
【要素列挙問題】
- 模範解答のすべての要素と予測のすべての要素が一致した場合のみ Perfect。
- 順序が正解条件でなければ並び順は問わない。
- 要素の不足または追加があれば Incorrect。Acceptable は使わない。
【抽出条件・集計内容】
- 文章全体の表現ではなく、(1)すべての抽出条件 (2)集計対象の列・項目 (3)集計方法(最大/最小/平均/合計/個数) で比較する。
- 必須要素がすべて含まれ意味が一致すれば、記載順や文形式が違っても Perfect。
- 模範解答に記載された要素だけを必須要素とし、記載のない条件を推測して足さない。

JSON のみを出力: {"label": "Perfect|Acceptable|Missing|Incorrect", "reason": "簡潔な理由"}"""

POINTS = {'Perfect': 1.0, 'Acceptable': 0.5, 'Missing': 0.0, 'Incorrect': -1.0}


def judge(client: llm_mod.Client, q: str, gt: str, pred: str) -> dict:
    msg = [{'role': 'user', 'content':
            f'質問:\n{q}\n\n模範解答:\n{gt}\n\n予測回答:\n{pred}'}]
    resp = client.chat(msg, system=RUBRIC, max_tokens=600)
    txt = (resp.get('text') or '').strip()
    start, end = txt.find('{'), txt.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(txt[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {'label': 'Incorrect', 'reason': f'判定不能: {txt[:150]}'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', default='valid')
    ap.add_argument('--model')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--tag', required=True, help='run.py の --tag と同じ名前')
    a = ap.parse_args()

    questions = run_mod.load_questions(a.questions)
    import re as _re
    slug = _re.sub(r'[^A-Za-z0-9._-]+', '_', a.tag)
    store_path = os.path.join(config.ANSWER_DIR, f'{a.questions}__{slug}.json')
    print(f'採点対象: {os.path.basename(store_path)}')
    store = json.load(open(store_path, encoding='utf-8'))
    client = llm_mod.Client(model=a.model)

    rows = [q for q in questions if q.get('answer')]
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(judge, client, q['question'], q['answer'],
                          (store.get(str(q['index'])) or {}).get('answer', '')): q
                for q in rows}
        for fut in as_completed(futs):
            q = futs[fut]
            results[q['index']] = fut.result()

    total = 0.0
    counts: dict[str, int] = {}
    print(f'\n{"idx":>4} {"label":<11} answer / expected')
    print('-' * 110)
    for q in rows:
        r = results[q['index']]
        lab = r.get('label', 'Incorrect')
        counts[lab] = counts.get(lab, 0) + 1
        total += POINTS.get(lab, -1.0)
        pred = (store.get(str(q['index'])) or {}).get('answer', '')
        mark = {'Perfect': 'OK ', 'Acceptable': 'ACC', 'Missing': '-- ', 'Incorrect': 'XX '}[lab]
        print(f'{q["index"]:>4} {mark} {lab:<11} {pred[:60]}')
        if lab != 'Perfect':
            print(f'{"":>4}     expected: {q["answer"][:70]}')
            print(f'{"":>4}     reason  : {r.get("reason", "")[:100]}')
    n = len(rows)
    print('-' * 110)
    print(f'score = {total / n:.4f}  ({total:+.1f} / {n} questions)')
    print('  ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))

    out = os.path.join(config.LOG_DIR, f'score_{a.questions}__{slug}.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'score': total / n, 'counts': counts,
                   'detail': {str(k): v for k, v in results.items()}},
                  fh, ensure_ascii=False, indent=1)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
