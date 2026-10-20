"""Second opinion on an answer that has already been produced.

Scoring pays -1 for a wrong answer and 0 for a stated non-answer, so an answer
worth keeping has to be right more often than not.  This pass hands a fresh
agent the question and the proposed answer — with none of the reasoning that led
to it — and asks it to check the claim against the drive.  When the check comes
back negative the answer is withdrawn rather than submitted, turning a -1 into a
0.

Nothing here is question-specific: one prompt runs over every stored answer, and
the verdict comes from the same tools the answering agent had.

  python rag/verify.py --tag v2 --questions valid
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import config
import agent as agent_mod
import llm as llm_mod
import run as run_mod

FALLBACK = run_mod.FALLBACK

SYSTEM = """あなたは、データ分析会社（データアステル社）の共有ドライブを調査して、
すでに作成された回答が正しいかどうかを検証する監査役です。

与えられるのは「質問」と「提出予定の回答」だけです。その回答がどうやって導かれたかは分かりません。
あなた自身がツールで資料を読み、回答の正しさを独立に確かめてください。

## 判定の基準
- 支持: 回答の内容が資料から確認でき、質問が問うているものに答えている。
- 否定: 資料と矛盾する、質問が問うている対象を取り違えている、または資料上まったく裏付けが取れない。
- 保留: 十分に調べたが、正しいとも誤りとも判断できない。

## 重要な原則
- **迷ったら「保留」にすること。** 「否定」は、資料と食い違うことをあなた自身が確認できた場合にのみ使う。
  裏付けを見つけられなかっただけで「否定」にしてはならない。探し方が悪かった可能性のほうが高い。
- 回答が「資料からは確認できませんでした」「該当なし」の場合は、それが妥当かを見る。
  資料に明確な答えが存在するのに諦めている場合のみ「否定」とする。
- 表現の違い（単位の有無、語順、言い換え、括弧書きの有無）は「否定」の理由にならない。
  内容が合っていれば「支持」とする。
- 数値・名称・IDは、資料上の値と一字一句突き合わせること。桁や単位の取り違えは「否定」である。

## 進め方
1. 質問に出てくる案件名・略称は lookup_glossary で引く。
2. list_files / search_corpus / read_document で根拠となる箇所を自分で見つける。
3. 計算が要るものは run_python で自分で計算し直し、提示された値と一致するか確かめる。
4. 図・グラフ・座席表など画像に基づくものは view_image で自分の目で確認する。

最後は必ず submit_verdict ツールで判定を提出してください。"""

VERDICT_TOOL = {
    'name': 'submit_verdict',
    'description': '検証結果を提出する。必ず最後にこれを呼ぶこと。',
    'parameters': {
        'type': 'object',
        'properties': {
            'verdict': {'type': 'string', 'enum': ['支持', '否定', '保留'],
                        'description': '提示された回答に対する判定'},
            'reason': {'type': 'string',
                       'description': '資料のどこを見てそう判断したか'},
        },
        'required': ['verdict', 'reason'],
    },
}


def verify_one(question: str, answer: str, tb: agent_mod.Toolbox,
               client: llm_mod.Client, max_steps: int = 12) -> dict:
    """Run the audit loop for a single stored answer."""
    tools = [t for t in agent_mod.TOOLS if t['name'] != 'submit_answer'] + [VERDICT_TOOL]
    messages = [{'role': 'user', 'content':
                 f'質問:\n{question}\n\n提出予定の回答:\n{answer}\n\n'
                 'この回答が正しいか、資料を自分で確認して判定してください。'}]
    n_calls = 0
    for step in range(max_steps):
        if max_steps - step <= 2:
            messages.append({'role': 'user', 'content':
                             '残り実行回数がわずかです。現時点の確認結果で '
                             'submit_verdict を呼んでください。'
                             '確信が持てない場合は「保留」としてください。'})
        messages = agent_mod._prune(messages)
        active = [VERDICT_TOOL] if max_steps - step <= 1 else tools
        resp = client.chat(messages, tools=active, system=SYSTEM, max_tokens=2048)
        calls = resp.get('tool_calls') or []
        if not calls:
            messages.append({'role': 'user', 'content':
                             'ツールで確認を続けるか、submit_verdict で判定してください。'})
            continue
        messages.append({'role': 'assistant', 'content': resp.get('text') or '',
                         'tool_calls': calls})
        for tc in calls:
            name, args = tc['name'], tc['arguments']
            if name == 'submit_verdict':
                return {'verdict': args.get('verdict') or '保留',
                        'reason': args.get('reason') or '', 'calls': n_calls}
            n_calls += 1
            fn = getattr(tb, name, None)
            if fn is None:
                result = f'未知のツール: {name}'
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = f'ツール実行エラー: {type(e).__name__}: {e}'
            if isinstance(result, dict) and 'image' in result:
                messages.append({'role': 'tool', 'tool_call_id': tc['id'],
                                 'content': f"画像を表示します: {result['caption']}"})
                messages.append({'role': 'user', 'content': [
                    {'type': 'text', 'text': f"[画像] {result['caption']}"},
                    result['image']]})
            else:
                messages.append({'role': 'tool', 'tool_call_id': tc['id'],
                                 'content': str(result)[:agent_mod.MAX_TOOL_RESULT]})
    return {'verdict': '保留', 'reason': '(判定に至らず)', 'calls': n_calls}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--questions', default='valid')
    ap.add_argument('--model')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out-tag', help='書き出し先タグ（既定は <tag>_ver）')
    ap.add_argument('--only', help='検証する index をカンマ区切りで指定（省略時は全問）')
    # The whole system has to reproduce inside 12 hours; answering already takes
    # six, so the audit gets a step budget rather than free rein.
    ap.add_argument('--max-steps', type=int, default=12)
    a = ap.parse_args()

    questions = {q['index']: q['question'] for q in run_mod.load_questions(a.questions)}
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', a.tag)
    src = os.path.join(config.ANSWER_DIR, f'{a.questions}__{slug}.json')
    store = json.load(open(src, encoding='utf-8'))

    out_slug = re.sub(r'[^A-Za-z0-9._-]+', '_', a.out_tag or f'{a.tag}_ver')
    dst = os.path.join(config.ANSWER_DIR, f'{a.questions}__{out_slug}.json')
    result = json.load(open(dst, encoding='utf-8')) if os.path.exists(dst) else {}

    client = llm_mod.Client(model=a.model)
    tb = agent_mod.Toolbox(vision_capable=run_mod.probe_vision(client))

    keys = [str(i) for i in sorted(questions)]
    if a.only:
        want = {x.strip() for x in a.only.split(',')}
        keys = [k for k in keys if k in want]
    todo = [k for k in keys if k not in result and k in store]
    print(f'{len(todo)} 件を検証 (model={client.model})')

    def work(k):
        t0 = time.time()
        try:
            v = verify_one(questions[int(k)], store[k].get('answer', ''), tb, client,
                           max_steps=a.max_steps)
        except Exception as e:
            v = {'verdict': '保留', 'reason': f'[検証エラー] {type(e).__name__}: {e}',
                 'calls': 0}
        v['seconds'] = round(time.time() - t0, 1)
        return k, v

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, k): k for k in todo}
        for fut in as_completed(futs):
            k, v = fut.result()
            keep = v['verdict'] != '否定'
            result[k] = {**store[k],
                         'answer': store[k]['answer'] if keep else FALLBACK,
                         'answer_before': store[k]['answer'],
                         'verdict': v['verdict'], 'verify_reason': v['reason'],
                         'verify_seconds': v['seconds']}
            done += 1
            mark = '維持' if keep else '棄権'
            print(f'  [{done}/{len(todo)}] Q{k} {v["verdict"]} → {mark} '
                  f'({v["seconds"]}s)  {store[k]["answer"][:44]}')
            json.dump(result, open(dst, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)

    json.dump(result, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n_no = sum(1 for v in result.values() if v.get('verdict') == '否定')
    print(f'\n否定={n_no} / {len(result)}  -> {dst}')


if __name__ == '__main__':
    main()
