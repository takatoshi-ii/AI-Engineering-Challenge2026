"""Answers a question file and writes the submission.

  python rag/run.py --questions test          -> submit/predictions.csv (+ zip)
  python rag/run.py --questions valid --score -> validation run with scoring

Answers are written incrementally so an interrupted run resumes where it
stopped, and every run records the model, seed and per-question tool trace.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows consoles are cp932 here; a model answer containing e.g. U+202F would
# otherwise abort the whole run inside a progress print.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


import agent as agent_mod
import config
import llm as llm_mod

FALLBACK = '資料からは確認できませんでした。'


def load_questions(which: str) -> list[dict]:
    qdir = config.questions_dir()
    if not qdir:
        raise RuntimeError('question folder not found under share/')
    target = None
    for n in sorted(os.listdir(qdir)):
        if not n.lower().endswith('.csv'):
            continue
        if which in n.lower():
            target = os.path.join(qdir, n)
    if target is None:
        raise RuntimeError(f'no question csv matching "{which}" in {qdir}')
    rows = []
    with open(target, encoding='utf-8-sig', newline='') as fh:
        for r in csv.DictReader(fh):
            rows.append({'index': int(r['index']), 'question': r['question'],
                         'answer': r.get('answer')})
    print(f'{len(rows)} questions from {os.path.basename(target)}')
    return rows


def truncate_tokens(text: str, limit: int = config.MAX_ANSWER_TOKENS) -> str:
    """Keep answers under the evaluator's hard token cap."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding('o200k_base')
        toks = enc.encode(text)
        if len(toks) <= limit - 5:
            return text
        return enc.decode(toks[:limit - 5])
    except Exception:
        return text if len(text) < limit * 1.5 else text[:int(limit * 1.5)]


def probe_vision(client: llm_mod.Client) -> bool:
    """Can this model accept an image at all?

    Text-only models are perfectly usable here because the OCR stage has
    already transcribed every picture — but the agent must not hand them an
    image and stall.  One tiny probe settles it.
    """
    import base64
    # 1x1 PNG
    px = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==')
    try:
        client.chat([{'role': 'user', 'content': [
            {'type': 'text', 'text': '1'},
            {'type': 'image', 'mime': 'image/png',
             'data': base64.b64encode(px).decode()}]}], max_tokens=8)
        return True
    except Exception:
        return False


def run(which: str, model: str | None, workers: int, limit: int | None,
        force: bool, max_steps: int, tag: str | None = None) -> None:
    config.ensure_dirs()
    questions = load_questions(which)
    if limit:
        questions = questions[:limit]

    client = llm_mod.Client(model=model)
    vision_ok = probe_vision(client)
    tb = agent_mod.Toolbox(vision_capable=vision_ok)
    print(f'provider={client.provider} model={client.model} workers={workers} '
          f'vision={"yes" if vision_ok else "no (OCR結果で代替)"}')

    # One store per model: comparing configurations means several runs over the
    # same questions, and a shared file would let the slower run overwrite the
    # faster one's results.
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', tag or client.model)
    store_path = os.path.join(config.ANSWER_DIR, f'{which}__{slug}.json')
    print(f'保存先: {os.path.basename(store_path)}')

    # Progress goes to a file the run owns.  A separate watcher process has to be
    # started by hand, gets forgotten, and silently keeps pointing at the last
    # run's store — this cannot drift out of sync with the job it belongs to.
    progress_path = os.path.join(config.ARTIFACTS, f'progress_{which}__{slug}.log')
    store: dict[str, dict] = {}
    if os.path.exists(store_path) and not force:
        store = json.load(open(store_path, encoding='utf-8'))
        print(f'{len(store)} answers already stored (resuming)')

    # a question that crashed is not "answered" — always retry it on resume
    todo = [q for q in questions
            if str(q['index']) not in store or store[str(q['index'])].get('errored')]
    retry = [q['index'] for q in todo if str(q['index']) in store]
    print(f'{len(todo)} to answer' + (f'（うち再試行 {len(retry)} 件: {retry}）' if retry else ''))


    total_q = len(questions)
    header = (f'# {which} / model={client.model} / tag={slug} / '
              f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n'
              f'# 全{total_q}問, 未回答{len(todo)}問\n')
    with open(progress_path, 'w', encoding='utf-8') as fh:
        fh.write(header)

    def note(idx, res, done_n, todo_n):
        pct = len(store) * 100 // max(total_q, 1)
        line = (f'{time.strftime("%H:%M:%S")}  {len(store):>3}/{total_q} ({pct:>3}%)  '
                f'[{done_n}/{todo_n}] Q{idx} {res["elapsed"]:.0f}s '
                f'{res["n_tool_calls"]}呼'
                + ('  !ERROR' if res.get('errored') else '')
                + f'  {" ".join(res["answer"].split())[:120]}')
        with open(progress_path, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    def work(q: dict) -> tuple[int, dict]:
        log: list = []
        t0 = time.time()
        try:
            res = agent_mod.answer_question(q['question'], tb, client,
                                            max_steps=max_steps, log=log)
        except Exception as e:
            traceback.print_exc()
            # Distinguish a genuine "not in the documents" from a crash: both
            # print the same sentence, and conflating them hid a bug once.
            res = {'answer': FALLBACK, 'confidence': 'low', 'errored': True,
                   'evidence': f'[実行エラー] {type(e).__name__}: {e}'}
        res['question'] = q['question']
        res['elapsed'] = round(time.time() - t0, 1)
        res['n_tool_calls'] = len(log)
        res['trace'] = log
        return q['index'], res

    t0 = time.time()
    done = 0
    if workers <= 1:
        for q in todo:
            idx, res = work(q)
            store[str(idx)] = res
            done += 1
            print(f"[{done}/{len(todo)}] Q{idx} ({res['elapsed']}s, {res['n_tool_calls']} calls) "
                  f"{res['answer'][:70]}")
            note(idx, res, done, len(todo))
            with open(store_path, 'w', encoding='utf-8') as fh:
                json.dump(store, fh, ensure_ascii=False, indent=1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, q): q for q in todo}
            for fut in as_completed(futs):
                idx, res = fut.result()
                store[str(idx)] = res
                done += 1
                print(f"[{done}/{len(todo)}] Q{idx} ({res['elapsed']}s, "
                      f"{res['n_tool_calls']} calls) {res['answer'][:70]}")
                note(idx, res, done, len(todo))
                with open(store_path, 'w', encoding='utf-8') as fh:
                    json.dump(store, fh, ensure_ascii=False, indent=1)

    with open(store_path, 'w', encoding='utf-8') as fh:
        json.dump(store, fh, ensure_ascii=False, indent=1)
    errored = [k for k, v in store.items() if v.get('errored')]
    if errored:
        print(f'\n!! 実行エラーで回答できなかった設問: {len(errored)} 件 -> {sorted(errored, key=int)}')
        print('   （--force なしで再実行すれば、この設問だけやり直せます）')
    with open(progress_path, 'a', encoding='utf-8') as fh:
        fh.write(f'# 完了 {time.strftime("%H:%M:%S")}  {len(store)}/{total_q}問  '
                 f'エラー{len(errored)}件\n')
    print(f'進捗ログ: {progress_path}')
    print(f'\nanswered in {time.time() - t0:.0f}s')

    # ------------------------------------------------------------- write out
    # Only the real question set may touch the submission files.  A validation
    # run once overwrote a finished 100-question submission with its own 30
    # rows, and the truncated file was submitted before anyone noticed.
    if which != 'test':
        out_csv = os.path.join(config.ANSWER_DIR, f'{which}__{slug}_predictions.csv')
        with open(out_csv, 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            for q in questions:
                a2 = (store.get(str(q['index'])) or {}).get('answer') or FALLBACK
                w.writerow([q['index'], ' '.join(a2.split())])
        print(f'wrote {out_csv}  （提出用 submit/ は test 以外では更新しません）')
        _write_meta(client, which, questions, max_steps, workers, store, t0)
        return

    # A run that mostly failed must not replace a good submission.  One did:
    # a NameError made all 100 questions "complete" instantly with the fallback
    # text, the run finished normally, and it overwrote a scored submission.
    n_bad = sum(1 for v in store.values()
                if v.get('errored') or (v.get('answer') or '').strip() == FALLBACK)
    if n_bad > len(questions) * 0.2:
        print(f'\n!! 回答できなかった設問が {n_bad}/{len(questions)} 件あります。'
              '\n   提出物 (submit/) は上書きしません。原因を確認して再実行してください。'
              f'\n   回答は {store_path} に保存済みです。')
        _write_meta(client, which, questions, max_steps, workers, store, t0)
        return

    out_csv = os.path.join(config.SUBMIT_DIR, 'predictions.csv')
    with open(out_csv, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        for q in questions:
            a = (store.get(str(q['index'])) or {}).get('answer') or FALLBACK
            # Models emit exotic spaces (U+202F, U+00A0, ideographic space).
            # Fold them to ASCII space so the answer is one clean line and the
            # evaluator's tokeniser sees nothing unexpected.
            a = re.sub(r'[  -​  　]', ' ', a)
            a = ' '.join(a.split())            # single line, no stray newlines
            w.writerow([q['index'], truncate_tokens(a)])
    print(f'wrote {out_csv}')

    zpath = os.path.join(config.SUBMIT_DIR, 'submission.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, 'predictions.csv')
    print(f'wrote {zpath}')

    _write_meta(client, which, questions, max_steps, workers, store, t0)


def _write_meta(client, which, questions, max_steps, workers, store, t0):
    elapsed = time.time() - t0
    per_q = [v.get('elapsed', 0) for v in store.values()]
    meta = {
        'model': client.model, 'provider': client.provider, 'seed': config.SEED,
        'temperature': client.temperature, 'questions': which,
        'n': len(questions), 'max_steps': max_steps, 'workers': workers,
        'answering_wall_clock_sec': round(elapsed, 1),
        'per_question_sec': {'min': round(min(per_q), 1) if per_q else 0,
                             'median': round(sorted(per_q)[len(per_q) // 2], 1) if per_q else 0,
                             'max': round(max(per_q), 1) if per_q else 0},
        'total_tool_calls': sum(v.get('n_tool_calls', 0) for v in store.values()),
        'python': sys.version.split()[0],
    }
    with open(os.path.join(config.LOG_DIR, f'run_{which}.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"回答フェーズ所要時間: {elapsed / 60:.1f} 分 "
          f"(1問あたり中央値 {meta['per_question_sec']['median']}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--questions', default='test')
    ap.add_argument('--model')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--max-steps', type=int, default=26)
    ap.add_argument('--tag', help='結果の保存先を分ける名前（既定はモデル名）')
    a = ap.parse_args()
    run(a.questions, a.model, a.workers, a.limit, a.force, a.max_steps, a.tag)


if __name__ == '__main__':
    main()
