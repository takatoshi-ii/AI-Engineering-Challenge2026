"""Score stored answers with the organiser's own evaluator.

Self-scoring with the model that wrote the answers proved worthless — it graded
its own verbosity generously.  evaluation/crag.py is the real thing, so this
feeds a stored answer set into it and reports the result per question.

The OpenAI key is used only here, for judging.  Answer generation stays local.

  python rag/tools/official_eval.py --tag terse --questions valid
"""
import argparse, csv, json, os, re, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import config
import run as run_mod

EVAL_DIR = os.path.join(config.BASE, 'evaluation')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--questions', default='valid')
    ap.add_argument('--model', default='gpt-5.2-2025-12-11')
    a = ap.parse_args()

    questions = run_mod.load_questions(a.questions)
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', a.tag)
    store = json.load(open(os.path.join(config.ANSWER_DIR,
                                        f'{a.questions}__{slug}.json'), encoding='utf-8'))

    # the evaluator reads submit/predictions.csv, headerless: index, answer
    sub = os.path.join(EVAL_DIR, 'submit', 'predictions.csv')
    with open(sub, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        for q in questions:
            ans = (store.get(str(q['index'])) or {}).get('answer') or run_mod.FALLBACK
            w.writerow([q['index'], ' '.join(ans.split())])
    print(f'{len(questions)} 件を {sub} に書き出し')

    key = subprocess.check_output(
        ['powershell', '-NoProfile', '-Command',
         '[Environment]::GetEnvironmentVariable("OPENAI_API_KEY","User")'],
        text=True).strip()
    env = dict(os.environ, OPENAI_API_KEY=key, PYTHONIOENCODING='utf-8')

    # crag.py and tiktoken both hit TLS-intercepted endpoints here; injecting the
    # OS trust store from a sitecustomize shim fixes both without touching the
    # organiser's code.
    shim = os.path.join(config.ARTIFACTS, 'evalshim')
    os.makedirs(shim, exist_ok=True)
    with open(os.path.join(shim, 'sitecustomize.py'), 'w', encoding='utf-8') as fh:
        fh.write('try:\n    import truststore\n    truststore.inject_into_ssl()\n'
                 'except Exception:\n    pass\n')
    env['PYTHONPATH'] = shim + os.pathsep + env.get('PYTHONPATH', '')

    print(f'公式採点を実行 (model={a.model}) …')
    p = subprocess.run([sys.executable, '-X', 'utf8', 'crag.py',
                        '--model-name', a.model],
                       cwd=EVAL_DIR, env=env, capture_output=True)
    out = p.stdout.decode('utf-8', 'replace')
    err = p.stderr.decode('utf-8', 'replace')
    print(out[-3000:])
    if p.returncode != 0:
        print('[stderr]', err[-2000:])
        return

    # per-question detail written by the evaluator
    res = os.path.join(EVAL_DIR, 'result', 'scoring.csv')
    if not os.path.exists(res):
        return
    gt = {int(r[0]): r[1] for r in csv.reader(open(
        os.path.join(EVAL_DIR, 'data', 'valid_txt.csv'), encoding='utf-8')) if r}
    print(f'\n{"idx":>4} {"判定":<11} 回答 / 正解')
    print('-' * 100)
    counts = {}
    for row in csv.reader(open(res, encoding='utf-8')):
        if not row:
            continue
        i, label = int(row[0]), row[1]
        counts[label] = counts.get(label, 0) + 1
        ans = (store.get(str(i)) or {}).get('answer', '')
        mark = {'Perfect': 'OK ', 'Acceptable': 'ACC', 'Missing': '-- ',
                'Incorrect': 'XX '}.get(label, '?  ')
        print(f'{i:>4} {mark} {label:<11} {ans[:60]}')
        if label != 'Perfect':
            print(f'{"":>4}     正解: {gt.get(i, "")[:70]}')
    print('-' * 100)
    print('  ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())))

    dst = os.path.join(config.LOG_DIR, f'official_{a.questions}__{slug}.csv')
    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(res, encoding='utf-8') as s, open(dst, 'w', encoding='utf-8') as d:
        d.write(s.read())
    print(f'-> {dst}')


if __name__ == '__main__':
    main()
