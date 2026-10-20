"""Ground-truth scoring that does not depend on an LLM's opinion.

The self-scoring used the same model that produced the answers, which flattered
them badly.  For the validation set the correct answers are known, so the useful
checks are mechanical:

  contains  - does the answer actually carry the expected string / all expected
              elements (normalised for spacing and list separators)?
  ratio     - how much extra text surrounds it?  The rubric treats an added
              element in an enumeration as Incorrect, so bulk is a real risk.

Usage: strict_eval.py TAG [--questions valid]
"""
import argparse, json, os, re, statistics, sys, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import config
import run as run_mod


def norm(s: str) -> str:
    s = unicodedata.normalize('NFKC', s or '').lower()
    return re.sub(r'[\s　,、。．\.]+', '', s)


def elements(s: str) -> list[str]:
    """Split an enumeration answer into its elements."""
    parts = re.split(r'[、,／/\n]|および|\band\b', s or '')
    return [norm(p) for p in parts if norm(p)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--questions', default='valid')
    ap.add_argument('--show', action='store_true', help='全件を表示')
    a = ap.parse_args()

    qs = {q['index']: q for q in run_mod.load_questions(a.questions)}
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', a.tag)
    store = json.load(open(os.path.join(config.ANSWER_DIR,
                                        f'{a.questions}__{slug}.json'), encoding='utf-8'))

    hit = miss = 0
    ratios = []
    rows = []
    for i, q in sorted(qs.items()):
        gt = (q.get('answer') or '').strip()
        if not gt:
            continue
        ans = (store.get(str(i)) or {}).get('answer', '').strip()
        ng, na = norm(gt), norm(ans)
        contains = ng in na
        # enumeration: every expected element must appear
        els = elements(gt)
        all_els = len(els) > 1 and all(e in na for e in els)
        ok = contains or all_els
        ratio = len(ans) / max(len(gt), 1)
        ratios.append(ratio)
        hit += ok
        miss += not ok
        rows.append((i, ok, ratio, gt, ans))

    n = len(rows)
    print(f'\n{"idx":>4} {"含":<3} {"倍率":>6}  正解 / 回答')
    print('-' * 100)
    for i, ok, ratio, gt, ans in rows:
        flag = 'OK ' if ok else 'NG '
        warn = ' ←冗長' if ratio > 2.5 else ''
        if a.show or not ok or ratio > 2.5:
            print(f'{i:>4} {flag} {ratio:>6.1f}{warn}')
            print(f'        正解: {gt[:80]}')
            print(f'        回答: {ans[:100]}')
    print('-' * 100)
    print(f'正解文字列を含む : {hit}/{n} ({hit/n*100:.0f}%)')
    print(f'含まない        : {miss}/{n}')
    print(f'長さ倍率        : 中央値 {statistics.median(ratios):.1f} / 平均 {statistics.mean(ratios):.1f}')
    print(f'倍率2.5超(冗長) : {sum(1 for r in ratios if r > 2.5)}/{n}')
    print()
    print('※ 「含む」は必要条件であって十分条件ではない。冗長な回答は')
    print('   列挙問題で「要素の追加」とみなされ Incorrect になりうる。')


if __name__ == '__main__':
    main()
