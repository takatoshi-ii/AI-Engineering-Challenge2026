"""Mechanical audit of answers against the format each question demands.

The test set has no published answers, so correctness cannot be measured
directly.  What can be checked is whether an answer even has the shape the
question asked for — a question that says "主略称で" expects an abbreviation,
"小数第2位まで" expects two decimals, "何ページ" expects a number.  A mismatch
is a near-certain Incorrect and points straight at what to fix.

  python rag/tools/audit_answers.py --tag final --questions test
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import config
import run as run_mod

ABBREVS = ['KSS', 'AYM', 'SHR', 'AOSHIO', 'SOHK', 'TOTO', 'AOMINE',
           'KAEDE', 'AOBM', 'MINAMINO']


def checks(q: str, a: str) -> list[str]:
    """Format requirements the question states, and whether the answer meets them."""
    bad = []
    has_num = bool(re.search(r'\d', a))

    m = re.search(r'小数第\s*(\d+)\s*位', q)
    if m:
        want = int(m.group(1))
        # accept any number carrying at least the requested precision
        got = [len(x.split('.')[1]) for x in re.findall(r'\d+\.\d+', a)]
        if not got:
            bad.append(f'小数第{want}位の指定だが小数が無い')
        elif max(got) < want:
            bad.append(f'小数第{want}位の指定だが最大{max(got)}桁')

    if re.search(r'主略称|案件略称', q):
        if not any(re.search(rf'(?<![A-Za-z]){x}(?![A-Za-z])', a) for x in ABBREVS):
            bad.append('略称での回答を求めているが略称が無い')

    if re.search(r'何ページ|ページ番号|ページ数|第何週|何年|何月|何日|何人|何件|'
                 r'いくつ|何時間|何倍|いくら|何%|何パーセント', q) and not has_num:
        bad.append('数値を問うているが数値が無い')

    if re.search(r'内線番号|EXT', q) and not re.search(r'\d{3,}', a):
        bad.append('内線番号を問うているが番号が無い')

    if re.search(r'すべて(挙げ|答え|抽出|列挙)|をすべて', q):
        n = len([x for x in re.split(r'[、,]', a) if x.strip()])
        if n == 1 and len(a) > 60:
            bad.append('列挙を求めているが文章1つ')

    if re.search(r'フルネーム', q) and not re.search(r'[一-鿿]{1,4}\s?[一-鿿]{1,4}', a):
        bad.append('フルネームを求めているが氏名らしき語が無い')

    # a reply that restates the question instead of answering it
    if len(a) > 100 and a[:12] in q:
        bad.append('質問文をなぞった書き出し')
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--questions', default='test')
    a = ap.parse_args()

    qs = {q['index']: q['question'] for q in run_mod.load_questions(a.questions)}
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', a.tag)
    store = json.load(open(os.path.join(config.ANSWER_DIR,
                                        f'{a.questions}__{slug}.json'), encoding='utf-8'))

    flagged = 0
    for i in sorted(qs):
        ans = (store.get(str(i)) or {}).get('answer', '')
        probs = checks(qs[i], ans)
        if probs:
            flagged += 1
            print(f'Q{i:<3} {" / ".join(probs)}')
            print(f'      問: {qs[i][:78]}')
            print(f'      答: {ans[:78]}')
    print()
    print(f'形式が要求と食い違う回答: {flagged} / {len(qs)}')

    # secondary signals
    lens = [(i, len((store.get(str(i)) or {}).get('answer', ''))) for i in qs]
    long_ = [i for i, n in lens if n > 150]
    conf = {}
    for i in qs:
        c = (store.get(str(i)) or {}).get('confidence', '?')
        conf[c] = conf.get(c, 0) + 1
    calls = [(i, (store.get(str(i)) or {}).get('n_tool_calls', 0)) for i in qs]
    few = [i for i, n in calls if n <= 3]
    print(f'150文字超の回答          : {len(long_)} 件 {long_[:15]}')
    print(f'確信度の分布             : {conf}')
    print(f'ツール呼び出し3回以下     : {len(few)} 件 {few[:15]}')


if __name__ == '__main__':
    main()
