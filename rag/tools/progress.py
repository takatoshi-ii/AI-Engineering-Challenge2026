"""Progress report for a running answer job. Usage: progress.py [tag] [questions]"""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import config

tag = sys.argv[1] if len(sys.argv) > 1 else 'final'
which = sys.argv[2] if len(sys.argv) > 2 else 'test'
total = 100 if which == 'test' else 30

p = os.path.join(config.ANSWER_DIR, f'{which}__{tag}.json')
if not os.path.exists(p):
    print('まだ結果がありません')
    raise SystemExit

d = json.load(open(p, encoding='utf-8'))
el = [v['elapsed'] for v in d.values()]
n = len(d)
pct = n * 100 // total
bar = '█' * (pct // 5) + '·' * (20 - pct // 5)

err = sorted((k for k, v in d.items() if v.get('errored')), key=int)
giveup = sorted((k for k, v in d.items()
                 if any(x in v['answer'] for x in ('確認できません', 'わかりません'))), key=int)

print(f'[{bar}] {n}/{total} 問 ({pct}%)')
print(f'  1問あたり : 中央値 {statistics.median(el):.0f}s / 平均 {statistics.mean(el):.0f}s / 最大 {max(el):.0f}s')
print(f'  積算経過   : {sum(el)/60:.0f} 分')
print(f'  残り見込み : {statistics.mean(el)*(total-n)/60:.0f} 分')
print(f'  実行エラー : {len(err)} 件 {err if err else ""}')
print(f'  未確認回答 : {len(giveup)} 件 {giveup if len(giveup) <= 15 else giveup[:15]}')

recent = sorted(d, key=lambda k: -d[k].get('_seq', int(k)))[:0]
last = sorted(d, key=int)[-3:]
print('  直近:')
for k in last:
    v = d[k]
    print(f"    Q{k} ({v['elapsed']:.0f}s/{v['n_tool_calls']}呼) {v['answer'][:55]}")
