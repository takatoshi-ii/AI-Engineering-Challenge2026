"""Check the submission against the format rules before uploading.

Mirrors the validator described in evaluation/readme.md: file name, extension,
column count, row coverage, blank answers, dtype and the per-answer token cap.
"""
import csv, os, sys, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import config
import run as run_mod

FAIL = []


def chk(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {detail}")
    if not ok:
        FAIL.append(name)


csv_path = os.path.join(config.SUBMIT_DIR, 'predictions.csv')
zip_path = os.path.join(config.SUBMIT_DIR, 'submission.zip')

print('=== ファイル ===')
chk('predictions.csv が存在する', os.path.exists(csv_path), csv_path)
chk('submission.zip が存在する', os.path.exists(zip_path))

if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    chk('zip 直下が predictions.csv のみ',
        names == ['predictions.csv'], str(names))
    base = os.path.splitext(os.path.basename(zip_path))[0]
    ok_name = all(c.isalnum() or c in '_-' for c in base)
    chk('zip 名が英数字/_/- のみ', ok_name, base)

print('\n=== 中身 ===')
rows = []
with open(csv_path, encoding='utf-8', newline='') as fh:
    for r in csv.reader(fh):
        rows.append(r)

questions = run_mod.load_questions('test')
want = [q['index'] for q in questions]

chk('ヘッダー行が無い', bool(rows) and rows[0][0].strip().isdigit(),
    f'1行目: {rows[0] if rows else "(空)"}')
chk('列数が 2（index, answer）',
    all(len(r) == 2 for r in rows), f'列数の種類: {sorted({len(r) for r in rows})}')
chk('行数が設問数と一致', len(rows) == len(want), f'{len(rows)} 行 / 設問 {len(want)}')

idx = [int(r[0]) for r in rows if r and r[0].strip().lstrip("-").isdigit()]
chk('index が全問を過不足なく網羅', sorted(idx) == sorted(want),
    f'不足: {sorted(set(want) - set(idx))[:5]} 余分: {sorted(set(idx) - set(want))[:5]}')
chk('index に重複が無い', len(idx) == len(set(idx)))

blank = [r[0] for r in rows if len(r) < 2 or not r[1].strip()]
chk('空の回答が無い', not blank, f'空: {blank[:8]}')

nl = [r[0] for r in rows if len(r) > 1 and ('\n' in r[1] or '\r' in r[1])]
chk('回答に改行が混入していない', not nl, f'該当: {nl[:5]}')

print('\n=== トークン数 ===')
try:
    import tiktoken
    enc = tiktoken.get_encoding('o200k_base')
    counts = [(int(r[0]), len(enc.encode(r[1]))) for r in rows if len(r) > 1]
    over = [(i, n) for i, n in counts if n > config.MAX_ANSWER_TOKENS]
    mx = max(counts, key=lambda x: x[1]) if counts else (0, 0)
    chk(f'全回答が {config.MAX_ANSWER_TOKENS} トークン以下', not over,
        f'最大 Q{mx[0]}={mx[1]}tok' + (f' / 超過 {over[:5]}' if over else ''))
    avg = sum(n for _, n in counts) / max(len(counts), 1)
    print(f'  平均 {avg:.0f} tok / 最大 {mx[1]} tok')
except ImportError:
    print('  (tiktoken 未導入のため未検査)')

print('\n=== 回答の中身の傾向 ===')
giveup = [r[0] for r in rows if len(r) > 1 and
          any(k in r[1] for k in ('確認できません', 'わかりません', '見つかりません'))]
print(f'  「確認できません」系の回答: {len(giveup)} 件  {giveup[:12]}')

print('\n' + ('提出可能な形式です。' if not FAIL else f'{len(FAIL)} 件 不合格: {FAIL}'))
sys.exit(1 if FAIL else 0)
