"""Runs model-authored analysis code in a separate process.

Questions routinely require real computation over the delivered data (group
means, correlations, fitting a threshold that maximises F1, day counts).  The
agent writes ordinary pandas code; this module executes it with a timeout and
returns stdout, so a bad snippet cannot hang or crash the pipeline.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

TIMEOUT = 180

PREAMBLE = '''
import os, sys, json, math, re, unicodedata, glob, zipfile, datetime, random
import numpy as np, pandas as pd
# fixed seeds so any sampling/model fitting the analysis does is reproducible
random.seed({seed})
np.random.seed({seed})
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 200)
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"""{rag_dir}""")

DRIVE = r"""{drive}"""

def find(*parts):
    """Case/normalisation-insensitive file lookup under the share drive.

    Give any number of path fragments; every fragment must appear somewhere in
    the file's relative path.  Use this instead of writing literal paths: the
    drive uses NFD-normalised Japanese names that will not match a typed string.
    """
    import unicodedata as _u
    def key(s):
        s = _u.normalize('NFKC', s).lower()
        return re.sub(r'[\\s\\u3000_\\-]+', '', s)
    hits = []
    for dp, dn, fn in os.walk(DRIVE):
        dn[:] = [d for d in dn if d != '__pycache__']
        for f in fn:
            p = os.path.join(dp, f)
            rel = _u.normalize('NFC', os.path.relpath(p, DRIVE))
            if all(key(x) in key(rel) for x in parts):
                hits.append(p)
    return sorted(hits)

def ls(*parts):
    for p in find(*parts):
        print(unicodedata.normalize('NFC', os.path.relpath(p, DRIVE)))

def read_csv(*parts, **kw):
    hits = find(*parts)
    if not hits:
        raise FileNotFoundError(parts)
    kw.setdefault('encoding', 'utf-8-sig')
    kw.setdefault('low_memory', False)
    return pd.read_csv(hits[0], **kw)

def read_excel(*parts, **kw):
    hits = find(*parts)
    if not hits:
        raise FileNotFoundError(parts)
    return pd.read_excel(hits[0], **kw)

# ---- extracted-corpus access (書式・数式・グラフ値つきのテキスト) ----------
ARTIFACTS = r"""{artifacts}"""
_MAN = json.load(open(os.path.join(ARTIFACTS, 'manifest.json'), encoding='utf-8'))
DOCS = _MAN['documents']

def plain(t):
    """書式マーカー [[bold]] などを除去した素のテキストにする。
    正規表現で語句を拾うときは必ずこれを通すこと（マーカーが語句を分断するため）。"""
    return re.sub(r'\\[\\[/?[^\\]]*\\]\\]', '', t)

def docs(*parts):
    """抽出済みドキュメントのメタ情報を部分一致で絞り込む。"""
    import unicodedata as _u
    def key(s):
        return re.sub(r'[\\s\\u3000_\\-]+', '', _u.normalize('NFKC', s).lower())
    return [d for d in DOCS if all(key(x) in key(d['rel_path']) for x in parts)]

def doc_text(*parts, raw=False):
    """最初に一致したドキュメントの抽出テキスト。既定では書式マーカーを除去済み。
    raw=True にすると [[bold]] などのマーカー付きで返す。"""
    hits = docs(*parts)
    if not hits:
        raise FileNotFoundError(parts)
    p = os.path.join(ARTIFACTS, 'text', hits[0]['id'] + '.md')
    t = open(p, encoding='utf-8').read()
    return t if raw else plain(t)

def each_text(*parts, raw=False):
    """一致した全ドキュメントについて (rel_path, text) を順に返す。
    案件横断の集計はこれで全案件を一度に走査できる。"""
    for d in docs(*parts):
        p = os.path.join(ARTIFACTS, 'text', d['id'] + '.md')
        if os.path.exists(p):
            t = open(p, encoding='utf-8').read()
            yield d['rel_path'], (t if raw else plain(t))
'''


def run(code: str, drive_root: str, rag_dir: str, timeout: int = TIMEOUT,
        artifacts: str | None = None, seed: int = 0) -> str:
    artifacts = artifacts or os.path.join(rag_dir, 'artifacts')
    src = (PREAMBLE.format(drive=drive_root, rag_dir=rag_dir, artifacts=artifacts,
                           seed=seed)
           + '\n' + textwrap.dedent(code))
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False,
                                     encoding='utf-8') as fh:
        fh.write(src)
        path = fh.name
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONWARNINGS='ignore')
    try:
        p = subprocess.run([sys.executable, '-X', 'utf8', path],
                           capture_output=True, timeout=timeout, env=env)
        out = p.stdout.decode('utf-8', 'replace')
        err = p.stderr.decode('utf-8', 'replace')
        # strip the noisy library warnings that swamp real errors
        err_lines = [l for l in err.split('\n')
                     if l.strip() and 'Warning' not in l and 'warn(' not in l]
        if err_lines and p.returncode != 0:
            out += '\n[stderr]\n' + '\n'.join(err_lines[-40:])
        elif err_lines:
            out += '\n[stderr(warn)]\n' + '\n'.join(err_lines[-10:])
        if not out.strip():
            out = '(コードは正常終了しましたが標準出力がありません。print() で結果を出力してください)'
        return out[:24000]
    except subprocess.TimeoutExpired:
        return f'[エラー] 実行が {timeout} 秒でタイムアウトしました。処理を軽くしてください。'
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
