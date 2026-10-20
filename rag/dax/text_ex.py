"""Plain-text, code, JSON and delimited-data extraction."""
from __future__ import annotations

import csv
import io
import json
import os

from .util import squeeze

MAX_TEXT = 200000
SAMPLE_ROWS = 25


def _read_text(path: str) -> str:
    raw = open(path, 'rb').read()
    for enc in ('utf-8-sig', 'utf-8', 'cp932', 'shift_jis', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'replace')


def extract_table(path: str) -> dict:
    """CSV/TSV: header, dtypes, row count and a head/tail sample.

    The full grid is deliberately *not* inlined — downstream code reads these
    files with pandas when a question needs aggregation.
    """
    delim = '\t' if path.lower().endswith(('.tsv', '.tab')) else ','
    text = _read_text(path)
    rdr = csv.reader(io.StringIO(text), delimiter=delim)
    rows = []
    total = 0
    for i, r in enumerate(rdr):
        total += 1
        if i < SAMPLE_ROWS or i < 1:
            rows.append(r)
    header = rows[0] if rows else []
    doc = {'kind': 'table', 'delimiter': delim, 'n_rows_including_header': total,
           'n_cols': len(header), 'header': header, 'sample': rows[:SAMPLE_ROWS]}

    prof = []
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=delim, encoding='utf-8-sig', low_memory=False)
        doc['n_rows'] = int(len(df))
        for col in df.columns:
            s = df[col]
            info = {'name': str(col), 'dtype': str(s.dtype),
                    'n_missing': int(s.isna().sum()),
                    'n_unique': int(s.nunique(dropna=True))}
            if str(s.dtype).startswith(('int', 'float')):
                d = s.dropna()
                if len(d):
                    info.update(min=float(d.min()), max=float(d.max()),
                                mean=float(d.mean()))
            else:
                vc = s.dropna().astype(str).value_counts().head(8)
                info['top_values'] = {str(k): int(v) for k, v in vc.items()}
            prof.append(info)
        doc['columns'] = prof
    except Exception as e:  # pragma: no cover - profiling is best effort
        doc['profile_error'] = f'{type(e).__name__}: {e}'

    doc['text'] = render_table(path, doc)
    return doc


def render_table(path: str, doc: dict) -> str:
    lines = [f"データファイル {os.path.basename(path)}",
             f"行数(ヘッダー除く)={doc.get('n_rows', doc['n_rows_including_header'] - 1)}, "
             f"列数={doc['n_cols']}"]
    if doc.get('columns'):
        lines.append('■ 列プロファイル:')
        for c in doc['columns']:
            bits = [f"{c['name']}", f"型={c['dtype']}", f"欠損={c['n_missing']}",
                    f"ユニーク={c['n_unique']}"]
            if 'min' in c:
                bits.append(f"min={c['min']:.6g} max={c['max']:.6g} mean={c['mean']:.6g}")
            if c.get('top_values'):
                bits.append('頻出値=' + ', '.join(f'{k}({v})' for k, v in c['top_values'].items()))
            lines.append('  - ' + ', '.join(bits))
    lines.append('■ 先頭サンプル:')
    for r in doc['sample']:
        lines.append('  ' + ' | '.join(r))
    return squeeze('\n'.join(lines))


def extract_json(path: str) -> dict:
    text = _read_text(path)
    try:
        obj = json.loads(text)
        pretty = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        obj, pretty = None, text
    if len(pretty) > MAX_TEXT:
        pretty = pretty[:MAX_TEXT] + '\n...(省略)'
    return {'kind': 'json', 'data': obj, 'text': pretty}


def extract_text(path: str) -> dict:
    text = _read_text(path)
    kind = 'code' if os.path.splitext(path)[1].lower() in (
        '.py', '.toml', '.lock', '.cfg', '.ini', '.yaml', '.yml', '.sh') else 'text'
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT] + '\n...(省略)'
    return {'kind': kind, 'text': text}
