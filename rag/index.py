"""Chunking and lexical retrieval over the extracted corpus.

Japanese text is tokenised with character bigrams (plus whole ASCII/number
tokens) rather than a morphological analyser: it needs no dictionary, is
deterministic, and matches identifier-like vocabulary (short alphanumeric codes,
column and parameter names) exactly rather than splitting it.

Run: python rag/index.py
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows consoles are cp932 here; a model answer containing e.g. U+202F would
# otherwise abort the whole run inside a progress print.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


import config
from dax.util import strip_fmt

CHUNK_CHARS = 2200
CHUNK_OVERLAP = 300

_ASCII = re.compile(r'[A-Za-z_][A-Za-z0-9_\-.]*|\d+(?:[.,]\d+)*')
_CJK = re.compile(r'[぀-ヿ㐀-鿿豈-﫿]')


def tokenize(text: str) -> list[str]:
    t = unicodedata.normalize('NFKC', text).lower()
    toks = _ASCII.findall(t)
    cjk_runs = re.findall(r'[぀-ヿ㐀-鿿豈-﫿]+', t)
    for run in cjk_runs:
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def _split_sections(text: str) -> list[tuple[str, str]]:
    """(anchor, body) pairs, preferring the document's own headings."""
    lines = text.split('\n')
    out: list[tuple[str, list[str]]] = []
    cur_anchor = ''
    cur: list[str] = []
    for ln in lines:
        m = re.match(r'^\s*#{1,4}\s*(.+)$', ln)
        if m:
            if cur:
                out.append((cur_anchor, cur))
            cur_anchor, cur = m.group(1).strip()[:80], [ln]
        else:
            cur.append(ln)
    if cur:
        out.append((cur_anchor, cur))
    return [(a, '\n'.join(b)) for a, b in out]


def chunk_document(rec_text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for anchor, body in _split_sections(rec_text):
        if len(body) <= CHUNK_CHARS:
            if body.strip():
                chunks.append((anchor, body))
            continue
        start = 0
        while start < len(body):
            piece = body[start:start + CHUNK_CHARS]
            if piece.strip():
                chunks.append((anchor, piece))
            if start + CHUNK_CHARS >= len(body):
                break
            start += CHUNK_CHARS - CHUNK_OVERLAP
    return chunks


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.72):
        self.k1, self.b = k1, b
        self.docs: list[dict] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.lens: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        self.avg_len = 0.0

    def add(self, meta: dict, text: str) -> None:
        toks = tokenize(text)
        c = Counter(toks)
        i = len(self.docs)
        self.docs.append(meta)
        self.tf.append(c)
        self.lens.append(len(toks))
        for t in c:
            self.df[t] += 1
            self.postings[t].append(i)

    def finalise(self) -> None:
        self.avg_len = sum(self.lens) / max(len(self.lens), 1)
        self.postings = dict(self.postings)

    def search(self, query: str, top_k: int = 12,
               allow: set[int] | None = None) -> list[tuple[float, dict]]:
        qt = tokenize(query)
        if not qt:
            return []
        n = len(self.docs)
        scores: dict[int, float] = defaultdict(float)
        for t, qcount in Counter(qt).items():
            plist = self.postings.get(t)
            if not plist:
                continue
            df = self.df[t]
            if df > n * 0.6:            # near-stopword
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i in plist:
                if allow is not None and i not in allow:
                    continue
                f = self.tf[i][t]
                dl = self.lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(s, self.docs[i]) for i, s in ranked]


def build() -> None:
    config.ensure_dirs()
    man = json.load(open(os.path.join(config.ARTIFACTS, 'manifest.json'), encoding='utf-8'))

    # OCR output has to be indexed too, otherwise the contents of scanned pages
    # and figures stay invisible to search even though they were transcribed.
    notes: dict[str, str] = {}
    images: list[dict] = []
    vpath = os.path.join(config.ARTIFACTS, 'vision.json')
    ipath = os.path.join(config.ARTIFACTS, 'images.json')
    if os.path.exists(vpath) and os.path.exists(ipath):
        notes = json.load(open(vpath, encoding='utf-8'))
        images = json.load(open(ipath, encoding='utf-8'))
    by_doc: dict[str, list[dict]] = {}
    for e in images:
        if notes.get(e['image_id']):
            by_doc.setdefault(e['doc_id'], []).append(e)

    bm = BM25()
    n_chunks = 0
    n_img_chunks = 0
    for d in man['documents']:
        tp = os.path.join(config.TEXT_DIR, f"{d['id']}.md")
        if not os.path.exists(tp):
            continue
        text = open(tp, encoding='utf-8').read()
        for ci, (anchor, body) in enumerate(chunk_document(text)):
            # the path is part of the indexed text so filename mentions retrieve;
            # formatting markers are stripped so phrases tokenise continuously
            indexed = (f"{d['rel_path']}\n{d.get('project') or ''} "
                       f"{d.get('project_abbrev') or ''}\n{anchor}\n{strip_fmt(body)}")
            bm.add({'doc_id': d['id'], 'rel_path': d['rel_path'],
                    'project': d.get('project'), 'abbrev': d.get('project_abbrev'),
                    'ext': d['ext'], 'anchor': anchor, 'chunk': ci,
                    'preview': body[:400]}, indexed)
            n_chunks += 1

        for e in by_doc.get(d['id'], []):
            note = notes[e['image_id']]
            for ci, (anchor, body) in enumerate(chunk_document(note)):
                label = f"{e['origin']} {e['locator']}"
                indexed = (f"{d['rel_path']}\n{d.get('project') or ''} "
                           f"{d.get('project_abbrev') or ''}\n{label}\n{strip_fmt(body)}")
                bm.add({'doc_id': d['id'], 'rel_path': d['rel_path'],
                        'project': d.get('project'), 'abbrev': d.get('project_abbrev'),
                        'ext': d['ext'], 'anchor': f'画像の読み取り: {label}',
                        'chunk': ci, 'image_id': e['image_id'],
                        'preview': body[:400]}, indexed)
                n_img_chunks += 1
    bm.finalise()
    out = os.path.join(config.INDEX_DIR, 'bm25.pkl')
    # plain state, not the instance: unpickling must not depend on which module
    # happens to be __main__ in the loading process
    with open(out, 'wb') as fh:
        pickle.dump({'k1': bm.k1, 'b': bm.b, 'docs': bm.docs, 'tf': bm.tf,
                     'df': bm.df, 'lens': bm.lens, 'postings': bm.postings,
                     'avg_len': bm.avg_len}, fh, protocol=4)
    print(f'{n_chunks} chunks from {len(man["documents"])} documents '
          f'(+{n_img_chunks} from image transcriptions) -> {out}')


def load() -> BM25:
    with open(os.path.join(config.INDEX_DIR, 'bm25.pkl'), 'rb') as fh:
        state = pickle.load(fh)
    bm = BM25(k1=state['k1'], b=state['b'])
    bm.docs = state['docs']
    bm.tf = state['tf']
    bm.df = state['df']
    bm.lens = state['lens']
    bm.postings = state['postings']
    bm.avg_len = state['avg_len']
    return bm


if __name__ == '__main__':
    build()
