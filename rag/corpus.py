"""Builds the document store: every file in the share drive is parsed with a
format-appropriate generic extractor and written out as structured JSON plus a
text rendering used for retrieval and for LLM context.

Run:  python rag/corpus.py [--force] [--no-render]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows consoles are cp932 here; a model answer containing e.g. U+202F would
# otherwise abort the whole run inside a progress print.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


import config
import glossary as glossary_mod
from dax import (chart_ex, com_render, docx_ex, nb_ex, pdf_ex, pptx_ex,
                 text_ex, unlock, xlsx_ex)
from dax.util import nfc, walk_files

OFFICE = {'.docx': 'docx', '.pptx': 'pptx', '.xlsx': 'xlsx'}
TABLE_EXT = {'.csv', '.tsv'}
JSON_EXT = {'.json', '.ipynb_meta'}
TEXT_EXT = {'.md', '.txt', '.py', '.toml', '.lock', '.cfg', '.ini', '.yaml',
            '.yml', '.sh', '.r', '.sql', '.rst'}
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
SKIP_EXT = {'.pyc', '.pyo', '.dll', '.so'}


def doc_id(rel: str) -> str:
    return hashlib.sha1(nfc(rel).encode('utf-8')).hexdigest()[:16]


def _dates_near(path: str, project_folder: str | None) -> list[str]:
    """8-digit date candidates visible in the file's own or sibling names."""
    out: list[str] = []
    scan = [path]
    if project_folder and os.path.isdir(project_folder):
        scan += walk_files(project_folder)
    for p in scan:
        for m in re.finditer(r'(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})', nfc(os.path.basename(p))):
            out.append(m.group(1) + m.group(2) + m.group(3))
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


class Builder:
    def __init__(self, force: bool = False, render: bool = True):
        self.force = force
        self.render = render
        config.ensure_dirs()
        self.root = config.drive_root()
        self.glossary = glossary_mod.load(self.root)
        self.projects = [p for p in self.glossary.projects if p.folder]
        self.manifest: list[dict] = []
        self.errors: list[dict] = []
        # documents that state how protected files are named/derived, and the
        # per-project text used to source date values; both are filled during
        # the first pass so nothing about the rule is baked into the code
        self.rule_texts: list[str] = []
        self.project_texts: dict[str, list[str]] = {}

    def _dates_from_text(self, proj) -> list[str]:
        """8-digit dates mentioned in a project's own documents, contract-like
        contexts first."""
        texts = self.project_texts.get(proj.display if proj else '', [])
        primary, secondary = [], []
        for t in texts:
            for m in re.finditer(r'(20\d{2})[-/.]?(\d{2})[-/.]?(\d{2})', t):
                ymd = m.group(1) + m.group(2) + m.group(3)
                window = t[max(0, m.start() - 60):m.start()]
                if any(k in window for k in ('契約期間', '開始', '効力発生', '締結')):
                    primary.append(ymd)
                else:
                    secondary.append(ymd)
        return list(dict.fromkeys(primary + secondary))[:60]

    # ------------------------------------------------------------- routing
    def project_of(self, path: str):
        best = None
        for p in self.projects:
            if os.path.abspath(path).startswith(os.path.abspath(p.folder) + os.sep):
                if best is None or len(p.folder) > len(best.folder):
                    best = p
        return best

    # ------------------------------------------------------------ extraction
    def extract_one(self, path: str) -> dict:
        ext = os.path.splitext(path)[1].lower()
        proj = self.project_of(path)
        rel = nfc(os.path.relpath(path, self.root))
        rec: dict = {
            'id': doc_id(rel),
            'rel_path': rel,
            'abs_path': path,
            'name': nfc(os.path.basename(path)),
            'ext': ext,
            'size': os.path.getsize(path),
            'folder': nfc(os.path.dirname(rel)),
            'project': proj.display if proj else None,
            'project_abbrev': proj.abbrev if proj else None,
            'encrypted': False,
        }

        source: object = path
        if ext in OFFICE and unlock.is_encrypted(path):
            abbrevs = []
            if proj and proj.abbrev:
                abbrevs.append(proj.abbrev)
            abbrevs += [p.abbrev for p in self.projects if p.abbrev]
            dates = _dates_near(path, proj.folder if proj else None)
            dates += self._dates_from_text(proj)
            data, pw = unlock.unlock(path, abbrevs, list(dict.fromkeys(dates)),
                                     self.rule_texts)
            rec['encrypted'] = True
            if data is None:
                rec['error'] = 'encrypted: password not derivable'
                rec['text'] = ''
                return rec
            rec['password_recovered'] = True
            tmp = os.path.join(config.PDF_CACHE, f'dec_{rec["id"]}{ext}')
            with open(tmp, 'wb') as fh:
                fh.write(data)
            source = tmp
            rec['decrypted_copy'] = tmp

        if ext == '.docx':
            doc = docx_ex.extract(source if isinstance(source, str) else path)
        elif ext == '.pptx':
            doc = pptx_ex.extract(source if isinstance(source, str) else path)
        elif ext == '.xlsx':
            doc = xlsx_ex.extract(source if isinstance(source, str) else path)
        elif ext == '.pdf':
            doc = pdf_ex.extract(path)
        elif ext == '.ipynb':
            doc = nb_ex.extract(path)
        elif ext in TABLE_EXT:
            doc = text_ex.extract_table(path)
        elif ext == '.json':
            doc = text_ex.extract_json(path)
        elif ext in TEXT_EXT:
            doc = text_ex.extract_text(path)
        elif ext in IMAGE_EXT:
            doc = {'kind': 'image', 'text': f'[画像ファイル {rec["name"]}]'}
        else:
            doc = {'kind': 'binary', 'text': ''}

        rec.update(doc)

        # page-accurate layout for documents whose pagination is a question
        if self.render and ext in ('.docx', '.pptx'):
            src = source if isinstance(source, str) else path
            pdf = com_render.to_pdf(src, config.PDF_CACHE)
            if pdf:
                rec['rendered_pdf'] = pdf
                try:
                    p = pdf_ex.extract(pdf)
                    rec['n_pages'] = p['n_pages']
                    rec['page_text'] = [{'page': x['page'], 'text': x['text']}
                                        for x in p['pages']]
                except Exception as e:
                    rec['render_error'] = f'{type(e).__name__}: {e}'
        return rec

    # ----------------------------------------------------------------- build
    def build(self) -> None:
        files = [p for p in walk_files(self.root)
                 if os.path.splitext(p)[1].lower() not in SKIP_EXT]
        # Protected files go last: unlocking them needs the naming rule and the
        # dates that only become available once the rest of the drive is read.
        locked = [p for p in files
                  if os.path.splitext(p)[1].lower() in OFFICE and unlock.is_encrypted(p)]
        files = [p for p in files if p not in set(locked)] + locked
        print(f'drive root : {nfc(self.root)}')
        print(f'projects   : {len(self.projects)}')
        print(f'files      : {len(files)} ({len(locked)} protected, processed last)')
        print(f'renderer   : {"available" if com_render.available() else "NOT available"}')
        t0 = time.time()
        for i, path in enumerate(files, 1):
            rel = nfc(os.path.relpath(path, self.root))
            did = doc_id(rel)
            jpath = os.path.join(config.DOCS_DIR, f'{did}.json')
            tpath = os.path.join(config.TEXT_DIR, f'{did}.md')
            if not self.force and os.path.exists(jpath) and os.path.exists(tpath):
                rec = json.load(open(jpath, encoding='utf-8'))
            else:
                try:
                    rec = self.extract_one(path)
                except Exception as e:
                    traceback.print_exc()
                    self.errors.append({'path': rel, 'error': f'{type(e).__name__}: {e}'})
                    rec = {'id': did, 'rel_path': rel, 'abs_path': path,
                           'name': nfc(os.path.basename(path)),
                           'ext': os.path.splitext(path)[1].lower(),
                           'size': os.path.getsize(path), 'kind': 'error',
                           'error': f'{type(e).__name__}: {e}', 'text': ''}
                with open(jpath, 'w', encoding='utf-8') as fh:
                    json.dump(rec, fh, ensure_ascii=False, default=str)
                with open(tpath, 'w', encoding='utf-8') as fh:
                    fh.write(self.render_text(rec))
            self.manifest.append({k: rec.get(k) for k in
                                  ('id', 'rel_path', 'name', 'ext', 'size', 'kind',
                                   'project', 'project_abbrev', 'folder', 'n_pages',
                                   'encrypted', 'error')})
            self._collect(rec)
            if i % 25 == 0 or i == len(files):
                print(f'  [{i}/{len(files)}] {time.time() - t0:.0f}s  {rel[:70]}')

        with open(os.path.join(config.ARTIFACTS, 'manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump({'root': nfc(self.root),
                       'projects': [{'display': p.display, 'abbrev': p.abbrev,
                                     'official': p.official, 'aliases': p.aliases,
                                     'folder': nfc(os.path.relpath(p.folder, self.root))}
                                    for p in self.projects],
                       'glossary_terms': self.glossary.terms,
                       'documents': self.manifest,
                       'errors': self.errors},
                      fh, ensure_ascii=False, indent=1)
        print(f'\ndone in {time.time() - t0:.0f}s, {len(self.errors)} errors')
        for e in self.errors:
            print('  ERROR', e['path'], e['error'])

    def _collect(self, rec: dict) -> None:
        """Accumulate the drive's own text for later rule/date resolution."""
        text = rec.get('text') or ''
        if not text:
            return
        if 'パスワード' in text or 'password' in text.lower():
            self.rule_texts.append(text)
        proj = rec.get('project')
        if proj:
            self.project_texts.setdefault(proj, []).append(text)

    # ---------------------------------------------------------------- render
    @staticmethod
    def render_text(rec: dict) -> str:
        head = [f"# ファイル: {rec['rel_path']}"]
        if rec.get('project'):
            head.append(f"案件: {rec['project']} ({rec.get('project_abbrev')})")
        head.append(f"種別: {rec.get('kind')}  サイズ: {rec.get('size')} bytes")
        if rec.get('encrypted'):
            head.append('※ パスワード保護ファイル（社内規定の導出規則により復号）')
        if rec.get('n_pages'):
            head.append(f"レイアウト後のページ数: {rec['n_pages']}")
        if rec.get('error'):
            head.append(f"抽出エラー: {rec['error']}")
        body = rec.get('text') or ''
        parts = ['\n'.join(head), body]
        if rec.get('page_text'):
            parts.append('\n## ページ割り（レイアウト実測）')
            for p in rec['page_text']:
                snippet = (p['text'] or '')[:1500]
                parts.append(f"\n### p.{p['page']}\n{snippet}")
        return '\n\n'.join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--no-render', action='store_true')
    args = ap.parse_args()
    Builder(force=args.force, render=not args.no_render).build()


if __name__ == '__main__':
    main()
