"""Run-time parsing of the in-drive glossary.

The share drive ships its own terminology document; the abbreviations used by
the questions are defined there.  Nothing is transcribed into code — the tables
are located by their header labels and parsed generically, so a new glossary
with new projects works unchanged.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dax import docx_ex
from dax.util import nfc, nkey, walk_files


@dataclass
class Project:
    folder: str                  # absolute path of the project folder
    display: str                 # folder name as shown
    official: str | None = None  # official company name from the glossary
    abbrev: str | None = None    # 主略称
    aliases: list[str] = field(default_factory=list)

    def all_names(self) -> list[str]:
        out = [self.display, self.official, self.abbrev, *self.aliases]
        return [x for x in out if x]


@dataclass
class Glossary:
    terms: dict[str, list[str]] = field(default_factory=dict)   # abbrev -> [official names]
    term_rows: list[dict] = field(default_factory=list)
    projects: list[Project] = field(default_factory=list)

    def expand(self, text: str) -> list[tuple[str, list[str]]]:
        """Abbreviations occurring in *text*, with their official meanings.

        Matching is whole-token for latin abbreviations so that "PP" does not
        fire inside "PPROC", and substring for Japanese ones.
        """
        hits: list[tuple[str, list[str]]] = []
        for abbr, meanings in self.terms.items():
            if not abbr:
                continue
            if re.fullmatch(r'[A-Za-z0-9\-]+', abbr):
                pat = r'(?<![A-Za-z0-9\-])' + re.escape(abbr) + r'(?![A-Za-z0-9\-])'
                if re.search(pat, text):
                    hits.append((abbr, meanings))
            elif abbr in text:
                hits.append((abbr, meanings))
        hits.sort(key=lambda kv: -len(kv[0]))
        return hits

    def match_projects(self, text: str) -> list[Project]:
        """Projects referenced by *text*, longest alias first."""
        t = nkey(text)
        scored: list[tuple[int, Project]] = []
        for p in self.projects:
            best = 0
            for name in p.all_names():
                k = nkey(name)
                if len(k) < 2:
                    continue
                if k in t:
                    best = max(best, len(k))
                elif re.fullmatch(r'[a-z0-9]+', k) and re.search(
                        r'(?<![a-z0-9])' + re.escape(k) + r'(?![a-z0-9])', t):
                    best = max(best, len(k))
            if best:
                scored.append((best, p))
        scored.sort(key=lambda kv: -kv[0])
        return [p for _, p in scored]


def _tables_from_docx(doc: dict) -> list[list[list[str]]]:
    out = []
    for b in doc['blocks']:
        if b['type'] == 'table':
            out.append([[c['text'].strip() for c in row] for row in b['rows']])
    return out


def _find_glossary_doc(root: str) -> str | None:
    """Locate the terminology document by content, not by a fixed name."""
    best, best_score = None, 0
    for p in walk_files(root):
        if not p.lower().endswith('.docx'):
            continue
        name = nkey(os.path.basename(p))
        score = 0
        if '用語' in nfc(os.path.basename(p)):
            score += 5
        if '略称' in nfc(os.path.basename(p)):
            score += 3
        if score > best_score:
            best, best_score = p, score
    return best


def load(drive_root: str) -> Glossary:
    g = Glossary()
    path = _find_glossary_doc(drive_root)
    if path:
        doc = docx_ex.extract(path)
        for tbl in _tables_from_docx(doc):
            if not tbl:
                continue
            header = [nkey(c) for c in tbl[0]]
            # term table: 正式名称 | 社内用語 | 補足
            if '正式名称' in tbl[0] and '社内用語' in tbl[0]:
                oi, ai = tbl[0].index('正式名称'), tbl[0].index('社内用語')
                for row in tbl[1:]:
                    if len(row) <= max(oi, ai):
                        continue
                    official, abbr = row[oi].strip(), row[ai].strip()
                    if not abbr or not official:
                        continue
                    for a in re.split(r'[/／,、]', abbr):
                        a = a.strip()
                        if a:
                            g.terms.setdefault(a, [])
                            if official not in g.terms[a]:
                                g.terms[a].append(official)
                    g.term_rows.append({'official': official, 'abbrev': abbr,
                                        'note': row[-1] if len(row) > 2 else ''})
            # project table: 案件名 | 主略称 | 別名候補 | 補足
            if '案件名' in tbl[0] and '主略称' in tbl[0]:
                ni, ai = tbl[0].index('案件名'), tbl[0].index('主略称')
                vi = tbl[0].index('別名候補') if '別名候補' in tbl[0] else None
                for row in tbl[1:]:
                    if len(row) <= max(ni, ai):
                        continue
                    official, abbr = row[ni].strip(), row[ai].strip()
                    aliases = []
                    if vi is not None and len(row) > vi:
                        aliases = [x.strip() for x in re.split(r'[,、／/]', row[vi]) if x.strip()]
                    g.projects.append(Project(folder='', display=official,
                                              official=official, abbrev=abbr,
                                              aliases=aliases))
                    g.terms.setdefault(abbr, [])
                    if official not in g.terms[abbr]:
                        g.terms[abbr].append(official)

    # bind glossary rows to the actual project folders on disk
    proj_root = None
    for dirpath, dirnames, _ in os.walk(drive_root):
        for d in dirnames:
            if nkey(d) in ('プロジェクト', 'projects', 'project'):
                proj_root = os.path.join(dirpath, d)
                break
        if proj_root:
            break

    folders = []
    if proj_root:
        folders = [os.path.join(proj_root, d) for d in sorted(os.listdir(proj_root))
                   if os.path.isdir(os.path.join(proj_root, d))]

    bound: list[Project] = []
    used = set()
    for folder in folders:
        disp = nfc(os.path.basename(folder))
        match = None
        for p in g.projects:
            if p.official and nkey(p.official) == nkey(disp):
                match = p
                break
        if match is None:                     # fall back to alias containment
            for p in g.projects:
                names = [n for n in p.all_names() if len(nkey(n)) >= 2]
                if any(nkey(n) in nkey(disp) for n in names):
                    match = p
                    break
        if match is not None:
            used.add(id(match))
            bound.append(Project(folder=folder, display=disp, official=match.official,
                                 abbrev=match.abbrev, aliases=match.aliases))
        else:
            bound.append(Project(folder=folder, display=disp, official=disp))
    # glossary entries with no folder (e.g. CROSS / INTERNAL markers) are kept
    for p in g.projects:
        if id(p) not in used and p.abbrev:
            bound.append(p)
    g.projects = bound
    return g
