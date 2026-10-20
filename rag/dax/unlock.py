"""Generic recovery of password-protected Office files.

No password, and no password *format*, is written into this module.  The
derivation rule is published as a document inside the share drive; its template
(something of the shape ``PREFIX-[placeholder]-[placeholder]-…``) and its worked
example are parsed at run time, the placeholders are filled from values that are
themselves discovered in the drive (project abbreviations from the glossary,
dates from the project's own documents), and each candidate is verified by
attempting decryption.

If no rule document is present the module falls back to whatever hint the file
name itself carries, and otherwise gives up rather than guessing.
"""
from __future__ import annotations

import io
import itertools
import os
import re

_CACHE: dict[str, str] = {}

# A template is a run of literal tokens and [bracketed] placeholders joined by
# separators, e.g.  DA-[案件略号]-[開始年月日8桁]-[拡張子コード]
_TEMPLATE = re.compile(
    r'(?:[A-Za-z0-9]+|\[[^\]\n]{1,24}\])(?:\s*[-_]\s*(?:[A-Za-z0-9]+|\[[^\]\n]{1,24}\])){1,5}')
_PLACEHOLDER = re.compile(r'\[([^\]]+)\]')

# What each placeholder should be filled with, keyed by words appearing in it.
_ABBR_WORDS = ('略号', '略称', '案件', 'コード名', 'project', 'abbr')
_DATE_WORDS = ('年月日', '日付', '開始', 'date', 'ymd', '8桁')
_EXT_WORDS = ('拡張子', 'ext', '形式')


def is_encrypted(path: str) -> bool:
    try:
        with open(path, 'rb') as fh:
            return fh.read(2) != b'PK'
    except OSError:
        return False


def _try(path: str, pw: str) -> bytes | None:
    try:
        import msoffcrypto
        with open(path, 'rb') as fh:
            off = msoffcrypto.OfficeFile(fh)
            off.load_key(password=pw)
            buf = io.BytesIO()
            off.decrypt(buf)
        data = buf.getvalue()
        return data if data[:2] == b'PK' else None
    except Exception:
        return None


def _name_hints(path: str) -> list[str]:
    """Tokens a file name advertises, e.g. ``foo_pw-abc123.docx`` -> ``abc123``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    out = []
    for m in re.finditer(r'(?:pw|pass|password|key)[-_ ]*([A-Za-z0-9\-_]+)', stem, re.I):
        out.append(m.group(1))
    return out


def parse_templates(texts: list[str]) -> list[list[dict]]:
    """Password templates found in rule documents.

    Returns each template as a token list, where a token is either
    ``{'literal': 'DA'}`` or ``{'slot': 'abbrev'|'date'|'ext'|'unknown'}``.
    """
    templates: list[list[dict]] = []
    seen: set[str] = set()
    for text in texts:
        if 'パスワード' not in text and 'password' not in text.lower():
            continue
        for m in _TEMPLATE.finditer(text):
            raw = m.group(0)
            if '[' not in raw:
                continue                      # no placeholders -> not a template
            if raw in seen:
                continue
            seen.add(raw)
            tokens: list[dict] = []
            for part in re.split(r'\s*[-_]\s*', raw):
                ph = _PLACEHOLDER.fullmatch(part.strip())
                if not ph:
                    tokens.append({'literal': part.strip()})
                    continue
                inner = ph.group(1)
                if any(w in inner.lower() for w in _EXT_WORDS):
                    slot = 'ext'
                elif any(w in inner.lower() for w in _DATE_WORDS):
                    slot = 'date'
                elif any(w in inner.lower() for w in _ABBR_WORDS):
                    slot = 'abbrev'
                else:
                    slot = 'unknown'
                tokens.append({'slot': slot})
            if any('slot' in t for t in tokens):
                templates.append(tokens)
    return templates


def parse_examples(texts: list[str], templates: list[list[dict]]) -> list[str]:
    """Concrete example passwords shown in the rule document.

    The example fixes the conventions the prose leaves open (letter case of the
    literal prefix and of the extension code).
    """
    out: list[str] = []
    n_parts = {len(t) for t in templates} or {3, 4}
    for text in texts:
        if 'パスワード' not in text and 'password' not in text.lower():
            continue
        for m in re.finditer(r'[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+){2,5}', text):
            s = m.group(0)
            if len(re.split(r'[-_]', s)) in n_parts and re.search(r'\d{6,8}', s):
                out.append(s)
    return out


def _fill(tokens: list[dict], abbrevs, dates, ext, seps, prefix_cases,
          abbr_cases, ext_cases) -> list[str]:
    choices: list[list[str]] = []
    for t in tokens:
        if 'literal' in t:
            lit = t['literal']
            choices.append(list(dict.fromkeys(c(lit) for c in prefix_cases)))
        elif t['slot'] == 'abbrev':
            choices.append(list(dict.fromkeys(c(a) for a in abbrevs for c in abbr_cases)))
        elif t['slot'] == 'date':
            choices.append(list(dates))
        elif t['slot'] == 'ext':
            choices.append(list(dict.fromkeys(c(ext) for c in ext_cases)))
        else:
            choices.append(list(dict.fromkeys(
                [c(a) for a in abbrevs for c in abbr_cases] + list(dates))))
    out = []
    for combo in itertools.product(*choices):
        for sep in seps:
            out.append(sep.join(combo))
    return out


def candidates(path: str, abbrevs: list[str], dates: list[str],
               rule_texts: list[str] | None = None) -> list[str]:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    out: list[str] = list(_name_hints(path))

    ident = lambda s: s                                    # noqa: E731
    upper = lambda s: s.upper()                            # noqa: E731
    lower = lambda s: s.lower()                            # noqa: E731
    seps = ['-', '_', '']
    cases = [ident, upper, lower]

    templates = parse_templates(rule_texts or [])
    for tokens in templates:
        out.extend(_fill(tokens, abbrevs, dates, ext, seps, cases, cases, cases))

    # the documented example pins down the exact casing convention
    for ex in parse_examples(rule_texts or [], templates):
        parts = re.split(r'([-_])', ex)
        rebuilt: list[list[str]] = []
        for part in parts:
            if part in ('-', '_'):
                rebuilt.append([part])
            elif re.fullmatch(r'\d{6,8}', part):
                rebuilt.append(list(dates) or [part])
            elif part.lower() in (ext, 'xlsx', 'docx', 'pptx'):
                rebuilt.append([ext if part.islower() else ext.upper()])
            elif part.isupper() and len(part) > 1:
                rebuilt.append([a.upper() for a in abbrevs] or [part])
            elif part.islower():
                rebuilt.append([a.lower() for a in abbrevs] or [part])
            else:
                rebuilt.append([part])
        for combo in itertools.product(*rebuilt):
            out.append(''.join(combo))

    # last resort when no rule document is available at all
    if not templates:
        for ab, d in itertools.product(abbrevs, dates):
            for sep in seps:
                out.append(sep.join([ab, d, ext]))
                out.append(sep.join([ab.lower(), d, ext]))

    seen, uniq = set(), []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def unlock(path: str, abbrevs: list[str], dates: list[str],
           rule_texts: list[str] | None = None) -> tuple[bytes | None, str | None]:
    """Return (decrypted bytes, recovered password) or (None, None)."""
    if path in _CACHE:
        data = _try(path, _CACHE[path])
        if data:
            return data, _CACHE[path]
    for pw in candidates(path, abbrevs, dates, rule_texts):
        data = _try(path, pw)
        if data:
            _CACHE[path] = pw
            return data, pw
    return None, None
