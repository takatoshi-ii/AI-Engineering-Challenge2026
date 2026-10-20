"""PDF extraction with page-accurate text, tables, annotations and figures.

Page numbers matter (several questions ask "on which page ..."), so text is
always kept partitioned by page.  Pages that carry little or no extractable
text are flagged as image pages so the caller can route them to OCR/vision.
"""
from __future__ import annotations

import io
import os
import re

import fitz  # PyMuPDF

from .util import color_info, color_label, squeeze, strip_fmt


_FLAG_ITALIC = 1 << 1
_FLAG_BOLD = 1 << 4


_TM_RE = re.compile(
    rb'(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+Tm')


def _italic_runs(page) -> list[tuple[float, float, float]]:
    """Horizontal ranges (y, x_start, x_end) drawn with a skewed text matrix.

    Word writes *synthetic* italic for fonts that have no italic face: the
    glyphs are slanted by the text matrix, so the font name and flags look
    upright and only the content stream reveals it.  A skewed run lasts until
    the next text-matrix operator on the same baseline.
    """
    try:
        raw = page.read_contents()
    except Exception:
        return []
    h = page.rect.height
    ops: list[tuple[float, float, bool]] = []       # (y_fitz, x, skewed)
    for m in _TM_RE.finditer(raw):
        try:
            a, b, c, d, e, f = (float(x) for x in m.groups())
        except ValueError:
            continue
        if abs(a) < 1e-9:
            continue
        ops.append((h - f, e, abs(c / a) >= 0.05))

    by_line: dict[float, list[tuple[float, bool]]] = {}
    for y, x, sk in ops:
        key = round(y, 1)
        by_line.setdefault(key, []).append((x, sk))
    runs = []
    for y, items in by_line.items():
        items.sort()
        for i, (x, sk) in enumerate(items):
            if not sk:
                continue
            end = items[i + 1][0] if i + 1 < len(items) else float('inf')
            runs.append((y, x, end))
    return runs


def _span_style(span, underlines, highlights) -> dict:
    """Bold/italic/colour from the font, underline/highlight from the geometry.

    PDFs carry no style markup: weight and slant have to be read off the font
    name and flags, and an underline is literally a thin line drawn under the
    glyphs, so it is recovered by intersecting text boxes with vector art.
    """
    font = (span.get('font') or '').lower()
    flags = span.get('flags') or 0
    out: dict = {}
    if flags & _FLAG_BOLD or any(k in font for k in ('bold', 'black', 'heavy', 'semib')):
        out['b'] = 1
    if flags & _FLAG_ITALIC or any(k in font for k in ('italic', 'oblique')):
        out['i'] = 1
    col = span.get('color')
    if isinstance(col, int) and col != 0:
        rgb = f'{col & 0xFFFFFF:06X}'
        out['color'] = rgb
        out['color_name'] = color_label(rgb)

    x0, y0, x1, y1 = span.get('bbox', (0, 0, 0, 0))
    w = max(x1 - x0, 1e-6)
    for (ux0, uy0, ux1, uy1) in underlines:
        # a rule sitting just below the baseline and covering most of the span
        if uy0 < y1 + 4.5 and uy1 > y1 - 3.0:
            overlap = min(x1, ux1) - max(x0, ux0)
            if overlap > 0.55 * w:
                out['u'] = 'single'
                break
    for (hx0, hy0, hx1, hy1, hrgb) in highlights:
        if hy0 <= (y0 + y1) / 2 <= hy1:
            overlap = min(x1, hx1) - max(x0, hx0)
            if overlap > 0.5 * w:
                out['hl'] = hrgb
                out['hl_name'] = color_label(hrgb)
                break
    return out


def _page_decorations(page) -> tuple[list, list]:
    """(underline-ish rules, highlight-ish filled boxes) on the page."""
    underlines, highlights = [], []
    try:
        drawings = page.get_drawings() or []
    except Exception:
        return underlines, highlights
    for d in drawings:
        rect = d.get('rect')
        if rect is None:
            continue
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
        h, w = y1 - y0, x1 - x0
        if w <= 2:
            continue
        if h <= 3.0 and w >= 8:                    # thin rule -> underline
            underlines.append((x0, y0, x1, y1))
            continue
        fill = d.get('fill')
        if d.get('type') in ('f', 'fs') and fill and 3 < h < 40:
            r, g, b = (int(round(c * 255)) for c in fill[:3])
            if (r, g, b) == (255, 255, 255) or (r < 40 and g < 40 and b < 40):
                continue
            name, _ = color_info(f'{r:02X}{g:02X}{b:02X}')
            if name in (None, 'white', 'black'):
                continue
            highlights.append((x0, y0, x1, y1, f'{r:02X}{g:02X}{b:02X}'))
    for a in (page.annots() or []):
        if a.type and a.type[1] in ('Highlight', 'Underline'):
            r = a.rect
            col = (a.colors or {}).get('stroke') or (a.colors or {}).get('fill')
            if a.type[1] == 'Underline':
                underlines.append((r.x0, r.y1 - 2, r.x1, r.y1))
            elif col:
                rr, gg, bb = (int(round(c * 255)) for c in col[:3])
                highlights.append((r.x0, r.y0, r.x1, r.y1, f'{rr:02X}{gg:02X}{bb:02X}'))
    return underlines, highlights


def _tags_of(st: dict) -> list[str]:
    bits = []
    if st.get('b'):
        bits.append('bold')
    if st.get('i'):
        bits.append('italic')
    if st.get('u'):
        bits.append('underline')
    if st.get('hl_name'):
        bits.append(f"hl:{st['hl_name']}")
    if st.get('color_name') and st['color_name'] not in ('black', 'gray'):
        bits.append(f"color:{st['color_name']}")
    return bits


def _styled_text(page) -> tuple[str, list[dict]]:
    """Page text with inline formatting tags, plus the list of styled runs.

    Character granularity matters: synthetic italic frequently covers only part
    of a PyMuPDF span, and "which text is bold+underline+italic" needs exactly
    the emphasised phrase, not the whole sentence containing it.
    """
    try:
        data = page.get_text('rawdict')
    except Exception:
        return '', []
    underlines, highlights = _page_decorations(page)
    italic_runs = _italic_runs(page)
    lines_out: list[str] = []
    styled: list[dict] = []
    blocks = sorted((b for b in data.get('blocks', []) if b.get('type') == 0),
                    key=lambda b: (round(b['bbox'][1], 1), round(b['bbox'][0], 1)))
    for b in blocks:
        for line in b.get('lines', []):
            runs: list[list] = []
            for span in line.get('spans', []):
                base = _span_style(span, underlines, highlights)
                for ch in span.get('chars', []):
                    c = ch.get('c') or ''
                    if not c:
                        continue
                    cx0, cy0, cx1, cy1 = ch.get('bbox', (0, 0, 0, 0))
                    mid = (cx0 + cx1) / 2
                    st = dict(base)
                    for (ry, rx0, rx1) in italic_runs:
                        if abs(ry - cy1) <= 3.5 and rx0 - 1.0 <= mid < rx1:
                            st['i'] = 1
                            break
                    tags = _tags_of(st)
                    if runs and runs[-1][0] == tags:
                        runs[-1][1] += c
                    else:
                        runs.append([tags, c])
            parts = []
            for tags, text in runs:
                if tags:
                    parts.append(f"[[{','.join(tags)}]]{text}[[/]]")
                    styled.append({'text': text, 'tags': tags})
                else:
                    parts.append(text)
            if parts:
                lines_out.append(''.join(parts))
    return '\n'.join(lines_out), styled


def _page_text(page) -> str:
    txt = page.get_text('text') or ''
    return txt


def _page_words_layout(page) -> str:
    """Reading-order-ish rendering that keeps columns from interleaving."""
    blocks = page.get_text('blocks') or []
    blocks = [b for b in blocks if len(b) >= 5 and isinstance(b[4], str) and b[4].strip()]
    blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
    return '\n'.join(b[4].strip() for b in blocks)


def extract(path: str, render_dpi: int = 150) -> dict:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, 1):
        styled_txt, styled = _styled_text(page)
        txt = squeeze(styled_txt or _page_words_layout(page) or _page_text(page))
        images = page.get_images(full=True)
        drawings = len(page.get_drawings() or [])
        tables = []
        try:
            for t in (page.find_tables() or []):
                rows = t.extract()
                if rows and any(any(c for c in r) for r in rows):
                    tables.append(rows)
        except Exception:
            pass
        annots = []
        for a in (page.annots() or []):
            info = a.info or {}
            annots.append({'type': a.type[1] if a.type else None,
                           'content': info.get('content'),
                           'title': info.get('title'),
                           'rect': [round(v, 1) for v in a.rect]})
        pages.append({
            'page': i,
            'text': txt,
            'styled': styled,
            'n_chars': len(strip_fmt(txt)),
            'n_images': len(images),
            'n_drawings': drawings,
            'tables': tables,
            'annots': annots,
            'is_image_page': len(strip_fmt(txt)) < 40 and (len(images) > 0 or drawings > 0),
            'size': [round(page.rect.width), round(page.rect.height)],
        })
    meta = dict(doc.metadata or {})
    toc = doc.get_toc() or []
    n = doc.page_count
    doc.close()
    out = {'kind': 'pdf', 'n_pages': n, 'pages': pages, 'metadata': meta, 'toc': toc}
    out['text'] = render(out)
    return out


def render_page_png(path: str, page_no: int, dpi: int = 170) -> bytes:
    """Rasterise one 1-indexed page (for vision on figure-heavy pages)."""
    doc = fitz.open(path)
    page = doc[page_no - 1]
    pix = page.get_pixmap(dpi=dpi)
    data = pix.tobytes('png')
    doc.close()
    return data


def render(doc: dict) -> str:
    lines = []
    if doc.get('toc'):
        lines.append('## 目次(TOC)')
        for lvl, title, pg in doc['toc']:
            lines.append(f"{'  ' * (lvl - 1)}- {title} … p.{pg}")
    for p in doc['pages']:
        lines.append(f"\n## ページ {p['page']} / {doc['n_pages']}")
        if p['is_image_page']:
            lines.append('(このページはテキストがほとんど無い画像ページ。図表はページ画像を参照)')
        if p['text']:
            lines.append(p['text'])
        for t in p['tables']:
            lines.append('[表]')
            for row in t:
                lines.append('| ' + ' | '.join('' if c is None else str(c).replace('\n', ' ')
                                               for c in row) + ' |')
        for a in p['annots']:
            if a.get('content'):
                lines.append(f"[注釈 {a.get('type')} {a.get('title') or ''}] {a['content']}")

    styled = [(p['page'], s) for p in doc['pages'] for s in (p.get('styled') or [])]
    if styled:
        lines.append('\n## 書式が付いている箇所の一覧（PDF）')
        for pg, s in styled[:300]:
            lines.append(f"- p.{pg} [{','.join(s['tags'])}] {s['text']}")
        if len(styled) > 300:
            lines.append(f'  ...ほか {len(styled) - 300} 箇所')
    return squeeze('\n'.join(lines))
