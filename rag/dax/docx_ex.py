"""Word (.docx) extraction preserving run-level formatting, tables and comments.

Works directly on the OOXML so that features python-docx hides (comments,
text boxes, shading, theme colours, field codes) are still recovered.
"""
from __future__ import annotations

import re
import zipfile

from lxml import etree

from . import chart_ex
from .util import (NS, apply_tint, color_label, color_name, highlight_to_rgb,
                   nfc, q, read_xml, rels_for, squeeze, theme_colors)


# --------------------------------------------------------------------- runs
def _run_props(rPr, theme: list[str]) -> dict:
    p: dict = {}
    if rPr is None:
        return p

    def on(tag: str) -> bool:
        el = rPr.find(q(tag))
        return el is not None and el.get(q('w:val')) not in ('0', 'false', 'none')

    if on('w:b') or on('w:bCs'):
        p['b'] = 1
    if on('w:i') or on('w:iCs'):
        p['i'] = 1
    if on('w:strike'):
        p['strike'] = 1
    u = rPr.find(q('w:u'))
    if u is not None and (u.get(q('w:val')) or 'single') != 'none':
        p['u'] = u.get(q('w:val')) or 'single'
    va = rPr.find(q('w:vertAlign'))
    if va is not None:
        p['va'] = va.get(q('w:val'))

    col = rPr.find(q('w:color'))
    if col is not None:
        v = (col.get(q('w:val')) or '').upper()
        if v and v != 'AUTO':
            p['color'] = v
        elif col.get(q('w:themeColor')):
            idx = {'dark1': 0, 'light1': 1, 'dark2': 2, 'light2': 3, 'accent1': 4,
                   'accent2': 5, 'accent3': 6, 'accent4': 7, 'accent5': 8,
                   'accent6': 9, 'hyperlink': 10}.get(col.get(q('w:themeColor')))
            if idx is not None and idx < len(theme) and theme[idx]:
                tint = col.get(q('w:themeTint')) or col.get(q('w:themeShade'))
                rgb = theme[idx]
                if tint:
                    t = int(tint, 16) / 255.0
                    rgb = apply_tint(rgb, (1 - t) if col.get(q('w:themeTint')) else -(1 - t))
                p['color'] = rgb
    if 'color' in p:
        p['color_name'] = color_name(p['color'])

    hl = rPr.find(q('w:highlight'))
    if hl is not None:
        rgb = highlight_to_rgb(hl.get(q('w:val')))
        if rgb:
            p['hl'] = rgb
            p['hl_name'] = color_name(rgb)
    shd = rPr.find(q('w:shd'))
    if shd is not None:
        fill = (shd.get(q('w:fill')) or '').upper()
        if fill and fill not in ('AUTO', 'FFFFFF'):
            p.setdefault('hl', fill)
            p.setdefault('hl_name', color_name(fill))

    sz = rPr.find(q('w:sz'))
    if sz is not None and sz.get(q('w:val')):
        try:
            p['sz'] = int(sz.get(q('w:val'))) / 2
        except ValueError:
            pass
    return p


def _run_text(r) -> str:
    out = []
    for el in r.iter():
        tag = etree.QName(el).localname
        if tag == 't':
            out.append(el.text or '')
        elif tag == 'tab':
            out.append('\t')
        elif tag in ('br', 'cr'):
            out.append('\n')
        elif tag == 'noBreakHyphen':
            out.append('-')
    return ''.join(out)


def _merge_runs(runs: list[dict]) -> list[dict]:
    """Join neighbouring runs that carry identical formatting (Word splits text
    on spell-check and rsid boundaries, which fragments phrases)."""
    merged: list[dict] = []
    for r in runs:
        if not r['t']:
            continue
        key = {k: v for k, v in r.items() if k != 't'}
        if merged and {k: v for k, v in merged[-1].items() if k != 't'} == key:
            merged[-1]['t'] += r['t']
        else:
            merged.append(dict(r))
    return merged


# --------------------------------------------------------------- paragraphs
def _para(el, theme, comment_map) -> dict:
    pPr = el.find(q('w:pPr'))
    style = None
    if pPr is not None:
        ps = pPr.find(q('w:pStyle'))
        if ps is not None:
            style = ps.get(q('w:val'))
    p_rpr = pPr.find(q('w:rPr')) if pPr is not None else None
    base = _run_props(p_rpr, theme)

    runs: list[dict] = []
    active: list[str] = []          # comment ids covering the current position
    for node in el.iter():
        tag = etree.QName(node).localname
        if tag == 'commentRangeStart':
            active.append(node.get(q('w:id')))
        elif tag == 'commentRangeEnd':
            cid = node.get(q('w:id'))
            if cid in active:
                active.remove(cid)
        elif tag == 'r' and node.getparent() is not None:
            t = _run_text(node)
            if not t:
                continue
            props = dict(base)
            props.update(_run_props(node.find(q('w:rPr')), theme))
            props['t'] = t
            if active:
                props['cmt'] = list(active)
            runs.append(props)

    runs = _merge_runs(runs)
    text = ''.join(r['t'] for r in runs)
    out = {'type': 'paragraph', 'text': text, 'runs': runs}
    if style:
        out['style'] = style
    if pPr is not None:
        num = pPr.find(q('w:numPr'))
        if num is not None:
            out['list'] = True
    return out


def _table(el, theme, comment_map) -> dict:
    rows = []
    for tr in el.findall(q('w:tr')):
        cells = []
        for tc in tr.findall(q('w:tc')):
            paras = [_para(p, theme, comment_map) for p in tc.findall(q('w:p'))]
            shd = tc.find(f"{q('w:tcPr')}/{q('w:shd')}")
            fill = (shd.get(q('w:fill')) or '').upper() if shd is not None else ''
            cell = {'text': '\n'.join(p['text'] for p in paras).strip(),
                    'runs': [r for p in paras for r in p['runs']]}
            if fill and fill not in ('AUTO', 'FFFFFF'):
                cell['fill'] = fill
                cell['fill_name'] = color_name(fill)
            cells.append(cell)
        rows.append(cells)
    return {'type': 'table', 'rows': rows}


def _body_blocks(body, theme, comment_map) -> list[dict]:
    blocks = []
    for el in body:
        tag = etree.QName(el).localname
        if tag == 'p':
            blocks.append(_para(el, theme, comment_map))
            # text boxes live inside runs; surface their paragraphs too
            for tb in el.iter(q('w:txbxContent')):
                for tp in tb.findall(q('w:p')):
                    b = _para(tp, theme, comment_map)
                    b['in_textbox'] = True
                    blocks.append(b)
        elif tag == 'tbl':
            blocks.append(_table(el, theme, comment_map))
        elif tag == 'sdt':
            content = el.find(q('w:sdtContent'))
            if content is not None:
                blocks.extend(_body_blocks(content, theme, comment_map))
    return blocks


# ----------------------------------------------------------------- comments
def _charts(zf, root) -> list[dict]:
    """Charts embedded in the document body, in document order."""
    if root is None:
        return []
    rels = rels_for(zf, 'word/document.xml')
    out = []
    for i, ref in enumerate(root.iter(q('c:chart')), 1):
        rid = ref.get(q('r:id'))
        part = rels.get(rid, (None, None))[0] if rid else None
        if not part:
            continue
        info = chart_ex.load_chart_part(zf, part)
        if info:
            info['ordinal'] = i
            out.append(info)
    return out


def _comments(zf, theme) -> list[dict]:
    root = read_xml(zf, 'word/comments.xml')
    if root is None:
        return []
    out = []
    for c in root.findall(q('w:comment')):
        paras = [_para(p, theme, {}) for p in c.findall(q('w:p'))]
        out.append({
            'id': c.get(q('w:id')),
            'author': c.get(q('w:author')),
            'date': c.get(q('w:date')),
            'text': '\n'.join(p['text'] for p in paras).strip(),
        })
    return out


# -------------------------------------------------------------------- render
def _fmt_tag(r: dict) -> str:
    bits = []
    if r.get('b'):
        bits.append('bold')
    if r.get('i'):
        bits.append('italic')
    if r.get('u'):
        bits.append('underline')
    if r.get('strike'):
        bits.append('strike')
    if r.get('hl_name'):
        bits.append(f"hl:{r['hl_name']}")
    if r.get('color_name') and r.get('color_name') not in ('black',):
        bits.append(f"color:{r['color_name']}")
    if r.get('cmt'):
        bits.append('comment:' + '/'.join(r['cmt']))
    return ','.join(bits)


def commented_spans(doc: dict) -> list[dict]:
    """Text ranges a Word comment is attached to, paired with the comment."""
    cmap = {c['id']: c for c in doc.get('comments', [])}
    spans: list[dict] = []

    def scan(runs, where):
        cur_ids, buf = None, ''
        for r in runs:
            ids = tuple(r.get('cmt') or ())
            if ids != cur_ids:
                if cur_ids and buf:
                    spans.append({'ids': list(cur_ids), 'text': buf, 'where': where})
                cur_ids, buf = ids, (r['t'] if ids else '')
            elif ids:
                buf += r['t']
        if cur_ids and buf:
            spans.append({'ids': list(cur_ids), 'text': buf, 'where': where})

    for b in doc['blocks']:
        if b['type'] == 'paragraph':
            scan(b['runs'], '本文')
        else:
            for ri, row in enumerate(b['rows']):
                for ci, c in enumerate(row):
                    scan(c.get('runs') or [], f'表 行{ri + 1} 列{ci + 1}')
    for s in spans:
        s['comments'] = [{'id': i, 'author': cmap.get(i, {}).get('author'),
                          'text': cmap.get(i, {}).get('text')} for i in s['ids']]
    return spans


def render(doc: dict) -> str:
    """Markdown-ish rendering that keeps formatting visible as inline tags."""
    lines: list[str] = []
    for b in doc['blocks']:
        if b['type'] == 'paragraph':
            if not b['text'].strip():
                continue
            parts = []
            for r in b['runs']:
                tag = _fmt_tag(r)
                parts.append(f'[[{tag}]]{r["t"]}[[/]]' if tag else r['t'])
            prefix = ''
            st = (b.get('style') or '')
            m = re.search(r'(\d+)$', st)
            if st.lower().startswith('heading') or st.startswith('見出し'):
                prefix = '#' * (int(m.group(1)) if m else 1) + ' '
            elif b.get('list'):
                prefix = '- '
            if b.get('in_textbox'):
                prefix = '(textbox) ' + prefix
            lines.append(prefix + ''.join(parts))
        else:
            for row in b['rows']:
                cells = []
                for c in row:
                    t = c['text'].replace('\n', ' / ')
                    if c.get('fill_name'):
                        t = f"[[cell:{c['fill_name']}]]{t}"
                    cells.append(t)
                lines.append('| ' + ' | '.join(cells) + ' |')
            lines.append('')
    if doc.get('comments'):
        lines.append('')
        lines.append('## コメント (Word comments)')
        for c in doc['comments']:
            lines.append(f"- [{c['id']}] {c.get('author') or ''}: {c['text']}")
        spans = commented_spans(doc)
        if spans:
            lines.append('')
            lines.append('### コメントが付いている箇所（コメント対象の本文）')
            for s in spans:
                who = ', '.join(f"[{c['id']}]{c.get('author') or ''}:{c.get('text') or ''}"
                                for c in s['comments'])
                lines.append(f"- 対象テキスト「{s['text']}」（{s['where']}） ← {who}")
    if doc.get('charts'):
        lines.append('')
        lines.append('## 埋め込みグラフ（文書内の出現順）')
        for info in doc['charts']:
            lines.append(chart_ex.render_chart(info, f"{info.get('ordinal')}"))
    if doc.get('headers'):
        lines.append('')
        lines.append('## ヘッダー/フッター')
        for h in doc['headers']:
            lines.append(f'- {h}')
    return squeeze('\n'.join(lines))


# ------------------------------------------------------------------ entry pt
def extract(path: str, password: str | None = None) -> dict:
    data = None
    if password:
        import io
        import msoffcrypto
        buf = io.BytesIO()
        with open(path, 'rb') as fh:
            off = msoffcrypto.OfficeFile(fh)
            off.load_key(password=password)
            off.decrypt(buf)
        buf.seek(0)
        data = buf

    zf = zipfile.ZipFile(data or path)
    theme = theme_colors(zf)
    comments = _comments(zf, theme)
    cmap = {c['id']: c for c in comments}

    root = read_xml(zf, 'word/document.xml')
    body = root.find(q('w:body')) if root is not None else None
    blocks = _body_blocks(body, theme, cmap) if body is not None else []

    headers = []
    for name in zf.namelist():
        if re.fullmatch(r'word/(header|footer)\d*\.xml', name):
            hroot = read_xml(zf, name)
            if hroot is None:
                continue
            txt = squeeze(''.join(hroot.itertext()))
            if txt:
                headers.append(f'{name}: {txt}')

    images = [n for n in zf.namelist() if n.startswith('word/media/')]
    charts = _charts(zf, root)
    zf.close()

    doc = {'kind': 'docx', 'blocks': blocks, 'comments': comments,
           'headers': headers, 'images': images, 'charts': charts}
    doc['text'] = render(doc)
    return doc
