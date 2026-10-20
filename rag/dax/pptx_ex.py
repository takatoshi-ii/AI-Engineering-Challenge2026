"""PowerPoint (.pptx) extraction.

Keeps run-level formatting, shape fill colours, shape geometry (needed to answer
positional questions such as "who sits to the right of X" or to reproduce a
column-wise reading order) plus embedded charts, tables and speaker notes.
"""
from __future__ import annotations

import re
import zipfile

from lxml import etree

from . import chart_ex
from .util import (NS, apply_tint, color_name, q, read_xml, rels_for, squeeze,
                   theme_colors)

EMU_PER_CM = 360000


# ------------------------------------------------------------------- colours
def _solid_color(parent, theme: list[str]) -> str | None:
    """First solid fill colour under *parent*, resolving scheme colours."""
    if parent is None:
        return None
    fill = parent.find(q('a:solidFill'))
    if fill is None:
        return None
    srgb = fill.find(q('a:srgbClr'))
    if srgb is not None and srgb.get('val'):
        return srgb.get('val').upper()
    sch = fill.find(q('a:schemeClr'))
    if sch is not None:
        idx = {'dk1': 0, 'lt1': 1, 'dk2': 2, 'lt2': 3, 'tx1': 0, 'bg1': 1,
               'tx2': 2, 'bg2': 3, 'accent1': 4, 'accent2': 5, 'accent3': 6,
               'accent4': 7, 'accent5': 8, 'accent6': 9, 'hlink': 10,
               'folHlink': 11}.get(sch.get('val'))
        rgb = theme[idx] if idx is not None and idx < len(theme) else None
        if rgb:
            lum = sch.find(q('a:lumMod'))
            off = sch.find(q('a:lumOff'))
            if off is not None and off.get('val'):
                rgb = apply_tint(rgb, int(off.get('val')) / 100000.0)
            elif lum is not None and lum.get('val'):
                rgb = apply_tint(rgb, -(1 - int(lum.get('val')) / 100000.0))
            return rgb
    return None


def _run_props(rPr, theme) -> dict:
    p: dict = {}
    if rPr is None:
        return p
    if rPr.get('b') in ('1', 'true'):
        p['b'] = 1
    if rPr.get('i') in ('1', 'true'):
        p['i'] = 1
    u = rPr.get('u')
    if u and u != 'none':
        p['u'] = u
    if rPr.get('strike') and rPr.get('strike') != 'noStrike':
        p['strike'] = 1
    if rPr.get('sz'):
        try:
            p['sz'] = int(rPr.get('sz')) / 100
        except ValueError:
            pass
    c = _solid_color(rPr, theme)
    if c:
        p['color'] = c
        p['color_name'] = color_name(c)
    hl = rPr.find(q('a:highlight'))
    if hl is not None:
        srgb = hl.find(q('a:srgbClr'))
        if srgb is not None and srgb.get('val'):
            p['hl'] = srgb.get('val').upper()
            p['hl_name'] = color_name(p['hl'])
    return p


def _merge(runs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in runs:
        if not r.get('t'):
            continue
        key = {k: v for k, v in r.items() if k != 't'}
        if out and {k: v for k, v in out[-1].items() if k != 't'} == key:
            out[-1]['t'] += r['t']
        else:
            out.append(dict(r))
    return out


def _paragraphs(txBody, theme) -> list[dict]:
    paras = []
    if txBody is None:
        return paras
    for p in txBody.findall(q('a:p')):
        runs = []
        for child in p:
            local = etree.QName(child).localname
            if local == 'r':
                t = child.find(q('a:t'))
                props = _run_props(child.find(q('a:rPr')), theme)
                props['t'] = t.text or '' if t is not None else ''
                runs.append(props)
            elif local == 'br':
                runs.append({'t': '\n'})
            elif local == 'fld':
                t = child.find(q('a:t'))
                if t is not None and t.text:
                    props = _run_props(child.find(q('a:rPr')), theme)
                    props['t'] = t.text
                    props['field'] = child.get('type')
                    runs.append(props)
        runs = _merge(runs)
        pPr = p.find(q('a:pPr'))
        lvl = int(pPr.get('lvl')) if pPr is not None and pPr.get('lvl') else 0
        bullet = pPr is not None and pPr.find(q('a:buNone')) is None
        paras.append({'text': ''.join(r['t'] for r in runs), 'runs': runs,
                      'level': lvl, 'bullet': bool(bullet)})
    return paras


# -------------------------------------------------------------------- shapes
def _xfrm(sp) -> dict | None:
    x = sp.find(f".//{q('a:xfrm')}")
    if x is None:
        return None
    off, ext = x.find(q('a:off')), x.find(q('a:ext'))
    if off is None or ext is None:
        return None
    try:
        return {
            'x': round(int(off.get('x')) / EMU_PER_CM, 2),
            'y': round(int(off.get('y')) / EMU_PER_CM, 2),
            'w': round(int(ext.get('cx')) / EMU_PER_CM, 2),
            'h': round(int(ext.get('cy')) / EMU_PER_CM, 2),
        }
    except (TypeError, ValueError):
        return None


def _table(graphic, theme) -> dict:
    tbl = graphic.find(f".//{q('a:tbl')}")
    rows = []
    for tr in tbl.findall(q('a:tr')):
        cells = []
        for tc in tr.findall(q('a:tc')):
            paras = _paragraphs(tc.find(q('a:txBody')), theme)
            cell = {'text': '\n'.join(p['text'] for p in paras).strip(),
                    'runs': [r for p in paras for r in p['runs']]}
            fill = _solid_color(tc.find(q('a:tcPr')), theme)
            if fill:
                cell['fill'] = fill
                cell['fill_name'] = color_name(fill)
            if tc.get('gridSpan'):
                cell['gridSpan'] = int(tc.get('gridSpan'))
            if tc.get('hMerge') or tc.get('vMerge'):
                cell['merged'] = True
            cells.append(cell)
        rows.append(cells)
    return {'type': 'table', 'rows': rows}


def _walk_shapes(tree, theme, zf, rels, out: list, depth=0):
    for sp in tree:
        local = etree.QName(sp).localname
        if local == 'grpSp':
            _walk_shapes(sp, theme, zf, rels, out, depth + 1)
            continue
        if local == 'sp':
            nv = sp.find(f"{q('p:nvSpPr')}/{q('p:cNvPr')}")
            paras = _paragraphs(sp.find(q('p:txBody')), theme)
            item = {
                'type': 'shape',
                'name': nv.get('name') if nv is not None else None,
                'pos': _xfrm(sp),
                'paragraphs': paras,
                'text': '\n'.join(p['text'] for p in paras).strip(),
            }
            fill = _solid_color(sp.find(q('p:spPr')), theme)
            if fill:
                item['fill'] = fill
                item['fill_name'] = color_name(fill)
            ph = sp.find(f"{q('p:nvSpPr')}/{q('p:nvPr')}/{q('p:ph')}")
            if ph is not None:
                item['placeholder'] = ph.get('type') or 'body'
            out.append(item)
        elif local == 'pic':
            nv = sp.find(f"{q('p:nvPicPr')}/{q('p:cNvPr')}")
            blip = sp.find(f".//{q('a:blip')}")
            rid = blip.get(q('r:embed')) if blip is not None else None
            out.append({'type': 'picture',
                        'name': nv.get('name') if nv is not None else None,
                        'pos': _xfrm(sp),
                        'media': rels.get(rid, (None, None))[0] if rid else None})
        elif local == 'graphicFrame':
            nv = sp.find(f"{q('p:nvGraphicFramePr')}/{q('p:cNvPr')}")
            if sp.find(f".//{q('a:tbl')}") is not None:
                t = _table(sp, theme)
                t['name'] = nv.get('name') if nv is not None else None
                t['pos'] = _xfrm(sp)
                out.append(t)
            else:
                ref = sp.find(f".//{q('c:chart')}")
                if ref is not None:
                    rid = ref.get(q('r:id'))
                    part = rels.get(rid, (None, None))[0]
                    info = chart_ex.load_chart_part(zf, part) if part else None
                    out.append({'type': 'chart',
                                'name': nv.get('name') if nv is not None else None,
                                'pos': _xfrm(sp), 'chart': info})
        elif local == 'cxnSp':
            continue


def _slide(zf, part, theme) -> dict:
    root = read_xml(zf, part)
    rels = {k: v for k, v in rels_for(zf, part).items()}
    rels = {k: (v[0], v[1]) for k, v in rels.items()}
    spTree = root.find(f"{q('p:cSld')}/{q('p:spTree')}")
    shapes: list = []
    if spTree is not None:
        _walk_shapes(spTree, theme, zf, {k: v for k, v in rels.items()}, shapes)

    notes_part = next((t for _, (t, typ) in rels.items() if typ == 'notesSlide'), None)
    notes = ''
    if notes_part:
        nroot = read_xml(zf, notes_part)
        if nroot is not None:
            texts = []
            for sp in nroot.iter(q('p:sp')):
                ph = sp.find(f".//{q('p:ph')}")
                if ph is not None and ph.get('type') == 'sldNum':
                    continue
                texts += [p['text'] for p in _paragraphs(sp.find(q('p:txBody')), theme)]
            notes = squeeze('\n'.join(texts))
    return {'part': part, 'shapes': shapes, 'notes': notes}


def _slide_order(zf) -> list[str]:
    pres = read_xml(zf, 'ppt/presentation.xml')
    rels = rels_for(zf, 'ppt/presentation.xml')
    order = []
    if pres is not None:
        lst = pres.find(q('p:sldIdLst'))
        if lst is not None:
            for sld in lst:
                rid = sld.get(q('r:id'))
                if rid in rels:
                    order.append(rels[rid][0])
    if not order:
        order = sorted((n for n in zf.namelist()
                        if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)),
                       key=lambda n: int(re.search(r'(\d+)', n).group(1)))
    return order


# -------------------------------------------------------------------- render
def _fmt_tag(r: dict) -> str:
    bits = []
    for k, lab in (('b', 'bold'), ('i', 'italic'), ('u', 'underline'), ('strike', 'strike')):
        if r.get(k):
            bits.append(lab)
    if r.get('hl_name'):
        bits.append(f"hl:{r['hl_name']}")
    if r.get('color_name') and r['color_name'] not in ('black', 'white', 'gray'):
        bits.append(f"color:{r['color_name']}")
    return ','.join(bits)


def _render_runs(runs) -> str:
    parts = []
    for r in runs:
        tag = _fmt_tag(r)
        parts.append(f'[[{tag}]]{r["t"]}[[/]]' if tag else r['t'])
    return ''.join(parts)


def render(doc: dict) -> str:
    lines: list[str] = []
    for i, s in enumerate(doc['slides'], 1):
        lines.append(f'\n## スライド {i}')
        for sh in s['shapes']:
            pos = sh.get('pos')
            ptxt = (f" @({pos['x']},{pos['y']} {pos['w']}x{pos['h']}cm)" if pos else '')
            if sh['type'] == 'shape':
                if not sh['text'].strip() and not sh.get('fill_name'):
                    continue
                head = f"[図形 {sh.get('name') or ''}{ptxt}"
                if sh.get('fill_name'):
                    head += f" 塗り={sh['fill_name']}({sh.get('fill')})"
                head += ']'
                lines.append(head)
                for p in sh['paragraphs']:
                    if not p['text'].strip():
                        continue
                    lines.append('  ' * (p['level'] + 1) + ('- ' if p['bullet'] else '') +
                                 _render_runs(p['runs']))
            elif sh['type'] == 'table':
                lines.append(f"[表 {sh.get('name') or ''}{ptxt}]")
                for row in sh['rows']:
                    cells = []
                    for c in row:
                        t = _render_runs(c['runs']) or c['text']
                        t = t.replace('\n', ' / ')
                        if c.get('fill_name'):
                            t = f"[[cell:{c['fill_name']}]]{t}"
                        cells.append(t)
                    lines.append('| ' + ' | '.join(cells) + ' |')
            elif sh['type'] == 'chart':
                lines.append(f"[グラフ {sh.get('name') or ''}{ptxt}]")
                if sh.get('chart'):
                    lines.append(chart_ex.render_chart(sh['chart'], sh.get('name') or ''))
            elif sh['type'] == 'picture':
                lines.append(f"[画像 {sh.get('name') or ''}{ptxt} media={sh.get('media')}]")
        if s.get('notes'):
            lines.append(f"[スピーカーノート] {s['notes']}")
    return squeeze('\n'.join(lines))


def extract(path: str, password: str | None = None) -> dict:
    src = path
    if password:
        import io
        import msoffcrypto
        buf = io.BytesIO()
        with open(path, 'rb') as fh:
            off = msoffcrypto.OfficeFile(fh)
            off.load_key(password=password)
            off.decrypt(buf)
        buf.seek(0)
        src = buf
    zf = zipfile.ZipFile(src)
    theme = theme_colors(zf)
    slides = [_slide(zf, p, theme) for p in _slide_order(zf)]
    media = [n for n in zf.namelist() if n.startswith('ppt/media/')]
    zf.close()
    doc = {'kind': 'pptx', 'slides': slides, 'media': media}
    doc['text'] = render(doc)
    return doc
