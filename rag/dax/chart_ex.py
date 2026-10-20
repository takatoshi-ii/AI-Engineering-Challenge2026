"""DrawingML chart extraction.

Office stores every plotted point in the chart part's numeric/string caches, so
series names, categories, values, axis scales and series colours can be read
exactly rather than estimated from a rendered picture.
"""
from __future__ import annotations

import zipfile

from lxml import etree

from .util import NS, color_name, q, read_xml, rels_for, squeeze, theme_colors

_CHART_TYPES = ('barChart', 'bar3DChart', 'lineChart', 'line3DChart', 'pieChart',
                'pie3DChart', 'doughnutChart', 'scatterChart', 'areaChart',
                'area3DChart', 'radarChart', 'bubbleChart', 'ofPieChart',
                'stockChart', 'surfaceChart')


def _cache(el) -> list:
    """Values from a c:numCache / c:strCache, ordered by c:pt/@idx."""
    if el is None:
        return []
    pts: dict[int, str] = {}
    for cache in (el.find(q('c:numCache')), el.find(q('c:strCache')),
                  el.find(q('c:numLit')), el.find(q('c:strLit'))):
        if cache is None:
            continue
        for pt in cache.findall(q('c:pt')):
            try:
                idx = int(pt.get('idx'))
            except (TypeError, ValueError):
                continue
            v = pt.find(q('c:v'))
            pts[idx] = v.text if v is not None else None
    if not pts:
        return []
    return [pts.get(i) for i in range(max(pts) + 1)]


def _ref(el) -> str | None:
    if el is None:
        return None
    for tag in ('c:numRef', 'c:strRef', 'c:multiLvlStrRef'):
        r = el.find(q(tag))
        if r is not None:
            f = r.find(q('c:f'))
            if f is not None:
                return f.text
    return None


_SCHEME_INDEX = {'dk1': 0, 'lt1': 1, 'dk2': 2, 'lt2': 3, 'tx1': 0, 'bg1': 1,
                 'tx2': 2, 'bg2': 3, 'accent1': 4, 'accent2': 5, 'accent3': 6,
                 'accent4': 7, 'accent5': 8, 'accent6': 9, 'hlink': 10,
                 'folHlink': 11}


def _series_color(ser, theme: list[str] | None = None) -> tuple[str | None, str | None]:
    """(hex, colour word) of the series' line or fill colour.

    Series usually carry a theme colour rather than a literal RGB, so the
    palette is resolved here — questions refer to "the blue line", not
    "the accent1 line".
    """
    for path in (f"{q('c:spPr')}/{q('a:ln')}/{q('a:solidFill')}/{q('a:srgbClr')}",
                 f"{q('c:spPr')}/{q('a:solidFill')}/{q('a:srgbClr')}"):
        el = ser.find(path)
        if el is not None and el.get('val'):
            v = el.get('val').upper()
            return v, color_name(v)
    for path in (f"{q('c:spPr')}/{q('a:ln')}/{q('a:solidFill')}/{q('a:schemeClr')}",
                 f"{q('c:spPr')}/{q('a:solidFill')}/{q('a:schemeClr')}"):
        el = ser.find(path)
        if el is not None and el.get('val'):
            name = el.get('val')
            idx = _SCHEME_INDEX.get(name)
            if theme and idx is not None and idx < len(theme) and theme[idx]:
                rgb = theme[idx]
                return rgb, color_name(rgb)
            return None, f'scheme:{name}'
    return None, None


def _axis(ax) -> dict:
    out: dict = {}
    t = ax.find(f"{q('c:title')}")
    if t is not None:
        txt = squeeze(''.join(t.itertext()))
        if txt:
            out['title'] = txt
    sc = ax.find(q('c:scaling'))
    if sc is not None:
        for k, tag in (('min', 'c:min'), ('max', 'c:max')):
            el = sc.find(q(tag))
            if el is not None and el.get('val'):
                out[k] = float(el.get('val'))
        o = sc.find(q('c:orientation'))
        if o is not None:
            out['orientation'] = o.get('val')
    for k, tag in (('major_unit', 'c:majorUnit'), ('minor_unit', 'c:minorUnit')):
        el = ax.find(q(tag))
        if el is not None and el.get('val'):
            out[k] = float(el.get('val'))
    fmt = ax.find(q('c:numFmt'))
    if fmt is not None and fmt.get('formatCode'):
        out['fmt'] = fmt.get('formatCode')
    d = ax.find(q('c:delete'))
    if d is not None and d.get('val') in ('1', 'true'):
        out['hidden'] = True
    return out


def parse_chart(root, theme: list[str] | None = None) -> dict:
    """Structured content of one c:chartSpace element."""
    info: dict = {'series': [], 'axes': []}
    title = root.find(f".//{q('c:chart')}/{q('c:title')}")
    if title is not None:
        t = squeeze(''.join(title.itertext()))
        if t:
            info['title'] = t

    kinds = []
    for el in root.iter():
        name = etree.QName(el).localname
        if name in _CHART_TYPES:
            kinds.append(name)
    info['chart_types'] = sorted(set(kinds))

    for ser in root.iter(q('c:ser')):
        plot = etree.QName(ser.getparent()).localname
        name_el = ser.find(q('c:tx'))
        cached = _cache(name_el.find(q('c:strRef')) if name_el is not None else None)
        name = None
        if name_el is not None:
            sr = name_el.find(q('c:strRef'))
            if sr is not None:
                name = (_cache(sr) or [None])[0]
            else:
                v = name_el.find(q('c:v'))
                name = v.text if v is not None else None
        hexc, cname = _series_color(ser, theme)
        cat_el = ser.find(q('c:cat')) if ser.find(q('c:cat')) is not None else ser.find(q('c:xVal'))
        val_el = ser.find(q('c:val')) if ser.find(q('c:val')) is not None else ser.find(q('c:yVal'))
        s = {
            'plot': plot,
            'name': name,
            'name_ref': _ref(name_el),
            'cat_ref': _ref(cat_el),
            'val_ref': _ref(val_el),
            'categories': _cache(cat_el.find(q('c:numRef')) if cat_el is not None and cat_el.find(q('c:numRef')) is not None
                                 else (cat_el.find(q('c:strRef')) if cat_el is not None else None)) or _cache(cat_el),
            'values': _cache(val_el.find(q('c:numRef')) if val_el is not None and val_el.find(q('c:numRef')) is not None
                             else val_el) if val_el is not None else [],
        }
        if hexc:
            s['color'] = hexc
        if cname:
            s['color_name'] = cname
        info['series'].append(s)

    for ax_tag in ('c:valAx', 'c:catAx', 'c:dateAx', 'c:serAx'):
        for ax in root.iter(q(ax_tag)):
            a = _axis(ax)
            a['kind'] = ax_tag.split(':')[1]
            info['axes'].append(a)
    return info


def load_chart_part(zf: zipfile.ZipFile, part: str) -> dict | None:
    root = read_xml(zf, part)
    if root is None:
        return None
    # a chart may override the package theme with its own colour scheme
    theme = None
    for _, (tgt, typ) in rels_for(zf, part).items():
        if typ == 'themeOverride':
            troot = read_xml(zf, tgt)
            if troot is not None:
                theme = _palette(troot)
    if theme is None:
        theme = theme_colors(zf)
    info = parse_chart(root, theme)
    info['part'] = part
    return info


def _palette(root) -> list[str]:
    """Theme palette from a themeOverride part (same order as util.theme_colors)."""
    scheme = root.find('.//' + q('a:clrScheme'))
    if scheme is None:
        return []
    order = ['dk1', 'lt1', 'dk2', 'lt2', 'accent1', 'accent2', 'accent3',
             'accent4', 'accent5', 'accent6', 'hlink', 'folHlink']
    found: dict[str, str] = {}
    for el in scheme:
        local = etree.QName(el).localname
        srgb = el.find(q('a:srgbClr'))
        sysc = el.find(q('a:sysClr'))
        if srgb is not None:
            found[local] = (srgb.get('val') or '').upper()
        elif sysc is not None:
            found[local] = (sysc.get('lastClr') or '').upper()
    pal = [found.get(k, '') for k in order]
    if len(pal) >= 4:
        pal[0], pal[1] = pal[1], pal[0]
        pal[2], pal[3] = pal[3], pal[2]
    return pal


# --------------------------------------------------------------- chartEx
# Histograms, box plots, waterfalls and friends use the 2014 "chartex" part,
# a different namespace whose data lives in a shared <cx:chartData> block.
def parse_chartex(root) -> dict:
    info: dict = {'series': [], 'axes': [], 'is_chartex': True}
    title = root.find(f"{q('cx:chart')}/{q('cx:title')}")
    if title is not None:
        t = squeeze(''.join(title.itertext()))
        if t:
            info['title'] = t

    data: dict[str, dict] = {}
    for d in root.iter(q('cx:data')):
        entry: dict = {'dims': []}
        for dim in d:
            local = etree.QName(dim).localname
            if local not in ('numDim', 'strDim'):
                continue
            f = dim.find(q('cx:f'))
            pts: list = []
            lvl = dim.find(q('cx:lvl'))
            if lvl is not None:
                for pt in lvl.findall(q('cx:pt')):
                    pts.append(pt.text)
            entry['dims'].append({'kind': local, 'type': dim.get('type'),
                                  'ref': f.text if f is not None else None,
                                  'values': pts})
        data[d.get('id')] = entry

    for ser in root.iter(q('cx:series')):
        name, name_ref = None, None
        tx = ser.find(f"{q('cx:tx')}/{q('cx:txData')}")
        if tx is not None:
            v = tx.find(q('cx:v'))
            f = tx.find(q('cx:f'))
            name = v.text if v is not None else None
            name_ref = f.text if f is not None else None
        did = ser.find(q('cx:dataId'))
        entry = data.get(did.get('val')) if did is not None else None
        s: dict = {'plot': ser.get('layoutId'), 'name': name, 'name_ref': name_ref,
                   'categories': [], 'values': []}
        if entry:
            s['data_refs'] = [{'type': d['type'], 'ref': d['ref']} for d in entry['dims']]
            for d in entry['dims']:
                if d['type'] == 'val' and d['values']:
                    s['values'] = d['values']
                elif d['type'] == 'cat' and d['values']:
                    s['categories'] = d['values']
        binning = ser.find(f"{q('cx:layoutPr')}/{q('cx:binning')}")
        if binning is not None:
            b: dict = {'interval_closed': binning.get('intervalClosed')}
            for tag, key in (('cx:binCount', 'bin_count'), ('cx:binSize', 'bin_size'),
                             ('cx:underflow', 'underflow'), ('cx:overflow', 'overflow')):
                el = binning.find(q(tag))
                if el is not None:
                    b[key] = el.get('val')
            if binning.find(q('cx:binCount')) is None and binning.find(q('cx:binSize')) is None:
                b['mode'] = 'automatic'
            s['binning'] = b
        stat = ser.find(f"{q('cx:layoutPr')}/{q('cx:statistics')}")
        if stat is not None:
            s['statistics'] = stat.get('quartileMethod')
        info['series'].append(s)

    for ax in root.iter(q('cx:axis')):
        a: dict = {'kind': 'catScaling' if ax.find(q('cx:catScaling')) is not None
                   else ('valScaling' if ax.find(q('cx:valScaling')) is not None else 'axis'),
                   'id': ax.get('id')}
        t = ax.find(q('cx:title'))
        if t is not None:
            txt = squeeze(''.join(t.itertext()))
            if txt:
                a['title'] = txt
        for scal in (ax.find(q('cx:valScaling')), ax.find(q('cx:catScaling'))):
            if scal is None:
                continue
            for k in ('min', 'max', 'majorUnit', 'minorUnit', 'gapWidth'):
                if scal.get(k):
                    a[k] = scal.get(k)
        info['axes'].append(a)
    return info


def render_chartex(info: dict, label: str = '') -> str:
    kinds = sorted({s.get('plot') for s in info.get('series', []) if s.get('plot')})
    title = info.get('title') or ''
    head = title if title else str(label)
    lines = [f"### グラフ {head} (拡張グラフ: {'/'.join(kinds)})".rstrip()]
    for a in info.get('axes', []):
        bits = [f'{k}={v}' for k, v in a.items() if k != 'id']
        if bits:
            lines.append(f"- 軸[{a.get('id')}]: " + ', '.join(bits))
    for i, s in enumerate(info.get('series', []), 1):
        head = f"- 系列{i} 種別={s.get('plot')}"
        if s.get('name'):
            head += f" 対象データ(列名)='{s['name']}'"
        if s.get('name_ref'):
            head += f" 名前参照={s['name_ref']}"
        lines.append(head)
        for ref in s.get('data_refs', []):
            lines.append(f"  データ参照[{ref['type']}]: {ref['ref']}")
        if s.get('binning'):
            b = s['binning']
            desc = ', '.join(f'{k}={v}' for k, v in b.items() if v is not None)
            lines.append(f"  ビン設定: {desc}"
                         + ('（自動ビン。度数はグラフXMLに保持されないため、'
                            '元データから計算すること）' if b.get('mode') == 'automatic' else ''))
        if s.get('values'):
            lines.append('  値: ' + ', '.join(map(str, s['values'][:200])))
    return '\n'.join(lines)


def load_chartex_part(zf: zipfile.ZipFile, part: str) -> dict | None:
    root = read_xml(zf, part)
    if root is None:
        return None
    info = parse_chartex(root)
    info['part'] = part
    return info


def render_chart(info: dict, label: str = '') -> str:
    if info.get('is_chartex'):
        return render_chartex(info, label)
    """Compact text rendering; series values are printed in full because the
    questions ask for exact values at specific x positions."""
    lines = [f"### グラフ {label} {info.get('title') or ''} ({'/'.join(info.get('chart_types') or [])})".rstrip()]
    for a in info.get('axes', []):
        bits = [f"{k}={v}" for k, v in a.items() if k != 'kind']
        if bits:
            lines.append(f"- 軸[{a['kind']}]: " + ', '.join(bits))
    for i, s in enumerate(info.get('series', []), 1):
        head = f"- 系列{i}"
        if s.get('name'):
            head += f" 名称='{s['name']}'"
        if s.get('color_name'):
            head += f" 色={s['color_name']}"
        if s.get('color'):
            head += f"({s['color']})"
        if s.get('val_ref'):
            head += f" 値参照={s['val_ref']}"
        if s.get('cat_ref'):
            head += f" 分類参照={s['cat_ref']}"
        lines.append(head)
        cats, vals = s.get('categories') or [], s.get('values') or []
        if vals:
            pairs = []
            for j, v in enumerate(vals):
                c = cats[j] if j < len(cats) else j + 1
                pairs.append(f'({c}, {v})')
            lines.append('  データ: ' + ', '.join(pairs))
    return '\n'.join(lines)
