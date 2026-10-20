"""Jupyter notebook extraction: source, execution order and *outputs*.

The outputs are the point — several questions ask about a value or an axis tick
that only exists in a stored output (text or PNG), never in the source.
"""
from __future__ import annotations

import base64
import json

import re

from .util import squeeze

MAX_OUT_CHARS = 6000

# Figures are sometimes pasted into a markdown cell as a data URI rather than
# produced as a cell output.  Left inline they would flood the text with base64
# (megabytes of it); dropped, a question about that figure becomes unanswerable.
# So they are lifted out into the image registry and replaced by a reference.
_DATA_URI = re.compile(
    r'!\[[^\]]*\]\(\s*data:image/(png|jpe?g|gif);base64,([A-Za-z0-9+/=\s]+?)\s*\)'
    r'|<img[^>]+src=["\']data:image/(png|jpe?g|gif);base64,([A-Za-z0-9+/=\s]+?)["\'][^>]*>',
    re.I)


def _lift_images(text: str, cell_index: int, sink: list) -> str:
    """Replace inline base64 images with a placeholder, collecting the bytes."""
    def repl(m):
        mime = m.group(1) or m.group(3)
        payload = m.group(2) or m.group(4)
        idx = len(sink) + 1
        iid = f'cell{cell_index}_embedded{idx}'
        sink.append({'kind': 'image', 'mime': f'image/{(mime or "png").lower()}',
                     'id': iid, 'b64': re.sub(r'\s+', '', payload), 'inline': True})
        return f'[埋め込み画像 id={iid} — 画像として要参照]'
    return _DATA_URI.sub(repl, text)


def extract(path: str) -> dict:
    with open(path, encoding='utf-8') as fh:
        nb = json.load(fh)
    cells = []
    img_i = 0
    for i, c in enumerate(nb.get('cells', [])):
        src = ''.join(c.get('source') or [])
        inline: list = []
        src = _lift_images(src, i, inline)
        # attachments are the other way notebooks carry pasted images
        for name, att in (c.get('attachments') or {}).items():
            for mime, payload in att.items():
                if mime.startswith('image/'):
                    inline.append({'kind': 'image', 'mime': mime,
                                   'id': f'cell{i}_attach{len(inline) + 1}',
                                   'b64': payload if isinstance(payload, str)
                                   else ''.join(payload), 'inline': True})
        item = {'index': i, 'type': c.get('cell_type'),
                'exec_count': c.get('execution_count'), 'source': src,
                'outputs': list(inline)}
        for o in c.get('outputs') or []:
            ot = o.get('output_type')
            if ot == 'stream':
                item['outputs'].append({'kind': 'stream', 'name': o.get('name'),
                                        'text': ''.join(o.get('text') or [])})
            elif ot in ('execute_result', 'display_data'):
                data = o.get('data') or {}
                if 'text/plain' in data:
                    item['outputs'].append({'kind': 'text',
                                            'text': ''.join(data['text/plain'])})
                if 'text/html' in data:
                    html = _lift_images(''.join(data['text/html']), i, item['outputs'])
                    item['outputs'].append({'kind': 'html', 'text': html})
                for mime in ('image/png', 'image/jpeg'):
                    if mime in data:
                        img_i += 1
                        item['outputs'].append({
                            'kind': 'image', 'mime': mime, 'id': f'cell{i}_img{img_i}',
                            'b64': data[mime] if isinstance(data[mime], str)
                            else ''.join(data[mime]),
                        })
            elif ot == 'error':
                item['outputs'].append({'kind': 'error',
                                        'text': o.get('ename', '') + ': ' + o.get('evalue', '')})
        cells.append(item)
    doc = {'kind': 'ipynb', 'cells': cells,
           'language': (nb.get('metadata', {}).get('kernelspec', {}) or {}).get('language')}
    doc['text'] = render(doc)
    return doc


def images(doc: dict) -> list[tuple[str, bytes]]:
    """(id, png-bytes) for every image output."""
    out = []
    for c in doc['cells']:
        for o in c['outputs']:
            if o['kind'] == 'image':
                try:
                    out.append((o['id'], base64.b64decode(o['b64'])))
                except Exception:
                    pass
    return out


def render(doc: dict) -> str:
    lines = []
    for c in doc['cells']:
        if c['type'] == 'markdown':
            lines.append(f"\n### [markdown cell {c['index']}]")
            lines.append(c['source'])
            for o in c['outputs']:
                if o['kind'] == 'image':
                    lines.append(f"[埋め込み画像 id={o['id']} ({o['mime']}) — 画像として要参照]")
            continue
        lines.append(f"\n### [code cell {c['index']} exec={c['exec_count']}]")
        lines.append('```python')
        lines.append(c['source'])
        lines.append('```')
        for o in c['outputs']:
            if o['kind'] == 'image':
                lines.append(f"[出力画像 id={o['id']} ({o['mime']}) — 画像として要参照]")
            else:
                t = o.get('text') or ''
                if len(t) > MAX_OUT_CHARS:
                    t = t[:MAX_OUT_CHARS] + f'\n...(出力を{len(t) - MAX_OUT_CHARS}文字省略)'
                lines.append(f"[出力:{o['kind']}]")
                lines.append(t)
    return squeeze('\n'.join(lines))
