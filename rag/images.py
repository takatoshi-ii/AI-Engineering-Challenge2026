"""Collects every image the corpus contains into one addressable registry.

Sources: standalone picture files, embedded media inside Office packages,
notebook image outputs, and rasterised pages of PDFs that carry no text layer
(scanned reports).  Each entry gets a stable id so a vision transcription can be
cached against it and reused across runs.

Run: python rag/images.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows consoles are cp932 here; a model answer containing e.g. U+202F would
# otherwise abort the whole run inside a progress print.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


import config
from dax import nb_ex, pdf_ex
from dax.util import nfc

PDF_PAGE_DPI = 190
MIN_BYTES = 3000        # skip icons/spacers


def _sha(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:16]


def _save(data: bytes, iid: str) -> str:
    path = os.path.join(config.IMAGE_DIR, f'{iid}.png')
    if not os.path.exists(path):
        from PIL import Image
        import io
        try:
            im = Image.open(io.BytesIO(data))
            if im.mode in ('P', 'RGBA', 'LA'):
                im = im.convert('RGB')
            # cap the long edge: vision models gain nothing from huge scans
            if max(im.size) > 2200:
                r = 2200 / max(im.size)
                im = im.resize((int(im.size[0] * r), int(im.size[1] * r)))
            im.save(path, 'PNG', optimize=True)
        except Exception:
            with open(path, 'wb') as fh:
                fh.write(data)
    return path


def build() -> list[dict]:
    config.ensure_dirs()
    man = json.load(open(os.path.join(config.ARTIFACTS, 'manifest.json'), encoding='utf-8'))
    entries: list[dict] = []

    for d in man['documents']:
        did, rel, ext = d['id'], d['rel_path'], d['ext']
        jpath = os.path.join(config.DOCS_DIR, f'{did}.json')
        if not os.path.exists(jpath):
            continue
        rec = json.load(open(jpath, encoding='utf-8'))
        abs_path = rec.get('abs_path')

        if ext in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'):
            data = open(abs_path, 'rb').read()
            iid = f'{did}_file_{_sha(data)}'
            entries.append({'image_id': iid, 'doc_id': did, 'rel_path': rel,
                            'origin': 'file', 'locator': rel,
                            'project': d.get('project'),
                            'path': _save(data, iid)})

        elif ext == '.ipynb':
            for name, data in nb_ex.images(rec):
                if len(data) < MIN_BYTES:
                    continue
                iid = f'{did}_{name}'
                entries.append({'image_id': iid, 'doc_id': did, 'rel_path': rel,
                                'origin': 'notebook_output', 'locator': name,
                                'project': d.get('project'),
                                'path': _save(data, iid)})

        elif ext == '.pdf':
            pages = rec.get('pages') or []
            need = [p for p in pages if p.get('is_image_page')]
            if not need:
                continue
            for p in need:
                data = pdf_ex.render_page_png(abs_path, p['page'], dpi=PDF_PAGE_DPI)
                iid = f'{did}_p{p["page"]:03d}'
                entries.append({'image_id': iid, 'doc_id': did, 'rel_path': rel,
                                'origin': 'pdf_page', 'locator': f"p.{p['page']}",
                                'page': p['page'], 'project': d.get('project'),
                                'path': _save(data, iid)})

        elif ext in ('.pptx', '.docx', '.xlsx'):
            src = rec.get('decrypted_copy') or abs_path
            try:
                zf = zipfile.ZipFile(src)
            except Exception:
                continue
            for name in zf.namelist():
                if '/media/' not in name:
                    continue
                if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    continue
                data = zf.read(name)
                if len(data) < MIN_BYTES:
                    continue
                iid = f'{did}_media_{_sha(data)}'
                entries.append({'image_id': iid, 'doc_id': did, 'rel_path': rel,
                                'origin': 'embedded_media', 'locator': name,
                                'project': d.get('project'),
                                'path': _save(data, iid)})
            zf.close()
            # rendered slide/page images make layout-dependent questions readable
            pdf = rec.get('rendered_pdf')
            if pdf and os.path.exists(pdf) and ext == '.pptx':
                try:
                    n = len(rec.get('page_text') or [])
                    for pg in range(1, n + 1):
                        data = pdf_ex.render_page_png(pdf, pg, dpi=150)
                        iid = f'{did}_slide{pg:03d}'
                        entries.append({'image_id': iid, 'doc_id': did, 'rel_path': rel,
                                        'origin': 'slide_render', 'locator': f'slide {pg}',
                                        'page': pg, 'project': d.get('project'),
                                        'path': _save(data, iid)})
                except Exception:
                    pass

    out = os.path.join(config.ARTIFACTS, 'images.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(entries, fh, ensure_ascii=False, indent=1)
    by_origin: dict[str, int] = {}
    for e in entries:
        by_origin[e['origin']] = by_origin.get(e['origin'], 0) + 1
    print(f'{len(entries)} images -> {out}')
    for k, v in sorted(by_origin.items()):
        print(f'  {k:18s} {v}')
    return entries


if __name__ == '__main__':
    build()
