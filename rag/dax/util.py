"""Shared helpers for the document-extraction layer.

Everything here is generic: no knowledge of any particular file, project or
question is encoded.  Paths are handled Unicode-normalisation-safe because the
distributed share drive uses NFD-normalised Japanese folder names.
"""
from __future__ import annotations

import os
import re
import unicodedata
import zipfile
from typing import Iterable

from lxml import etree

# ---------------------------------------------------------------- namespaces
NS = {
    'w':    'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'w14':  'http://schemas.microsoft.com/office/word/2010/wordml',
    'a':    'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p':    'http://schemas.openxmlformats.org/presentationml/2006/main',
    'c':    'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'r':    'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'rel':  'http://schemas.openxmlformats.org/package/2006/relationships',
    's':    'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
    'xdr':  'http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing',
    'mc':   'http://schemas.openxmlformats.org/markup-compatibility/2006',
    'cx':   'http://schemas.microsoft.com/office/drawing/2014/chartex',
    'x14':  'http://schemas.microsoft.com/office/spreadsheetml/2009/9/main',
    'pkg':  'http://schemas.microsoft.com/office/2006/xmlPackage',
}


def q(tag: str) -> str:
    """'w:val' -> '{namespace}val' for lxml attribute/element access."""
    prefix, _, local = tag.partition(':')
    return f'{{{NS[prefix]}}}{local}'


# ---------------------------------------------------------------- text utils
def nfc(s: str) -> str:
    """Canonical composition — the share drive stores names as NFD."""
    return unicodedata.normalize('NFC', s)


def nkey(s: str) -> str:
    """Aggressive normalisation used for fuzzy matching of names/terms."""
    s = unicodedata.normalize('NFKC', s)
    s = s.lower()
    s = re.sub(r'[\s　_\-–—・,，、.。()（）\[\]【】"\'"'']+', '', s)
    return s


_FMT_TAG = re.compile(r'\[\[/?[^\]]*\]\]')


def strip_fmt(s: str) -> str:
    """Drop the inline formatting markers so phrases read continuously.

    The markers make bold/highlight visible, but they also cut phrases in half
    when only part of a phrase is emphasised, which defeats regex and lexical
    matching.
    """
    return _FMT_TAG.sub('', s)


def squeeze(s: str) -> str:
    """Collapse runs of whitespace but keep newlines meaningful."""
    s = s.replace('\r\n', '\n').replace('\r', '\n').replace('\x0b', '\n')
    s = re.sub(r'[ \t　]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


# --------------------------------------------------------------- path helpers
def walk_files(root: str, skip_dirs: Iterable[str] = ('__pycache__', '.git', '.ipynb_checkpoints')) -> list[str]:
    """All files under *root*, sorted, with junk directories and Office
    lock-files (``~$name``) excluded."""
    skip = set(skip_dirs)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        for f in sorted(filenames):
            if f.startswith('~$'):
                continue
            out.append(os.path.join(dirpath, f))
    return out


def find_child_dir(parent: str, name: str) -> str | None:
    """Locate a subdirectory by normalisation-insensitive name."""
    target = nkey(name)
    for n in os.listdir(parent):
        if nkey(n) == target and os.path.isdir(os.path.join(parent, n)):
            return os.path.join(parent, n)
    return None


# --------------------------------------------------------------- colour utils
_HIGHLIGHT_NAMES = {
    'yellow': 'FFFF00', 'green': '00FF00', 'cyan': '00FFFF', 'magenta': 'FF00FF',
    'blue': '0000FF', 'red': 'FF0000', 'darkBlue': '000080', 'darkCyan': '008080',
    'darkGreen': '008000', 'darkMagenta': '800080', 'darkRed': '800000',
    'darkYellow': '808000', 'darkGray': '808080', 'lightGray': 'C0C0C0',
    'black': '000000', 'white': 'FFFFFF',
}

# Coarse human colour buckets.  Questions ask for "yellow", "orange", "blue"
# highlights, so a hue/lightness classifier is the generic way to answer them.
# Office themes use very pale tints of the same hues as banding, so the hue is
# reported together with a strength qualifier instead of being collapsed to
# "white" — the caller decides whether a pale tint counts as a highlight.
def color_info(rgb: str | None) -> tuple[str | None, str | None]:
    """``RRGGBB`` -> (colour word, strength in {vivid, medium, pale})."""
    if not rgb:
        return None, None
    rgb = rgb.strip().lstrip('#')
    if len(rgb) == 8:            # ARGB from Excel
        rgb = rgb[2:]
    if len(rgb) != 6:
        return None, None
    try:
        r, g, b = (int(rgb[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None, None
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d <= 8:                   # achromatic
        if mx >= 235:
            return 'white', 'vivid'
        if mx <= 40:
            return 'black', 'vivid'
        return 'gray', 'vivid'
    if mx == r:
        h = 60 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60 * (((b - r) / d) + 2)
    else:
        h = 60 * (((r - g) / d) + 4)
    strength = 'vivid' if d >= 120 else ('medium' if d >= 45 else 'pale')
    if h < 16 or h >= 330:
        name = 'red'
    elif h < 40:
        name = 'orange'
    elif h < 70:
        name = 'yellow'
    elif h < 160:
        name = 'green'
    elif h < 195:
        # only genuine cyans; teal-blues above this read as "青" to a reader
        name = 'cyan'
    elif h < 262:
        name = 'blue'
    else:
        name = 'purple'
    return name, strength


def color_name(rgb: str | None) -> str | None:
    return color_info(rgb)[0]


def color_label(rgb: str | None) -> str | None:
    """'yellow' for a saturated fill, 'pale-orange' for a faint tint."""
    name, strength = color_info(rgb)
    if not name:
        return None
    if strength == 'pale' and name not in ('white', 'black', 'gray'):
        return f'pale-{name}'
    return name


def highlight_to_rgb(val: str | None) -> str | None:
    if not val or val == 'none':
        return None
    return _HIGHLIGHT_NAMES.get(val, None) or (val if re.fullmatch(r'[0-9A-Fa-f]{6}', val or '') else None)


def apply_tint(rgb: str, tint: float) -> str:
    """OOXML theme-colour tint/shade, as used by Excel's ``theme``+``tint``."""
    if not tint:
        return rgb
    out = []
    for i in (0, 2, 4):
        c = int(rgb[i:i + 2], 16)
        if tint < 0:
            c = int(round(c * (1.0 + tint)))
        else:
            c = int(round(c * (1.0 - tint) + (255 * tint)))
        out.append(f'{max(0, min(255, c)):02X}')
    return ''.join(out)


def theme_colors(zf: zipfile.ZipFile) -> list[str]:
    """Ordered theme colour palette (`RRGGBB`) from a package's theme part.

    Index order follows the ``clrScheme`` mapping used by Excel/Word/PowerPoint
    (lt1/dk1 and lt2/dk2 are swapped relative to XML document order).
    """
    name = next((n for n in zf.namelist() if re.search(r'theme/theme\d*\.xml$', n)), None)
    if not name:
        return []
    root = etree.fromstring(zf.read(name))
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
    palette = [found.get(k, '') for k in order]
    if len(palette) >= 4:        # Excel's index order swaps the first two pairs
        palette[0], palette[1] = palette[1], palette[0]
        palette[2], palette[3] = palette[3], palette[2]
    return palette


def read_xml(zf: zipfile.ZipFile, name: str):
    try:
        return etree.fromstring(zf.read(name))
    except KeyError:
        return None


def rels_for(zf: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """rId -> (target-path-resolved, relationship-type) for an OPC part."""
    d, base = os.path.split(part)
    rp = f'{d}/_rels/{base}.rels' if d else f'_rels/{base}.rels'
    root = read_xml(zf, rp)
    out: dict[str, tuple[str, str]] = {}
    if root is None:
        return out
    for rel in root:
        rid = rel.get('Id')
        tgt = rel.get('Target') or ''
        typ = (rel.get('Type') or '').rsplit('/', 1)[-1]
        if rel.get('TargetMode') == 'External':
            out[rid] = (tgt, typ)
            continue
        out[rid] = (os.path.normpath(os.path.join(d, tgt)).replace('\\', '/'), typ)
    return out
