"""Optional high-fidelity rendering of Office documents via the installed suite.

Two things are impossible to obtain from the OOXML alone:

* **Where a page break falls.**  Word decides pagination at layout time, so
  "which page is heading X on?" can only be answered from a laid-out document.
* **What a slide/sheet actually looks like.**  Rendering to an image lets a
  vision model read figures, callouts and overlapping shapes.

Both are produced by exporting to PDF.  Everything here is best-effort: if no
suite is available the pipeline still works, minus page-level attribution.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys

_WORD_FMT_PDF = 17
_PPT_FMT_PDF = 32
_XL_FMT_PDF = 0


def _cache_path(cache_dir: str, src: str) -> str:
    h = hashlib.sha1(os.path.abspath(src).encode('utf-8')).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(src))[0][:40]
    safe = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in stem)
    return os.path.join(cache_dir, f'{safe}_{h}.pdf')


def _soffice() -> str | None:
    for c in (shutil.which('soffice'), shutil.which('libreoffice'),
              r'C:\Program Files\LibreOffice\program\soffice.exe',
              r'C:\Program Files (x86)\LibreOffice\program\soffice.exe'):
        if c and os.path.exists(c):
            return c
    return None


def _via_libreoffice(src: str, out_pdf: str) -> bool:
    exe = _soffice()
    if not exe:
        return False
    outdir = os.path.dirname(out_pdf)
    try:
        subprocess.run([exe, '--headless', '--norestore', '--convert-to', 'pdf',
                        '--outdir', outdir, src],
                       check=True, capture_output=True, timeout=300)
    except Exception:
        return False
    produced = os.path.join(outdir, os.path.splitext(os.path.basename(src))[0] + '.pdf')
    if os.path.exists(produced):
        if produced != out_pdf:
            shutil.move(produced, out_pdf)
        return True
    return False


def _via_com(src: str, out_pdf: str) -> bool:
    if sys.platform != 'win32':
        return False
    try:
        import pythoncom  # noqa: F401
        import win32com.client as win32
    except ImportError:
        return False

    ext = os.path.splitext(src)[1].lower()
    pythoncom.CoInitialize()
    app = None
    try:
        if ext in ('.docx', '.doc', '.rtf'):
            app = win32.DispatchEx('Word.Application')
            app.Visible = False
            app.DisplayAlerts = 0
            doc = app.Documents.Open(os.path.abspath(src), ReadOnly=True,
                                     AddToRecentFiles=False, Visible=False,
                                     PasswordDocument='-no-')
            try:
                doc.ExportAsFixedFormat(os.path.abspath(out_pdf), _WORD_FMT_PDF)
            finally:
                doc.Close(False)
        elif ext in ('.pptx', '.ppt'):
            app = win32.DispatchEx('PowerPoint.Application')
            pres = app.Presentations.Open(os.path.abspath(src), ReadOnly=True,
                                          WithWindow=False)
            try:
                pres.SaveAs(os.path.abspath(out_pdf), _PPT_FMT_PDF)
            finally:
                pres.Close()
        elif ext in ('.xlsx', '.xls', '.xlsm'):
            app = win32.DispatchEx('Excel.Application')
            app.Visible = False
            app.DisplayAlerts = False
            wb = app.Workbooks.Open(os.path.abspath(src), ReadOnly=True,
                                    UpdateLinks=0, Password='-no-')
            try:
                wb.ExportAsFixedFormat(_XL_FMT_PDF, os.path.abspath(out_pdf))
            finally:
                wb.Close(False)
        else:
            return False
        return os.path.exists(out_pdf)
    except Exception:
        return False
    finally:
        try:
            if app is not None:
                app.Quit()
        except Exception:
            pass
        pythoncom.CoUninitialize()


def to_pdf(src: str, cache_dir: str) -> str | None:
    """Convert an Office document to PDF, caching the result.  None if no
    converter is available."""
    os.makedirs(cache_dir, exist_ok=True)
    out = _cache_path(cache_dir, src)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    if _via_com(src, out) or _via_libreoffice(src, out):
        return out
    return None


def available() -> bool:
    if _soffice():
        return True
    if sys.platform != 'win32':
        return False
    try:
        import win32com.client as win32  # noqa: F401
        return True
    except ImportError:
        return False
