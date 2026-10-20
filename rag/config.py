"""Paths and tunables.  Only the competition input locations are fixed here."""
from __future__ import annotations

import os
import unicodedata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHARE_DIR = os.path.join(BASE, 'share')
QUESTIONS_DIR_HINT = '質問回答'
ARTIFACTS = os.path.join(BASE, 'rag', 'artifacts')

DOCS_DIR = os.path.join(ARTIFACTS, 'docs')
TEXT_DIR = os.path.join(ARTIFACTS, 'text')
PDF_CACHE = os.path.join(ARTIFACTS, 'pdfcache')
IMAGE_DIR = os.path.join(ARTIFACTS, 'images')
INDEX_DIR = os.path.join(ARTIFACTS, 'index')
ANSWER_DIR = os.path.join(ARTIFACTS, 'answers')
LOG_DIR = os.path.join(ARTIFACTS, 'logs')

SUBMIT_DIR = os.path.join(BASE, 'submit')

# Answers are hard-capped by the evaluator at 1000 tokens.
MAX_ANSWER_TOKENS = 1000

SEED = 20250815


def find_dir(parent: str, name: str) -> str | None:
    """Locate a child directory whose name matches after normalisation."""
    target = unicodedata.normalize('NFKC', name)
    if not os.path.isdir(parent):
        return None
    for n in os.listdir(parent):
        if unicodedata.normalize('NFKC', n) == target and os.path.isdir(os.path.join(parent, n)):
            return os.path.join(parent, n)
    return None


def drive_root() -> str:
    """The share-drive root, located without depending on its exact bytes."""
    for n in sorted(os.listdir(SHARE_DIR)):
        p = os.path.join(SHARE_DIR, n)
        if os.path.isdir(p) and find_dir(p, 'プロジェクト'):
            return p
    # fall back: the only directory that is not the question folder
    for n in sorted(os.listdir(SHARE_DIR)):
        p = os.path.join(SHARE_DIR, n)
        if os.path.isdir(p) and unicodedata.normalize('NFKC', n) != QUESTIONS_DIR_HINT:
            return p
    raise RuntimeError(f'share drive root not found under {SHARE_DIR}')


def questions_dir() -> str | None:
    return find_dir(SHARE_DIR, QUESTIONS_DIR_HINT)


def ensure_dirs() -> None:
    for d in (ARTIFACTS, DOCS_DIR, TEXT_DIR, PDF_CACHE, IMAGE_DIR, INDEX_DIR,
              ANSWER_DIR, LOG_DIR, SUBMIT_DIR):
        os.makedirs(d, exist_ok=True)
