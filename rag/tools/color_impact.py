"""How much do the tuned hue boundaries change, versus untuned ones?

Compares the classifier in use against a plain textbook hue split, over every
fill/font/series colour that actually occurs in the corpus.
"""
import collections, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import config
from dax.util import color_info


def textbook(rgb):
    """Un-tuned reference: equal-ish hue sectors, saturation cut-off."""
    r, g, b = (int(rgb[i:i + 2], 16) for i in (0, 2, 4))
    mx, mn = max(r, g, b), min(r, g, b)
    v = mx / 255.0
    s = 0.0 if mx == 0 else (mx - mn) / mx
    if s < 0.12:
        return 'white' if v > 0.93 else ('black' if v < 0.15 else 'gray')
    d = mx - mn
    if mx == r:
        h = 60 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60 * (((b - r) / d) + 2)
    else:
        h = 60 * (((r - g) / d) + 4)
    for lim, nm in ((15, 'red'), (45, 'orange'), (70, 'yellow'), (165, 'green'),
                    (195, 'cyan'), (255, 'blue'), (345, 'purple')):
        if h < lim:
            return nm
    return 'red'


# every hex colour the extraction actually emitted
seen = collections.Counter()
for name in os.listdir(config.TEXT_DIR):
    t = open(os.path.join(config.TEXT_DIR, name), encoding='utf-8').read()
    for m in re.finditer(r'\(([0-9A-F]{6})\)', t):
        seen[m.group(1)] += 1

print(f'{len(seen)} distinct colours in the corpus\n')
print(f"{'hex':8s} {'count':>6}  {'tuned':14s} {'untuned':10s}")
print('-' * 48)
diff = 0
for hexv, n in seen.most_common():
    a = color_info(hexv)
    tuned = a[0] if a[1] != 'pale' or a[0] in ('white', 'black', 'gray') else f'pale-{a[0]}'
    un = textbook(hexv)
    mark = '' if tuned.replace('pale-', '') == un else '  <-- differs'
    if mark:
        diff += 1
    print(f'{hexv:8s} {n:>6}  {tuned:14s} {un:10s}{mark}')
print(f'\n{diff} of {len(seen)} colours classified differently')
