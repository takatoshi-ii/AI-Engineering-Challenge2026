"""The answering agent.

One question in, one Japanese answer out.  The model is given navigation,
reading, vision and computation tools over the *unmodified* share drive and
must ground every answer in what it retrieves; nothing about any specific
question, project or file is encoded here.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import index as index_mod
import llm as llm_mod
import pysandbox
from dax.util import nkey, nfc, strip_fmt

RAG_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM = """あなたは、あるデータ分析会社（データアステル社）の共有ドライブを調査して、社内からの質問に日本語で回答するアシスタントです。

## 絶対的な原則
- 回答は必ず、ツールで実際に読み取った共有ドライブ内の資料の内容だけに基づくこと。あなたの一般知識や推測で数値・名称・ページ番号を作ってはならない。
- 根拠が見つからないまま推測で答えてはならない。十分に探索したうえで確証が得られない場合のみ「資料からは確認できませんでした」と答える。
- 採点は 完全正解=+1、軽微な誤りを含む有用な回答=+0.5、「わからない」=0、誤答・無関係=-1 である。したがって、正しい確信が半分未満なら「わからない」と答えるほうが得点は高い。ただし少し探せば分かることを安易に諦めてはならない。

## 調査の進め方
1. まず質問文に出てくる案件名・略称・ファイル名を特定する。意味の取れない英字略語や、案件の通称と思われる語が含まれている場合は、推測せず lookup_glossary で必ず引く（用語集は共有ドライブ内にあり、略称・案件名の対応はそこに定義されている）。
2. list_files / search_corpus で該当ファイルを絞り込み、read_document で中身を読む。
3. 計算・集計・相関・しきい値探索などが必要な場合は run_python で実際に計算する。目分量で答えない。
4. 図・グラフ・画像PDF・座席表など、テキストとして読めない情報は view_image で実際に画像を見る。
5. 複数案件にまたがる質問（「全案件で」「すべて挙げよ」「最も多い案件」など）は、search_corpus で断片を拾うのではなく、run_python で each_text("01.契約") のように該当種別のファイルを全件走査し、正規表現で値を取り出して集計する。この方が速く確実である。
6. 1回 read_document しただけで諦めないこと。1回の読み込みでは全文は返らないので offset を進めて読む、grep の語を言い換える、別のファイルを当たる、といった手を尽くす。

## 読み取りのヒント
- 抽出テキストでは書式が [[bold]]…[[/]] / [[hl:yellow]]…[[/]] / [[color:red]]…[[/]] の形で示される。「太字」「黄色ハイライト」「赤字」を問う質問はこれで判定する。図形の塗り色は「塗り=red(...)」のように示される。
- 書式マーカーは語句を分断することがある（例:「合計 [[bold]]1,234円[[/]] です」のように、数値だけ別扱いになる）。書式が関係ない質問では read_document に plain=true を付けるか、run_python では plain() を通してから正規表現をかけること。
- Excel は「塗りつぶし色の使用状況」「数式セル一覧」「オートフィルター」「ピボットテーブル」の節を見る。少数派の色が「ハイライト」であることが多い。pale-xxx は淡い地色で、通常は装飾。
- グラフは系列名・カテゴリ・値がそのまま数値で書き出されている。x=◯ のときの y はそこから読む。
- 「何ページ目か」を問う質問は、docx/pptx なら「ページ割り（レイアウト実測）」の節、PDF なら「## ページ N」の見出しで判断する。pptx の「ページ」はスライド番号を指すこともあるので文脈で判断する。
- 旧版と新版の比較（old/_v1/_r1 と最新版など）は、両方を read_document で全文読むと文脈が溢れて回答できなくなる。必ず run_python で機械的に差分を取ること。例:
  ```
  import difflib
  a = plain(doc_text("案件名", "old", "提案書")).splitlines()
  b = plain(doc_text("案件名", "提案書")).splitlines()
  for l in difflib.unified_diff(a, b, lineterm="", n=1):
      if l[:1] in "+-" and l[:3] not in ("+++", "---"): print(l)
  ```
  差分行だけを見れば、変更箇所は確実かつ短く把握できる。書式の変化も見たい場合は raw=True を付ける。

## 回答の書き方（最重要・厳守）
採点は「模範解答と同じ要素が過不足なく含まれるか」で行われる。**説明を足すと「要素の追加」とみなされ、誤答（-1点）になる。**
模範解答は「Recall」「43日」「5,775,000円」「T09、T10、T11、T12」「アサインされていない」のような、答えそのものだけの短い文字列である。同じ粒度で答えること。

- **答えそのものだけを書く。** 前置き、理由、根拠、出典、ファイル名、経緯、言い換え、確認の経過を書いてはならない。
- ただし**「答えそのもの」が説明である設問もある。** 「規定内容を答えよ」「変更を挙げよ」「項目を挙げよ」「抽出条件と集計内容を答えよ」なら、その規定・変更・項目・条件の**中身が答え**である。中身を書かずに一語で済ませてはならない。削るのは出典やファイル名や調査経過であって、問われている中身ではない。
- **「該当なし」と書いてよいのは、探した結果その対象が本当に存在しないと確認できたときだけ。** 見つけられなかった、判断に迷った、という理由で「該当なし」と書いてはならない。その場合は「資料からは確認できませんでした」と書く。両者は意味が違う。
- submit_answer を呼ぶ直前に確認すること: **evidence に書いた内容のうち、質問が答えとして求めているものが answer に入っているか。** 調べた結果を evidence だけに書いて answer を空にするのは誤りである。
- 括弧書きの補足を付けない。ただし答え自体が括弧を含む場合（例:「対象外（契約明記）」）はそのまま書く。
- 複数要素は「、」で区切って並べるだけにする。番号や箇条書き記号、各要素への注釈を付けない。
- 「〜です」「〜になります」「以下のとおりです」といった説明的な文にせず、値・名称・IDを直接書く。
- 質問が単位・小数桁・丸め方を指定していればその形式に従う。指定がなければ資料の表記に合わせる。
- **単位・助数詞は答えの一部なので必ず残す。** 「3年間」を「3年」に、「20日」を「20」に縮めてはならない。短くするのは説明部分だけで、値そのものの表記は資料や質問の言い方に合わせる。
- 「主略称で」「案件略称で」と指定された場合は、用語集の**主略称**（例: 別名ではなく AOMINE、AYM のような正式な略称）で答える。案件名や別名で答えてはならない。迷ったら lookup_glossary で主略称を確認する。
- 根拠は answer ではなく evidence に書く。evidence は採点対象外なので、そちらには自由に書いてよい。

悪い例: 「Recall（再現率）です。提案書のスライド4「目的」の分析KPI欄に『Recall ★重視』と明記されています。」
良い例: 「Recall」
悪い例: 「変更された箇所は1箇所です（スライド8「5. 実施体制」）。変更前：QAレビューア 池田 直哉、変更後：小林 直樹」
良い例: 「QAレビューア：池田 直哉 → 小林 直樹」
- 質問文が指定する形式・単位・小数桁・丸め方に厳密に従う。指定がなければ資料の表記に合わせる。
- 列挙を求められたら、過不足なく列挙する。余計な要素を足すと誤答扱いになる。順序の指定（ID昇順、座席表の順、資料の出現順）があれば従う。
- 資料で定義された識別子（タスクID、アクションID、マイルストーンID、列名、パラメータ名など）は資料の表記どおりに書く。
- それ以外は、社内略称ではなく通常の表現で書く（質問が略称での回答を明示している場合を除く）。
- 該当するものが存在しない場合は、「該当なし」である旨を明確に答える。
- 回答は1000トークン以内。長い説明は不要。

最後は必ず submit_answer ツールで最終回答を提出すること。"""


TOOLS = [
    {
        'name': 'list_files',
        'description': '共有ドライブのファイル一覧を取得する。project や pattern で絞り込める。',
        'parameters': {
            'type': 'object',
            'properties': {
                'project': {'type': 'string', 'description': '案件名または略称（部分一致）。省略時は全件。'},
                'pattern': {'type': 'string', 'description': 'パス・ファイル名に含まれる文字列（部分一致、空白区切りでAND）。'},
            },
        },
    },
    {
        'name': 'search_corpus',
        'description': '抽出済み全文に対するキーワード検索。該当箇所の抜粋とファイルパスを返す。',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string'},
                'project': {'type': 'string'},
                'ext': {'type': 'string', 'description': '.xlsx / .docx / .pptx / .pdf / .py などで絞る'},
                'top_k': {'type': 'integer'},
            },
            'required': ['query'],
        },
    },
    {
        'name': 'read_document',
        'description': ('指定ファイルの抽出済みテキストを読む。書式・色・数式・グラフ数値・'
                        'ページ割りを含む。長い場合は offset/limit で続きを読む。'),
        'parameters': {
            'type': 'object',
            'properties': {
                'rel_path': {'type': 'string', 'description': 'list_files/search_corpus が返すパス（部分一致でも可）'},
                'offset': {'type': 'integer', 'default': 0},
                'limit': {'type': 'integer', 'default': 18000},
                'grep': {'type': 'string', 'description': '指定すると、その語を含む箇所の前後だけを抜き出す'},
                'plain': {'type': 'boolean',
                          'description': 'true にすると [[bold]] 等の書式マーカーを除去した素の文章で返す。'
                                         '書式が関係ない質問や、語句が分断されて読みにくいときに使う。'},
            },
            'required': ['rel_path'],
        },
    },
    {
        'name': 'view_image',
        'description': ('図・グラフ・画像PDFのページ・座席表などを画像として実際に見る。'
                        'list_images で得た image_id、または rel_path(+page) を指定する。'),
        'parameters': {
            'type': 'object',
            'properties': {
                'image_id': {'type': 'string'},
                'rel_path': {'type': 'string'},
                'page': {'type': 'integer', 'description': 'PDF/スライドのページ番号'},
            },
        },
    },
    {
        'name': 'list_images',
        'description': '指定ファイル（または案件）に紐づく画像の一覧を返す。',
        'parameters': {
            'type': 'object',
            'properties': {
                'rel_path': {'type': 'string'},
                'project': {'type': 'string'},
            },
        },
    },
    {
        'name': 'locate_in_pages',
        'description': ('「何ページ目か」を調べる。指定した語句が、レイアウト実測後の'
                        '何ページ目（PDFならページ番号、pptxならスライド番号）に現れるかを返す。'
                        '空白や表記ゆれを無視して照合する。'),
        'parameters': {
            'type': 'object',
            'properties': {
                'rel_path': {'type': 'string'},
                'text': {'type': 'string', 'description': '探したい見出しや語句'},
            },
            'required': ['rel_path', 'text'],
        },
    },
    {
        'name': 'run_python',
        'description': ('Python(pandas/numpy/sklearn/openpyxl等)を実行して集計・計算する。'
                        'ヘルパー: find("案件","train.csv")=生ファイル検索, read_csv()/read_excel(), '
                        'doc_text(<パスの一部>...)=抽出済みテキスト取得, '
                        'each_text(<パスの一部>...)=一致する全ファイルを(パス,本文)で順に走査（案件横断集計に有効）。'
                        '結果は必ず print すること。'),
        'parameters': {
            'type': 'object',
            'properties': {'code': {'type': 'string'}},
            'required': ['code'],
        },
    },
    {
        'name': 'lookup_glossary',
        'description': '社内用語集を引く。略称→正式名称、案件略称→案件名。引数なしで全件返す。',
        'parameters': {
            'type': 'object',
            'properties': {'term': {'type': 'string'}},
        },
    },
    {
        'name': 'submit_answer',
        'description': ('最終回答を提出する。answer には答えそのものだけを入れ、'
                        '説明・根拠は evidence に分けること。'),
        'parameters': {
            'type': 'object',
            'properties': {
                'answer': {'type': 'string', 'description':
                           '答えそのものだけ。前置き・理由・出典・ファイル名・調査経過は入れない。'
                           '値や名称を問う設問は値だけ書く（例:「Recall」「43日」'
                           '「T09、T10、T11、T12」）。'
                           '一方、規定内容・変更点・項目・抽出条件などを問う設問は、'
                           'その中身自体が答えなので必ず書く（一語に潰さない）。'
                           '「該当なし」は、存在しないと確認できた場合のみ。'
                           '見つけられなかった場合は「資料からは確認できませんでした」と書く。'},
                'evidence': {'type': 'string', 'description': '根拠となったファイルと箇所（採点対象外なので自由に書いてよい）'},
                'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
            },
            'required': ['answer'],
        },
    },
]


class Toolbox:
    def __init__(self, vision_capable: bool = True):
        # a text-only answering model still works: the OCR stage has already
        # turned every picture into text, so view_image degrades to that
        self.vision_capable = vision_capable
        self.root = config.drive_root()
        self.manifest = json.load(open(os.path.join(config.ARTIFACTS, 'manifest.json'),
                                       encoding='utf-8'))
        self.docs = self.manifest['documents']
        self.by_id = {d['id']: d for d in self.docs}
        self.projects = self.manifest['projects']
        self.terms = self.manifest['glossary_terms']
        self.bm25 = index_mod.load()
        ipath = os.path.join(config.ARTIFACTS, 'images.json')
        self.images = json.load(open(ipath, encoding='utf-8')) if os.path.exists(ipath) else []
        self.vision_notes = {}
        vpath = os.path.join(config.ARTIFACTS, 'vision.json')
        if os.path.exists(vpath):
            self.vision_notes = json.load(open(vpath, encoding='utf-8'))

    # ------------------------------------------------------------- helpers
    def _match_docs(self, project: str | None = None, pattern: str | None = None) -> list[dict]:
        out = self.docs
        if project:
            keys = [nkey(project)]
            for p in self.projects:
                names = [p.get('display'), p.get('official'), p.get('abbrev')] + (p.get('aliases') or [])
                if any(nkey(project) == nkey(n) or (len(nkey(project)) >= 2 and nkey(project) in nkey(n))
                       for n in names if n):
                    keys.append(nkey(p['display']))
            out = [d for d in out if d.get('project') and
                   any(k in nkey(d['project']) or nkey(d['project']) in k for k in keys)]
        if pattern:
            frags = [nkey(x) for x in pattern.split() if x.strip()]
            out = [d for d in out if all(f in nkey(d['rel_path']) for f in frags)]
        return out

    def _resolve_doc(self, rel_path: str) -> dict | None:
        target = nkey(rel_path)
        exact = [d for d in self.docs if nkey(d['rel_path']) == target]
        if exact:
            return exact[0]
        cands = [d for d in self.docs if target in nkey(d['rel_path'])]
        if not cands:
            frags = [nkey(x) for x in re.split(r'[\\/\s]+', rel_path) if x.strip()]
            cands = [d for d in self.docs if all(f in nkey(d['rel_path']) for f in frags)]
        if not cands:
            return None
        cands.sort(key=lambda d: len(d['rel_path']))
        return cands[0]

    # --------------------------------------------------------------- tools
    def list_files(self, project=None, pattern=None, **_) -> str:
        docs = self._match_docs(project, pattern)
        if not docs:
            return '該当ファイルなし。project/pattern を緩めて再試行してください。'
        lines = [f'{len(docs)} 件']
        for d in docs[:400]:
            extra = f" p{d['n_pages']}" if d.get('n_pages') else ''
            lines.append(f"- {d['rel_path']} ({d['size']}B{extra})")
        if len(docs) > 400:
            lines.append(f'...ほか {len(docs) - 400} 件')
        return '\n'.join(lines)

    def search_corpus(self, query='', project=None, ext=None, top_k=12, **_) -> str:
        allow = None
        if project or ext:
            ok = self._match_docs(project, None)
            ids = {d['id'] for d in ok} if project else {d['id'] for d in self.docs}
            if ext:
                ids &= {d['id'] for d in self.docs if d['ext'] == ext}
            allow = {i for i, m in enumerate(self.bm25.docs) if m['doc_id'] in ids}
            if not allow:
                return '絞り込み条件に合うファイルがありません。'
        hits = self.bm25.search(query, top_k=int(top_k or 12), allow=allow)
        if not hits:
            return '該当なし。別の語で検索してください。'
        out = []
        for score, m in hits:
            out.append(f"--- {m['rel_path']}  [{m.get('anchor') or ''}] score={score:.1f}\n"
                       f"{m['preview']}")
        return '\n'.join(out)

    def read_document(self, rel_path='', offset=0, limit=18000, grep=None,
                      plain=False, **_) -> str:
        d = self._resolve_doc(rel_path)
        if not d:
            return f'ファイルが見つかりません: {rel_path}\nlist_files で確認してください。'
        tp = os.path.join(config.TEXT_DIR, f"{d['id']}.md")
        if not os.path.exists(tp):
            return f'抽出テキストがありません: {d["rel_path"]}'
        text = open(tp, encoding='utf-8').read()
        notes = self._vision_for(d['id'])
        if notes:
            text += '\n\n## 画像の読み取り結果（自動転記）\n' + notes
        if plain:
            text = strip_fmt(text)
        if grep:
            pieces, seen = [], 0
            for m in re.finditer(re.escape(grep), text, re.I):
                s = max(0, m.start() - 900)
                e = min(len(text), m.end() + 1500)
                pieces.append(text[s:e])
                seen += 1
                if seen >= 12:
                    break
            if not pieces:
                return (f"『{grep}』は {d['rel_path']} 内に見つかりません。"
                        f'（全体 {len(text)} 文字。grep なしで読むこともできます）')
            body = ('\n---\n'.join(pieces))[:int(limit)]
            return f"# {d['rel_path']}（『{grep}』該当箇所 {seen} 件）\n{body}"
        offset, limit = int(offset or 0), int(limit or 18000)
        body = text[offset:offset + limit]
        tail = ('\n\n…続きがあります。offset=%d を指定して続けて読んでください。'
                % (offset + limit)) if offset + limit < len(text) else ''
        return f"# {d['rel_path']}（{offset}-{offset + len(body)} / 全{len(text)}文字）\n{body}{tail}"

    def _vision_for(self, doc_id: str) -> str:
        rows = [e for e in self.images if e['doc_id'] == doc_id]
        out = []
        for e in rows:
            note = self.vision_notes.get(e['image_id'])
            if note:
                out.append(f"[{e['origin']} {e['locator']} image_id={e['image_id']}]\n{note}")
        return '\n\n'.join(out)

    def list_images(self, rel_path=None, project=None, **_) -> str:
        rows = self.images
        if rel_path:
            d = self._resolve_doc(rel_path)
            if not d:
                return f'ファイルが見つかりません: {rel_path}'
            rows = [e for e in rows if e['doc_id'] == d['id']]
        if project:
            keys = nkey(project)
            rows = [e for e in rows if e.get('project') and
                    (keys in nkey(e['project']) or nkey(e['project']) in keys)]
        if not rows:
            return '該当画像なし。'
        lines = [f'{len(rows)} 件の画像']
        for e in rows[:200]:
            lines.append(f"- image_id={e['image_id']}  {e['origin']} {e['locator']}  ({e['rel_path']})")
        return '\n'.join(lines)

    def view_image(self, image_id=None, rel_path=None, page=None, **_):
        entry = None
        if image_id:
            entry = next((e for e in self.images if e['image_id'] == image_id), None)
        if entry is None and rel_path:
            d = self._resolve_doc(rel_path)
            if d:
                rows = [e for e in self.images if e['doc_id'] == d['id']]
                if page:
                    entry = next((e for e in rows if e.get('page') == int(page)), None)
                elif rows:
                    entry = rows[0]
                if entry is None and rows:
                    entry = rows[0]
        if entry is None:
            return '画像が見つかりません。list_images で image_id を確認してください。'
        if not os.path.exists(entry['path']):
            return f'画像ファイルがありません: {entry["path"]}'
        caption = f"{entry['rel_path']} / {entry['origin']} {entry['locator']}"
        note = self.vision_notes.get(entry['image_id'])
        if not self.vision_capable:
            # a text-only answering model cannot receive the picture; hand back
            # the stored transcription instead of failing the call
            if note:
                return (f'[画像 {caption}]\n'
                        f'（このモデルは画像を直接読めないため、OCR結果を返します）\n{note}')
            return (f'このモデルは画像入力に対応していません（{caption}）。'
                    'この画像の書き起こしも未作成です。')
        return {'image': llm_mod.image_part(entry['path']), 'caption': caption,
                'note': note}

    def locate_in_pages(self, rel_path='', text='', **_) -> str:
        d = self._resolve_doc(rel_path)
        if not d:
            return f'ファイルが見つかりません: {rel_path}'
        rec = json.load(open(os.path.join(config.DOCS_DIR, f"{d['id']}.json"),
                             encoding='utf-8'))
        pages = rec.get('page_text')
        source = 'レイアウト実測（Word/PowerPoint のページ割り）'
        if not pages:
            pages = [{'page': p['page'], 'text': p['text']}
                     for p in (rec.get('pages') or [])]
            source = 'PDF のページ'
        if not pages:
            return (f"{d['rel_path']} にはページ情報がありません。"
                    'テキスト中の「## スライド N」などを read_document で確認してください。')

        def squash(s: str) -> str:
            return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s)).lower()

        needle = squash(text)
        hits = []
        for p in pages:
            body = squash(p['text'] or '')
            if needle and needle in body:
                raw = p['text'] or ''
                j = squash(raw).find(needle)
                hits.append((p['page'], raw[:0] or '', j))
        if not hits:
            return (f"『{text}』は {d['rel_path']} のページ本文に見つかりません"
                    f"（全{len(pages)}ページ）。表記を短くして再試行してください。")
        out = [f"{d['rel_path']}（全{len(pages)}ページ, 判定元={source}）",
               f"『{text}』が現れるページ: " + ', '.join(f'p.{h[0]}' for h in hits)]
        first = hits[0][0]
        body = next(p['text'] for p in pages if p['page'] == first)
        out.append(f'--- p.{first} の内容（先頭1200文字）---\n{body[:1200]}')
        return '\n'.join(out)

    def run_python(self, code='', **_) -> str:
        return pysandbox.run(code, self.root, RAG_DIR, artifacts=config.ARTIFACTS,
                             seed=config.SEED)

    def canonicalise_abbrev(self, question: str, answer: str) -> str:
        """When a question demands 主略称, replace any alias with the canonical one.

        The glossary names one 主略称 per project and lists the rest as 別名候補;
        answering with an alias violates the stated answer format.  The mapping
        is read from the drive at run time, so this stays a general rule rather
        than a table of names.
        """
        if not re.search(r'主略称|案件略称', question or ''):
            return answer
        out = answer
        for p in self.projects:
            ab = p.get('abbrev')
            if not ab:
                continue
            if re.search(rf'(?<![A-Za-z]){re.escape(ab)}(?![A-Za-z])', out):
                continue          # already canonical for this project
            names = [p.get('display'), p.get('official')] + (p.get('aliases') or [])
            names = [n for n in names if n and n != ab]
            # Match the longest run of project-name characters around the alias,
            # not the alias alone: replacing 「青葉」 inside 「青葉与信」 would
            # leave the stray tail 「与信」 glued to the abbreviation.
            best = None
            for name in sorted(names, key=len, reverse=True):
                if re.fullmatch(r'[A-Za-z0-9]+', name):
                    pat = rf'(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])'
                else:
                    # extend over adjacent CJK/word characters so partial
                    # aliases swallow the whole company-name fragment
                    pat = rf'[一-鿿ぁ-ヿA-Za-z0-9株式会社]*{re.escape(name)}[一-鿿ぁ-ヿA-Za-z0-9]*'
                m = re.search(pat, out)
                if m and (best is None or len(m.group(0)) > len(best.group(0))):
                    best = m
            if best:
                out = out[:best.start()] + ab + out[best.end():]
        return out

    def lookup_glossary(self, term=None, **_) -> str:
        if not term:
            lines = ['■ 案件略称']
            for p in self.projects:
                lines.append(f"- {p.get('abbrev')}: {p.get('official')} 別名={p.get('aliases')}")
            lines.append('■ 用語（抜粋）')
            for k, v in list(self.terms.items())[:400]:
                lines.append(f'- {k}: {", ".join(v)}')
            return '\n'.join(lines)
        t = term.strip()
        out = []
        for k, v in self.terms.items():
            if nkey(t) == nkey(k) or nkey(t) in nkey(k) or any(nkey(t) in nkey(x) for x in v):
                out.append(f'- {k}: {", ".join(v)}')
        for p in self.projects:
            names = [p.get('display'), p.get('official'), p.get('abbrev')] + (p.get('aliases') or [])
            if any(n and (nkey(t) == nkey(n) or nkey(t) in nkey(n)) for n in names):
                out.append(f"- 案件 {p.get('abbrev')}: {p.get('official')} "
                           f"(フォルダ: {p.get('folder')}, 別名: {p.get('aliases')})")
        return '\n'.join(out) if out else f'「{term}」は用語集に見つかりません。'


import re as _re

# A "couldn't find it" answer is correct behaviour when the evidence really is
# absent (0 points beats -1 for a guess), but it is pure loss when the search
# simply stopped too early.  These two constants separate the cases: giving up
# after only a handful of tool calls gets one push to keep looking.
GIVEUP_PAT = _re.compile(
    r'確認できません|わかりません|分かりません|見つかりません|特定できません|該当なし|不明です')
MIN_EFFORT_CALLS = 9        # below this, a give-up is treated as premature

KEEP_FULL_RESULTS = 4       # recent tool results kept verbatim
STUB_CHARS = 400            # older ones shrink to this much
MAX_TOOL_RESULT = 14000     # hard cap on a single tool result

# Wall-clock budget per question; 0 disables it.  Measured on a full run, a cap
# costs more than it saves: the long questions are mostly long because the agent
# is still making progress, so they finish correctly if left alone.
MAX_QUESTION_SECONDS = 0

# Comparing an old and a new revision means holding two large documents at once,
# which is what pushes a transcript past the context limit.  Rather than trust a
# fixed step count, keep the running total under an explicit character budget.
CONTEXT_CHAR_BUDGET = 60000


def _prune(messages: list[dict]) -> list[dict]:
    """Shrink old tool results before resending the conversation.

    The whole transcript is resent on every step, so full-size results would
    make token use grow with the square of the number of steps.  Recent results
    stay intact; older ones keep only a head plus a note that the tool can be
    called again, which bounds the growth to roughly linear.
    """
    idx = [i for i, m in enumerate(messages) if m['role'] == 'tool']
    stub_upto = set(idx[:-KEEP_FULL_RESULTS]) if len(idx) > KEEP_FULL_RESULTS else set()

    def shrink(m, keep):
        body = m['content'][:keep]
        return {**m, 'content': body + f'\n…(省略。全{len(m["content"])}文字。'
                                       '続きは offset/grep を指定して読み直してください)'}

    out = []
    for i, m in enumerate(messages):
        if i in stub_upto and isinstance(m.get('content'), str) and len(m['content']) > STUB_CHARS:
            out.append(shrink(m, STUB_CHARS))
        else:
            out.append(m)

    # Even the retained results can overflow together (two full documents in a
    # version-comparison question).  Trim the oldest survivors until the whole
    # transcript fits the budget, so the request is never rejected outright.
    def total(msgs):
        return sum(len(x['content']) for x in msgs if isinstance(x.get('content'), str))

    if total(out) > CONTEXT_CHAR_BUDGET:
        for i in [j for j, m in enumerate(out) if m['role'] == 'tool'][:-1]:
            if total(out) <= CONTEXT_CHAR_BUDGET:
                break
            if isinstance(out[i].get('content'), str) and len(out[i]['content']) > STUB_CHARS:
                out[i] = shrink(out[i], STUB_CHARS)
    return out


def answer_question(question: str, tb: Toolbox, client: llm_mod.Client,
                    max_steps: int = 26, log: list | None = None) -> dict:
    messages: list[dict] = [{'role': 'user', 'content': f'質問: {question}'}]
    final = None
    pushed_back = False
    started = time.time()
    out_of_time = False
    for step in range(max_steps):
        remaining = max_steps - step
        spent = time.time() - started
        # A wall-clock cap was tried and removed: of the questions that ran long,
        # most were long *because they were working* — 18 of 24 reached a real
        # answer past the five-minute mark.  Cutting them off traded points for
        # runtime.  MAX_QUESTION_SECONDS <= 0 disables the cap.
        if MAX_QUESTION_SECONDS > 0 and spent > MAX_QUESTION_SECONDS and not out_of_time:
            out_of_time = True
            messages.append({'role': 'user', 'content':
                             f'この設問に既に {spent / 60:.0f} 分を使いました。'
                             'これ以上の調査は打ち切ります。いま手元にある根拠だけで '
                             'submit_answer を呼んでください。'
                             '確証が得られている部分があればそれを答え、'
                             '答えに到達できていなければ「資料からは確認できませんでした」と'
                             '回答してください。'})
        elif remaining <= 3:
            messages.append({'role': 'user', 'content':
                             f'残りツール実行回数は {remaining} 回です。現時点の根拠で '
                             'submit_answer を呼んで回答を確定してください。'})
        messages = _prune(messages)
        # once the budget is spent, the only move left is to answer
        active_tools = ([t for t in TOOLS if t['name'] == 'submit_answer']
                        if out_of_time else TOOLS)
        try:
            resp = client.chat(messages, tools=active_tools, system=SYSTEM, max_tokens=4096)
        except llm_mod.LLMError as e:
            # Last-resort recovery: if the server still says the request is too
            # long, collapse every stored tool result and continue rather than
            # losing the question entirely.
            if 'context' not in str(e).lower():
                raise
            for i, m in enumerate(messages):
                if m['role'] == 'tool' and isinstance(m.get('content'), str):
                    messages[i] = {**m, 'content': m['content'][:600]
                                   + '\n…(文脈超過のため大幅に省略。必要なら読み直してください)'}
            messages.append({'role': 'user', 'content':
                             '文脈が上限に達したため、これまでのツール結果を要約状態に切り詰めました。'
                             '必要な箇所は grep や offset で範囲を絞って読み直し、'
                             '回答に必要な情報だけを集めてください。'})
            resp = client.chat(messages, tools=active_tools, system=SYSTEM, max_tokens=4096)
        calls = resp.get('tool_calls') or []
        if not calls:
            if resp.get('text'):
                final = {'answer': resp['text'].strip(), 'confidence': 'low',
                         'evidence': '(submit_answer 未使用)'}
                break
            messages.append({'role': 'user', 'content':
                             'ツールを使って調査を続けるか、submit_answer で回答してください。'})
            continue

        messages.append({'role': 'assistant', 'content': resp.get('text') or '',
                         'tool_calls': calls})
        for tc in calls:
            name, args = tc['name'], tc['arguments']
            if log is not None:
                log.append({'step': step, 'tool': name,
                            'args': {k: (v[:400] if isinstance(v, str) else v)
                                     for k, v in args.items()}})
            if name == 'submit_answer':
                ans = (args.get('answer') or '').strip()
                n_calls = len(log) if log is not None else step
                if (GIVEUP_PAT.search(ans) and n_calls < MIN_EFFORT_CALLS
                        and not pushed_back and step < max_steps - 4):
                    pushed_back = True
                    messages.append({'role': 'tool', 'tool_call_id': tc['id'],
                                     'content': (
                        'まだ調査が不十分です。この共有ドライブは、書式・色・数式・グラフの数値・'
                        '画像の読み取り結果まで抽出済みです。以下を試してから再度 submit_answer '
                        'してください。\n'
                        '1) read_document を offset を進めて最後まで読む（1回の読み込みで全文は返りません）\n'
                        '2) grep に別の語（言い換え・英語表記・略称・列名）を指定して読み直す\n'
                        '3) 複数案件にまたがる集計なら run_python で '
                        'each_text("01.契約") のように全案件を一括走査して計算する\n'
                        '4) 図・スキャンPDFなら list_images → view_image で該当ページを見る\n'
                        '注意: 探索を促していますが、推測での回答を求めているのではありません。'
                        '誤答は -1 点、「確認できない」は 0 点です。'
                        '質問が問うている対象（例:「支払った金額」と「契約金額」、'
                        '「実績」と「見込」）を取り違えたまま計算すると誤答になります。'
                        '根拠が資料上で直接確認できた場合にのみ具体的に答え、'
                        'そうでなければ確認できない旨を回答してください。\n'
                        'また、問われている事柄が見つからないときに、近そうな別の事柄'
                        '（別の指標・別の列・別の集計）で代用してはいけません。'
                        'それは無関係な回答として誤答になり、「確認できない」より低い評価です。')})
                    continue
                final = {'answer': tb.canonicalise_abbrev(question, ans),
                         'evidence': args.get('evidence') or '',
                         'confidence': args.get('confidence') or 'medium'}
                messages.append({'role': 'tool', 'tool_call_id': tc['id'],
                                 'content': '受理しました。'})
                break
            fn = getattr(tb, name, None)
            if fn is None:
                result = f'未知のツール: {name}'
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = f'ツール実行エラー: {type(e).__name__}: {e}'
            if isinstance(result, dict) and 'image' in result:
                messages.append({'role': 'tool', 'tool_call_id': tc['id'],
                                 'content': f"画像を表示します: {result['caption']}"})
                messages.append({'role': 'user', 'content': [
                    {'type': 'text', 'text': f"[画像] {result['caption']}"},
                    result['image']]})
            else:
                text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                if len(text) > MAX_TOOL_RESULT:
                    text = (text[:MAX_TOOL_RESULT]
                            + f'\n…(出力を切り詰めました。全{len(text)}文字。'
                              'offset や grep で範囲を絞って読み直してください)')
                messages.append({'role': 'tool', 'tool_call_id': tc['id'], 'content': text})
        if final:
            break
    if final is None:
        final = {'answer': '資料からは確認できませんでした。', 'confidence': 'low',
                 'evidence': 'ステップ上限に到達'}
    return final
