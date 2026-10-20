# AI-Engineering-Challenge2026

～RAG（検索拡張生成）によって煩雑な社内ドライブをハックせよ～

SIGNATE「AI Engineering Challenge」提出システム。未加工の社内共有ドライブと質問文のみを入力として、設問に自動回答する RAG パイプラインです。

設問固有の答え・参照ファイル・ページ番号・セル番地はコードに一切書かれていません。案件略称、パスワード導出規則、決裁基準といった社内ルールも、実行時にドライブ内の該当文書を読んで解決します。未知の案件フォルダや資料が追加されても同じ処理で動きます。

**回答生成は完全にローカルLLMで完結**させており、外部APIへのデータ送出はありません。

---

## 構成

```
共有ドライブ → 形式別の抽出 → BM25索引 → エージェント（道具9種）→ 回答
```

| ディレクトリ | 内容 |
|---|---|
| `rag/` | パイプライン本体 |
| `rag/dax/` | 形式別の抽出器（docx / pptx / xlsx / pdf / ipynb / テキスト ほか） |
| `rag/tools/` | 診断・監査・採点連携のスクリプト |
| `docs/` | 設計と検証の記録 |

---

## 必要なもの

- Python 3.10 以降
- 調査対象のドキュメント群（後述。コンペデータは含んでいません）
- OpenAI 互換のAPIを話すモデルサーバ、または OpenAI / Anthropic のAPIキー

Windows で `.docx` / `.pptx` のページ割りをレイアウト実測する場合のみ、Microsoft Office が必要です。無い環境では該当処理を飛ばして動作します。

```bash
pip install -r rag/requirements.txt
```

---

## データについて

**対象データはこのリポジトリに含まれていません。** コンペ主催者が配布したものであり、再配布しないためです。コンペ終了に伴い配布も終了しているため、現在は入手できません。

したがって**このリポジトリをクローンしても、そのままでは実行できません。** 実装の記録としてお読みください。

同じ構造のデータを用意すれば動作します。想定する配置は次のとおりです。

```
share/
├── 共有ドライブ/      ← 調査対象の本体（任意の業務フォルダ）
│   └── プロジェクト/
└── 質問回答/
    └── questions_test.csv    ← index,question の2列
```

ドライブ本体のフォルダ名は固定していません。`プロジェクト` を含むディレクトリを自動で探索します（NFD正規化されたフォルダ名にも対応）。設問ファイルは `index,question` の2列を持つCSVであれば、内容を問いません。

---

## モデルの指定

環境変数で切り替えます。いずれか一つを設定してください。

**ローカルのモデルサーバを使う場合（推奨）**

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama / vLLM / LM Studio など
export RAG_MODEL=qwen3.8:27b                       # サーバが提供するモデル名
export RAG_TIMEOUT=900                             # 1回の呼び出しの上限秒数（既定420）
```

`RAG_TIMEOUT` は**1問あたりではなく1回のLLM呼び出し**の上限です。複数の資料を横断して集計する設問は文脈が大きくなり、既定値では足りないことがあります。

**外部APIを使う場合**

```bash
export OPENAI_API_KEY=...        # または
export ANTHROPIC_API_KEY=...
```

設定が正しいかは次で確認できます。

```bash
python rag/tools/check_llm.py
```

---

## 実行手順

`rag/artifacts/` 以下の中間生成物はすべて実行時に自動で作られます。あらかじめ用意する必要はありません。

```bash
# 1. 全ファイルを形式別に抽出（書式・色・数式・グラフ数値・ページ割りを含む）
python rag/corpus.py

# 2. 画像を台帳に集約
python rag/images.py

# 3. 画像をOCRで本文化（視覚対応モデルが必要。飛ばしても動作します）
python rag/vision.py

# 4. BM25索引を構築
python rag/index.py

# 5. 回答を生成し、提出物を書き出す
python rag/run.py --questions test
```

完了すると `submit/predictions.csv` と `submit/submission.zip` が生成されます。

### 主なオプション

```bash
python rag/corpus.py --force        # 抽出をやり直す
python rag/corpus.py --no-render    # Office によるページ割り実測を行わない

python rag/vision.py --workers 6 --model <視覚モデル名>

python rag/run.py --questions valid # 検証セット（正解あり）で実行
python rag/run.py --workers 3       # 並列数。帯域律速の機材では3前後が有効
python rag/run.py --limit 5         # 先頭N問だけ試す
python rag/run.py --tag v2          # 結果の保存先を分ける
python rag/run.py --force           # 回答済みも解き直す
```

実行中の進捗は別のシェルから確認できます。

```bash
python rag/tools/progress.py <tag> test
```

---

## 検証

```bash
# パイプラインの健全性確認（モデル不要）
python rag/tools/selfcheck.py

# 提出物の形式検査（行数・index網羅・空欄・改行混入・トークン数）
python rag/tools/validate_submission.py

# 回答の形式監査（設問が要求する形式と食い違う回答を検出）
python rag/tools/audit_answers.py --tag <tag> --questions test

# 正解データとの機械的な照合（LLMの判定に依存しない）
python rag/tools/strict_eval.py --tag <tag> --questions valid
```

### 同梱していないものに依存するツール

`rag/tools/official_eval.py` は、主催者が配布した採点スクリプト（`evaluation/crag.py` と正解データ `evaluation/data/valid_txt.csv`）を呼び出します。**これらは再配布しないため同梱しておらず、このツールは動作しません。**

同等の仕組みを用意する場合は、リポジトリ直下に次の構成を置いてください。

```
evaluation/
├── crag.py                 採点の実行スクリプト
├── data/valid_txt.csv      正解データ
├── submit/                 採点対象の書き出し先
└── result/scoring.csv      判定結果
```

`official_eval.py` は、保存済みの回答を `evaluation/submit/predictions.csv` に書き出し、`crag.py` を実行して `result/scoring.csv` を読み戻す、という流れで動きます。採点にのみ外部APIを使用し、回答生成には使いません。

---

## コンペ規約への適合

禁止事項との対応は [`rag/README.md`](rag/README.md) に記載しています。設問固有の値がコードに書かれていないことは、次のコマンドで検査できます。

```bash
grep -rnE "案件名|略称|[0-9]{1,3},[0-9]{3},[0-9]{3}" rag --include=*.py | grep -v rag/tools/
```

---

## 記録

設計判断・実装・検証の詳細、および効かなかった施策の測定結果は [`docs/`](docs/) にまとめています。

---

## 公開について

本リポジトリのソースコードは、コンペの情報公開ポリシーが公開を認めていることを確認したうえで公開しています。

一方、主催者から配布されたデータ・評価スクリプト・説明資料は、再配布を避けるため含めていません。
