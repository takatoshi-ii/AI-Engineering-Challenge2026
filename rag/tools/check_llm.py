"""Verify the configured model endpoint can do everything the pipeline needs.

Checks, in order: reachability, Japanese generation, tool calling, and vision.
A backend that fails the tool-calling or vision check cannot run the agent.
"""
import base64, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.stdout.reconfigure(encoding='utf-8')
import config
import llm as llm_mod

print('=== 設定 ===')
for k in ('OPENAI_BASE_URL', 'RAG_MODEL', 'RAG_PROVIDER'):
    print(f'  {k:18s} = {os.environ.get(k) or "(未設定)"}')
for k in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
    v = os.environ.get(k)
    print(f'  {k:18s} = {"設定あり(" + v[:6] + "...)" if v else "(未設定)"}')

try:
    client = llm_mod.Client(use_cache=False)
except llm_mod.LLMError as e:
    print(f'\n[NG] {e}')
    sys.exit(1)
print(f'\nprovider={client.provider} model={client.model} local={client.is_local}')

ok = True


def step(name, fn):
    global ok
    t0 = time.time()
    try:
        detail = fn()
        print(f'  PASS  {name:22s} ({time.time() - t0:.1f}s)  {detail}')
    except Exception as e:
        ok = False
        print(f'  FAIL  {name:22s} ({time.time() - t0:.1f}s)  {type(e).__name__}: {str(e)[:180]}')


print('\n=== 能力チェック ===')


def t_text():
    r = client.chat([{'role': 'user', 'content': '「動作確認」とだけ返してください。'}],
                    max_tokens=64)
    return repr((r['text'] or '').strip()[:40])


def t_tools():
    tools = [{'name': 'add', 'description': '2つの整数を足す',
              'parameters': {'type': 'object',
                             'properties': {'a': {'type': 'integer'},
                                            'b': {'type': 'integer'}},
                             'required': ['a', 'b']}}]
    r = client.chat([{'role': 'user', 'content': '17と25を足してください。必ずaddツールを使うこと。'}],
                    tools=tools, max_tokens=256)
    calls = r.get('tool_calls') or []
    if not calls:
        raise RuntimeError(f'ツールを呼びませんでした: {r["text"][:80]!r}')
    return f"{calls[0]['name']}({calls[0]['arguments']})"


def t_vision():
    imgs = json.load(open(os.path.join(config.ARTIFACTS, 'images.json'), encoding='utf-8'))
    target = next(e for e in imgs if e['origin'] == 'pdf_page')
    r = client.chat([{'role': 'user', 'content': [
        {'type': 'text', 'text': 'この画像に書かれている文字を最初の30文字だけ書き出してください。'},
        llm_mod.image_part(target['path'])]}], max_tokens=200)
    txt = (r['text'] or '').strip()
    if not txt:
        raise RuntimeError('空の応答')
    return repr(txt[:50])


step('テキスト生成(日本語)', t_text)
step('ツール呼び出し', t_tools)
step('画像入力(必須)', t_vision)

print('\n' + ('この構成で実行可能です。' if ok else
              '不合格の項目があります。ツール呼び出しと画像入力の両方に対応したモデルが必要です。'))
sys.exit(0 if ok else 1)
