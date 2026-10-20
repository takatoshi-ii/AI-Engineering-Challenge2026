"""Model access layer.

Provider-agnostic so the pipeline can run against either an OpenAI-compatible
endpoint or Anthropic.  Every call is cached on disk keyed by the full request,
which keeps re-runs reproducible and cheap — a requirement of the competition
("re-running should reproduce equivalent answers").
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

try:                       # use the OS trust store; corporate TLS interception
    import truststore      # otherwise breaks certificate validation
    truststore.inject_into_ssl()
except Exception:
    pass

CACHE_DIR = os.path.join(config.ARTIFACTS, 'llmcache')


class LLMError(RuntimeError):
    pass


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:32]


_THINK = re.compile(r'<(think|thinking|reasoning)>.*?</\1>', re.S | re.I)
_OPEN_THINK = re.compile(r'<(think|thinking|reasoning)>.*\Z', re.S | re.I)


def strip_reasoning(text: str) -> str:
    """Remove chain-of-thought blocks from a reply.

    Reasoning models (Qwen3 and friends) emit ``<think>…</think>`` inline when
    served through an OpenAI-compatible shim.  Left in place it would be treated
    as part of the answer and submitted verbatim.
    """
    if not text or '<' not in text:
        return text
    out = _THINK.sub('', text)
    out = _OPEN_THINK.sub('', out)      # unterminated block (hit the token cap)
    return out.strip()


class Client:
    """Thin wrapper exposing one `chat` method with tool-calling and images."""

    def __init__(self, model: str | None = None, provider: str | None = None,
                 use_cache: bool = True, temperature: float = 0.0,
                 seed: int | None = None):
        self.provider = provider or os.environ.get('RAG_PROVIDER') or self._autodetect()
        self.model = model or os.environ.get('RAG_MODEL') or self._default_model()
        self.use_cache = use_cache
        self.temperature = temperature
        self.seed = config.SEED if seed is None else seed
        self._client = None
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------- provider
    @staticmethod
    def _autodetect() -> str:
        # a base URL wins: it is how a local server (vLLM / Ollama / LM Studio)
        # or a corporate gateway is pointed at, and it needs no vendor key
        if os.environ.get('OPENAI_BASE_URL'):
            return 'openai'
        if os.environ.get('ANTHROPIC_API_KEY'):
            return 'anthropic'
        if os.environ.get('OPENAI_API_KEY'):
            return 'openai'
        raise LLMError(
            'No model endpoint configured. Set one of:\n'
            '  OPENAI_API_KEY        (OpenAI)\n'
            '  ANTHROPIC_API_KEY     (Anthropic)\n'
            '  OPENAI_BASE_URL       (local server or gateway, e.g. '
            'http://localhost:8000/v1) plus RAG_MODEL')

    def _default_model(self) -> str:
        if os.environ.get('OPENAI_BASE_URL'):
            raise LLMError('Set RAG_MODEL to the model name served by '
                           f"{os.environ['OPENAI_BASE_URL']}")
        return {'anthropic': 'claude-sonnet-5',
                'openai': 'gpt-5'}[self.provider]

    @property
    def is_local(self) -> bool:
        url = os.environ.get('OPENAI_BASE_URL', '')
        return any(h in url for h in ('localhost', '127.0.0.1', '0.0.0.0', '::1'))

    def _ensure(self):
        if self._client is not None:
            return self._client
        if self.provider == 'anthropic':
            from anthropic import Anthropic
            self._client = Anthropic()
        else:
            from openai import OpenAI
            kw = {}
            base = os.environ.get('OPENAI_BASE_URL')
            if base:
                kw['base_url'] = base
                # local servers ignore the key but the SDK requires one
                kw['api_key'] = os.environ.get('OPENAI_API_KEY') or 'local'
                # A local server can wedge with the HTTP port still answering,
                # so a request must not be allowed to block the run forever.
                kw['timeout'] = float(os.environ.get('RAG_TIMEOUT', '420'))
                kw['max_retries'] = 0        # retries are handled in chat()
            self._client = OpenAI(**kw)
        return self._client

    # ------------------------------------------------------------------ API
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 4096, system: str | None = None) -> dict:
        """messages: [{role, content}] where content is a string or a list of
        parts ({'type':'text'|'image', ...}).  Returns a normalised dict:
        {'text': str, 'tool_calls': [{'id','name','arguments'}], 'raw': ...}
        """
        payload = {'provider': self.provider, 'model': self.model,
                   'messages': messages, 'tools': tools, 'system': system,
                   'max_tokens': max_tokens, 'temperature': self.temperature,
                   'seed': self.seed}
        key = _cache_key(payload)
        cpath = os.path.join(CACHE_DIR, f'{key}.json')
        if self.use_cache and os.path.exists(cpath):
            with open(cpath, encoding='utf-8') as fh:
                return json.load(fh)

        last = None
        msgs = messages
        for attempt in range(5):
            try:
                out = (self._chat_anthropic if self.provider == 'anthropic'
                       else self._chat_openai)(msgs, tools, max_tokens, system)
                # Never cache an empty reply.  A model that returns nothing has
                # failed, and storing that would make the failure permanent:
                # every retry would hit the cache and "fail" instantly.
                if (out.get('text') or '').strip() or out.get('tool_calls'):
                    with open(cpath, 'w', encoding='utf-8') as fh:
                        json.dump(out, fh, ensure_ascii=False)
                return out
            except Exception as e:            # transient API/network failures
                last = e
                msg = str(e).lower()
                if any(k in msg for k in ('invalid_api_key', 'authentication',
                                          'not found', 'does not exist')):
                    raise LLMError(f'{type(e).__name__}: {e}') from e
                # Some local servers reject a transcript whose final turn is a
                # tool result ("no user query found in messages").  Waiting does
                # not help — the request has to be reshaped, so close the turn
                # with an explicit user message and try again immediately.
                # A wedged local server answers /api/version but never returns
                # from a generation; the client timeout fires instead.  Give it
                # a chance to recover, but do not sit through five long waits.
                if any(k in msg for k in ('timeout', 'timed out', 'read operation')):
                    if attempt >= 1:
                        raise LLMError(f'model server unresponsive: {e}') from e
                    time.sleep(20)
                    continue
                if 'no user query' in msg and msgs is not None:
                    msgs = list(msgs) + [{
                        'role': 'user',
                        'content': '上記のツール実行結果を踏まえて、調査を続けるか '
                                   'submit_answer で回答を確定してください。'}]
                    continue
                time.sleep(min(2 ** attempt * 2, 30))
        raise LLMError(f'LLM call failed after retries: {last}')

    # ------------------------------------------------------------- backends
    @staticmethod
    def _parts_openai(content) -> list | str:
        if isinstance(content, str):
            return content
        out = []
        for p in content:
            if p['type'] == 'text':
                out.append({'type': 'text', 'text': p['text']})
            elif p['type'] == 'image':
                b64 = p['data'] if isinstance(p['data'], str) else base64.b64encode(p['data']).decode()
                out.append({'type': 'image_url',
                            'image_url': {'url': f"data:{p.get('mime', 'image/png')};base64,{b64}"}})
        return out

    def _chat_openai(self, messages, tools, max_tokens, system) -> dict:
        client = self._ensure()
        msgs = []
        if system:
            msgs.append({'role': 'system', 'content': system})
        for m in messages:
            if m['role'] == 'tool':
                msgs.append({'role': 'tool', 'tool_call_id': m['tool_call_id'],
                             'content': m['content']})
            elif m['role'] == 'assistant' and m.get('tool_calls'):
                msgs.append({'role': 'assistant', 'content': m.get('content') or None,
                             'tool_calls': [
                                 {'id': tc['id'], 'type': 'function',
                                  'function': {'name': tc['name'],
                                               'arguments': json.dumps(tc['arguments'],
                                                                       ensure_ascii=False)}}
                                 for tc in m['tool_calls']]})
            else:
                msgs.append({'role': m['role'], 'content': self._parts_openai(m['content'])})

        kw: dict = {'model': self.model, 'messages': msgs}
        kw['max_completion_tokens'] = max_tokens
        # determinism knobs; silently dropped below if the model rejects them
        if self.seed is not None:
            kw['seed'] = self.seed
        if self.temperature is not None:
            kw['temperature'] = self.temperature
        if tools:
            kw['tools'] = [{'type': 'function',
                            'function': {'name': t['name'], 'description': t['description'],
                                         'parameters': t['parameters']}} for t in tools]
            if self.is_local:
                # open-weight models are far more reliable issuing one call at
                # a time than a batch they then have to keep straight
                kw['parallel_tool_calls'] = False
        for _ in range(4):
            try:
                resp = client.chat.completions.create(**kw)
                break
            except Exception as e:
                msg = str(e)
                # models differ in which sampling parameters they accept; drop
                # the offending one and retry rather than failing the run
                if 'max_completion_tokens' in msg and 'max_completion_tokens' in kw:
                    kw.pop('max_completion_tokens')
                    kw['max_tokens'] = max_tokens
                elif "'temperature'" in msg and 'temperature' in kw:
                    kw.pop('temperature')
                elif "'seed'" in msg and 'seed' in kw:
                    kw.pop('seed')
                elif 'parallel_tool_calls' in msg and 'parallel_tool_calls' in kw:
                    kw.pop('parallel_tool_calls')
                else:
                    raise
        else:
            raise LLMError('could not find an accepted parameter set')
        choice = resp.choices[0].message
        calls = []
        for tc in (choice.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or '{}')
            except json.JSONDecodeError:
                args = {'_raw': tc.function.arguments}
            calls.append({'id': tc.id, 'name': tc.function.name, 'arguments': args})
        return {'text': strip_reasoning(choice.content or ''), 'tool_calls': calls,
                'usage': {'in': getattr(resp.usage, 'prompt_tokens', 0),
                          'out': getattr(resp.usage, 'completion_tokens', 0)}}

    @staticmethod
    def _parts_anthropic(content) -> list:
        if isinstance(content, str):
            return [{'type': 'text', 'text': content}]
        out = []
        for p in content:
            if p['type'] == 'text':
                out.append({'type': 'text', 'text': p['text']})
            elif p['type'] == 'image':
                b64 = p['data'] if isinstance(p['data'], str) else base64.b64encode(p['data']).decode()
                out.append({'type': 'image',
                            'source': {'type': 'base64',
                                       'media_type': p.get('mime', 'image/png'),
                                       'data': b64}})
        return out

    def _chat_anthropic(self, messages, tools, max_tokens, system) -> dict:
        client = self._ensure()
        msgs = []
        for m in messages:
            if m['role'] == 'tool':
                msgs.append({'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': m['tool_call_id'],
                     'content': m['content']}]})
            elif m['role'] == 'assistant' and m.get('tool_calls'):
                blocks = []
                if m.get('content'):
                    blocks.append({'type': 'text', 'text': m['content']})
                for tc in m['tool_calls']:
                    blocks.append({'type': 'tool_use', 'id': tc['id'],
                                   'name': tc['name'], 'input': tc['arguments']})
                msgs.append({'role': 'assistant', 'content': blocks})
            else:
                msgs.append({'role': m['role'], 'content': self._parts_anthropic(m['content'])})
        # Anthropic exposes no seed parameter; temperature 0 is the available
        # determinism control and the model name is recorded with the run.
        kw: dict = {'model': self.model, 'messages': msgs, 'max_tokens': max_tokens,
                    'temperature': self.temperature}
        if system:
            # the instructions are identical on every step -> cache them
            kw['system'] = [{'type': 'text', 'text': system,
                             'cache_control': {'type': 'ephemeral'}}]
        if tools:
            kw['tools'] = [{'name': t['name'], 'description': t['description'],
                            'input_schema': t['parameters']} for t in tools]
        # the transcript only ever grows, so marking its tail caches everything
        # before it and the next step reuses that prefix
        if len(msgs) > 2 and isinstance(msgs[-1].get('content'), list) and msgs[-1]['content']:
            msgs[-1]['content'][-1] = {**msgs[-1]['content'][-1],
                                       'cache_control': {'type': 'ephemeral'}}
        resp = client.messages.create(**kw)
        text, calls = '', []
        for block in resp.content:
            if block.type == 'text':
                text += block.text
            elif block.type == 'tool_use':
                calls.append({'id': block.id, 'name': block.name,
                              'arguments': dict(block.input)})
        return {'text': strip_reasoning(text), 'tool_calls': calls,
                'usage': {'in': resp.usage.input_tokens, 'out': resp.usage.output_tokens}}


def image_part(path: str) -> dict:
    with open(path, 'rb') as fh:
        data = fh.read()
    mime = 'image/png' if path.lower().endswith('.png') else 'image/jpeg'
    return {'type': 'image', 'mime': mime, 'data': base64.b64encode(data).decode()}
