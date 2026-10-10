"""Model-aware trace, prompt and cache-candidate contracts for paired replay."""
from dataclasses import dataclass
import hashlib
import json

BLEND_SPECIAL_STR = ' # #'
# The trailing space prevents Qwen BPE from merging the final '#' with '\n'.
SEP = '\n' + BLEND_SPECIAL_STR + ' \n'


@dataclass(frozen=True)
class Geometry:
    model_type: str
    layers: int
    full_attention_layers: int
    kv_heads: int
    head_dim: int
    hybrid: bool

    @classmethod
    def from_config(cls, config):
        text = config.get('text_config', config)
        kind = text.get('model_type', config.get('model_type', 'unknown'))
        layers = int(text['num_hidden_layers'])
        heads = int(text['num_attention_heads'])
        dim = text.get('head_dim')
        if dim is None:
            if int(text['hidden_size']) % heads:
                raise ValueError('Cannot infer head dimension')
            dim = int(text['hidden_size']) // heads
        types = text.get('layer_types') or []
        hybrid = any(t != 'full_attention' for t in types) or kind.startswith(('qwen3_5', 'qwen3_next'))
        if hybrid and len(types) != layers:
            raise ValueError('Hybrid full-attention layer list must be explicit')
        full = sum(t == 'full_attention' for t in types) if types else layers
        return cls(kind, layers, full, int(text['num_key_value_heads']), int(dim), hybrid)

    def full_kv_bytes_per_token(self, element_bytes=2):
        return 2 * self.full_attention_layers * self.kv_heads * self.head_dim * element_bytes

    def require_blend(self):
        if self.hybrid:
            raise ValueError('Hybrid non-prefix state repair is out of scope')
        if self.model_type not in ('qwen2', 'qwen3'):
            raise ValueError('Model has no validated standard-Attention Blend adapter')


def validate_trace(row):
    """The schema records origin/labels; it does not infer truth from Full output."""
    for key in ('request_id', 'session_id', 'system', 'l1', 'l3', 'query', 'split', 'origin'):
        if key not in row:
            raise ValueError(f'Missing {key}')
    if row['split'] not in ('calibration', 'evaluation', 'profile'):
        raise ValueError('Unknown split')
    if row['origin'] not in ('real_redacted', 'synthetic'):
        raise ValueError('Trace origin must be explicit')
    public_view = row.get('memory_view') == 'bumi_public_context'
    if not isinstance(row['l1'], list) or not isinstance(row['l3'], list) or (len(row['l3']) > 5 and not public_view):
        raise ValueError('L1 messages / Top-5 memories expected')
    for message in row['l1']:
        if message.get('role') not in ('system', 'user', 'assistant', 'tool') or not isinstance(message.get('content'), str):
            raise ValueError('This text-only replay requires role/content messages')
    for memory in row['l3']:
        if not all(isinstance(memory.get(k), str) for k in ('id', 'version', 'text')):
            raise ValueError('Memory id/version/text must be explicit strings')
    for text in [row['system'], row['query'], *(m['content'] for m in row['l1']), *(m['text'] for m in row['l3'])]:
        if not isinstance(text, str) or BLEND_SPECIAL_STR in text:
            raise ValueError('Text must not contain the configured chunk delimiter')
    target = row.get('target')
    if target is not None:
        if target.get('type') not in ('exact', 'choice', 'task_success'):
            raise ValueError('Unsupported independent target')
        if 'value' not in target or not target.get('provenance') or target.get('derived_from_model_output') is not False:
            raise ValueError('Independent target provenance required')
    return row


@dataclass(frozen=True)
class Assembly:
    request_id: str
    prompt_token_ids: tuple
    input_hash: str
    segments: tuple
    query_span: tuple
    enable_thinking: bool
    l3_content_tokens: int

    def reusable_ranges(self, candidates):
        """Exclude Query before transfer; partial crossing chunks are rejected."""
        q0, q1 = self.query_span
        selected = []
        for start, end in candidates:
            if not 0 <= start < end <= len(self.prompt_token_ids):
                raise ValueError('Invalid candidate range')
            if start < q1 and end > q0:
                continue
            selected.append((start, end))
        return tuple(selected)


def assemble(row, tokenizer):
    validate_trace(row)
    pieces = ['Long-term memories in retrieval order:\n']
    char_starts = []
    for memory in row['l3']:
        pieces.append(SEP)
        char_starts.append(sum(map(len, pieces)))
        # Cache identity is internal. Public owner/event-time metadata retains
        # the meaning returned by Bumi without presenting invented Mem0 IDs.
        display = memory.get('display_metadata', {})
        metadata = json.dumps(display, ensure_ascii=False, sort_keys=True)
        pieces.append(f'Memory {metadata}:\n{memory["text"]}\n')
    pieces.append(SEP)
    query_char = sum(map(len, pieces))
    query_metadata = row.get('query_metadata')
    pieces.append('Current query:\n' + (json.dumps(query_metadata, ensure_ascii=False, sort_keys=True) + '\n' if query_metadata else '') + row['query'])
    body = ''.join(pieces)
    messages = [{'role': 'system', 'content': row['system']}, *[dict(m) for m in row['l1']],
                {'role': 'user', 'content': body}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    tokens = encoded['input_ids']
    expected = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                             add_generation_prompt=True, enable_thinking=False)
    if list(tokens) != list(expected):
        raise ValueError('Chat template/tokenizer identity mismatch')
    # The full current body is longer than its Query and occurs last. A repeated
    # Query or previous assembled body inside L1 cannot select an old suffix.
    base = rendered.rfind(body)
    if base < 0:
        raise ValueError('Tokenizer transformed current message content')
    offsets = encoded['offset_mapping']
    def token_at(char):
        return next(i for i, (_, end) in enumerate(offsets) if end > char)
    semantic_query_start = token_at(base + query_char)
    delimiter = tokenizer.encode(BLEND_SPECIAL_STR, add_special_tokens=False)
    matches = [i for i in range(len(tokens) - len(delimiter) + 1)
               if tokens[i:i + len(delimiter)] == delimiter]
    if len(matches) != len(row['l3']) + 1:
        raise ValueError('Configured LMCache delimiter is not stable in the actual token stream')
    chunks = []
    start = 0
    for match in matches:
        chunks.append((start, match));start = match + len(delimiter)
    chunks.append((start, len(tokens)))
    q0 = chunks[-1][0]
    if q0 > semantic_query_start:
        raise ValueError('Query candidate boundary misses current-query tokens')
    segments = [{'kind': 'system_l1', 'range': chunks[0], 'cacheable': True}]
    for i, memory in enumerate(row['l3']):
        segments.append({'kind': 'l3', 'id': memory['id'], 'version': memory['version'],
                         'range': chunks[i + 1], 'cacheable': True})
    segments.append({'kind': 'query_and_generation_suffix', 'range': (q0, len(tokens)), 'cacheable': False})
    digest = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
    content_tokens = sum(len(tokenizer.encode(m['text'], add_special_tokens=False)) for m in row['l3'])
    return Assembly(row['request_id'], tuple(tokens), digest, tuple(segments), (q0, len(tokens)), False, content_tokens)


def check_split_sessions(rows):
    seen = {}
    for row in rows:
        split = row['split'];session = row['session_id']
        if session in seen and seen[session] != split:
            raise ValueError('A session crosses calibration/evaluation/profile splits')
        seen[session] = split


def exact_repeat_fraction(previous, current, tokenizer):
    """Length-weighted exact memory-version/content repetition; not semantic similarity."""
    previous_set = {(m['id'], m['version'], m['text']) for m in previous}
    lengths = [len(tokenizer.encode(m['text'], add_special_tokens=False)) for m in current]
    total = sum(lengths)
    return sum(n for m, n in zip(current, lengths) if (m['id'], m['version'], m['text']) in previous_set) / total if total else None
