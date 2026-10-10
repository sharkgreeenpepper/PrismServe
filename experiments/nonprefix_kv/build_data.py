"""Deterministic tool/RAG traces; labels come from record state, never model output."""
import argparse
import hashlib
import json
import random
from pathlib import Path

KINDS = ('local_edit', 'document_reorder', 'tool_expired', 'branch_rollback', 'fact_version')
SEP = '\n # # \n'


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def build(tokenizer, split, sessions, lengths, output):
    rng = random.Random(20261008 if split == 'eval' else 20261009)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with (output / f'{split}.jsonl').open('w') as f:
        for kind in KINDS:
            for sid in range(sessions):
                session = f'{split}-{kind}-{sid:02d}'
                base = [rng.randrange(1000, 9000) for _ in range(16)]
                states = []
                for step in range(10):
                    values = base.copy()
                    order = list(range(16))
                    target = (sid * 7) % 16
                    revision = step
                    if kind == 'document_reorder':
                        rng.shuffle(order)
                    elif kind == 'branch_rollback':
                        revision = step if step < 5 else 9 - step
                        values[target] = base[target] + revision
                    else:
                        values[target] = base[target] + step
                    states.append((values, order, target, revision))
                for length in lengths:
                    for step, (values, order, target, revision) in enumerate(states):
                        docs = []
                        for i in order:
                            # Version IDs and distinct fillers prevent accidental inter-session KV sharing.
                            doc = f'Document {session}/record-{i:02d}. Entity ITEM-{i:02d}. '
                            if i == target and kind in ('fact_version', 'tool_expired'):
                                doc += f'Previous revision -1 amount {base[i] - 1:04d} is obsolete. '
                            doc += f'Current revision {revision if i == target else 0} amount {values[i]:04d}. '
                            filler = (f'Archive for record {i:02d}: delivery records were checked; '
                                      'the supporting notes describe packaging, inventory, and shipment procedures. ')
                            if kind == 'tool_expired' and i == target:
                                doc += f'Tool observation {step:02d}, replacing prior observation. '
                            docs.append(doc + filler * (20 if length == 8192 else 43))
                        system = ('You read tool records. Use only the current revision in the supplied records. '
                                  'Return only the four digit amount requested, without explanation.')
                        query = f'What is the current amount for ITEM-{target:02d}?'
                        # Padding is placed within the final document before the query, not after it.
                        def encode():
                            body = SEP.join(docs) + SEP + query
                            return tokenizer.apply_chat_template(
                                [{'role': 'system', 'content': system}, {'role': 'user', 'content': body}],
                                tokenize=True, add_generation_prompt=True, return_dict=False)
                        tokens = encode()
                        # Trim filler only; all authoritative records remain intact.
                        while len(tokens) > length:
                            j = max(range(16), key=lambda j: len(docs[j]))
                            cut = min(len(docs[j]) - 220, max(20, (len(tokens) - length) * 3))
                            if cut <= 0:
                                raise ValueError('Cannot fit authoritative records')
                            docs[j] = docs[j][:-cut]
                            tokens = encode()
                        # Add token-level inert padding immediately before query's separator.
                        # The input is supplied as token IDs, so exact lengths are reproducible.
                        pad_id = tokenizer.encode(' archive', add_special_tokens=False)[0]
                        rendered = tokenizer.apply_chat_template(
                            [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': SEP.join(docs) + SEP + query}],
                            tokenize=False, add_generation_prompt=True)
                        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
                        assert encoded['input_ids'] == tokens
                        query_start = rendered.rindex(query)
                        at = next(i for i, (_, end) in enumerate(encoded['offset_mapping'])
                                  if end > query_start)
                        padding = length - len(tokens)
                        tokens[at:at] = [pad_id] * padding
                        assert len(tokens) == length
                        # BPE may merge whitespace in separators. Use actual chat offsets instead.
                        boundaries = [0]
                        for i in order:
                            char_start = rendered.index(f'Document {session}/record-{i:02d}.')
                            boundaries.append(next(j for j,(_,end) in enumerate(encoded['offset_mapping'])
                                                   if end > char_start))
                        boundaries += [at + padding, len(tokens)]
                        row = {'request_id': f'{session}-{length}-{step:02d}', 'session_id': session,
                               'kind': kind, 'step': step, 'length': length, 'split': split,
                               'prompt_token_ids': tokens, 'input_hash': token_hash(tokens),
                               'answer': f'{values[target]:04d}', 'revision': revision,
                               'document_order': order, 'target_entity': target,
                               'document_spans': list(zip(boundaries[:-1], boundaries[1:]))}
                        f.write(json.dumps(row, separators=(',', ':')) + '\n')
                        count += 1
    (output / f'{split}-manifest.json').write_text(json.dumps(
        {'count': count, 'sessions_per_kind': sessions, 'lengths': lengths,
         'label_source': 'deterministic authoritative record state',
         'workload': 'synthetic structured tool/RAG; not end-to-end coding agent'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['eval', 'calibration'], default='eval')
    p.add_argument('--sessions', type=int, default=20)
    p.add_argument('--lengths', type=int, nargs='+', default=[8192, 16384])
    a = p.parse_args()
    from transformers import AutoTokenizer
    build(AutoTokenizer.from_pretrained(a.model, local_files_only=True),
          a.split, a.sessions, a.lengths, a.output)
