"""Bounded, independently labelled synthetic memory workloads, never production.

Top-5 sizes refer to total memory content tokens, not per memory. Nominal
repetition is a count of unchanged slots; measured token-weighted repetition
is saved by prepare.py and can differ slightly from the nominal value.
"""
import argparse
import copy
import json
from pathlib import Path
import random

ANCHORS = [
    (1024, 128, 1.0), (1024, 512, .6), (1024, 2048, .2), (1024, 4096, 1.0),
    (4096, 2048, .6), (8192, 2048, .6), (16384, 2048, .6), (4096, 4096, 0.0)]


def pad(tokenizer, prefix, target):
    base = tokenizer.encode(prefix, add_special_tokens=False)
    if len(base) > target:
        raise ValueError('Fact exceeds requested content budget')
    filler = tokenizer.encode('背景记录：常规事项已经归档。', add_special_tokens=False)
    extra = (filler * ((target - len(base)) // len(filler) + 2))[:target - len(base)]
    text = prefix + tokenizer.decode(extra, skip_special_tokens=False)
    if abs(len(tokenizer.encode(text, add_special_tokens=False)) - target) > 4:
        raise ValueError('Unexpected padding/tokenizer seam')
    return text


def generate(tokenizer, split, seed, sessions, steps):
    rng = random.Random(seed)
    rows = []
    for anchor, (l1_tokens, l3_tokens, repetition) in enumerate(ANCHORS):
        for scenario in ('stable_l1', 'growing_l1', 'memory_reorder'):
            for session in range(sessions):
                sid = f'{split}-s{seed}-a{anchor}-{scenario}-{session}'
                project = f'P{rng.randrange(10000,99999)}'
                bulk = pad(tokenizer, '会话背景：以下归档记录与项目金额无关。', l1_tokens)
                previous = None; previous_values = None; history = []
                for step in range(steps):
                    keep = {(step + j) % 5 for j in range(round(5 * repetition))}
                    memories = []; values = []
                    for slot in range(5):
                        if previous is not None and slot in keep:
                            memory = copy.deepcopy(previous[slot]);value = previous_values[slot]
                        else:
                            value = rng.randrange(1000,10000)
                            text = pad(tokenizer, f'项目{project}-{slot}当前有效金额为{value}。',
                                       l3_tokens // 5 + (slot < l3_tokens % 5))
                            memory = {'id': f'{sid}:memory:{slot}', 'version': f'v{step}', 'text': text}
                        memories.append(memory);values.append(value)
                    previous = copy.deepcopy(memories);previous_values = list(values)
                    displayed = copy.deepcopy(memories)
                    if scenario == 'memory_reorder':
                        rng.shuffle(displayed)
                    l1 = [{'role': 'user', 'content': bulk}]
                    if scenario == 'growing_l1':l1.extend(copy.deepcopy(history))
                    query = f'执行轮次{step}：项目{project}-0当前有效金额是多少？只输出四位数字。'
                    rows.append({'request_id': f'{sid}:{step}', 'session_id': sid,
                        'system': '读取当前长期记忆中的有效项目事实。历史回答可能过期。按当前查询只输出四位数字。',
                        'l1': l1, 'l3': displayed, 'query': query, 'split': split, 'origin': 'synthetic',
                        'target': {'type': 'exact', 'value': str(values[0]),
                                   'provenance': 'independent generator state, serialized before inference',
                                   'derived_from_model_output': False},
                        'workload': {'anchor': anchor, 'scenario': scenario, 'nominal_l1_content_tokens': l1_tokens,
                                     'nominal_l3_total_content_tokens': l3_tokens,
                                     'nominal_repeated_slot_fraction': repetition, 'seed': seed}})
                    # Fixed, scripted history keeps every arm's logical input
                    # identical. This is not an autonomous Agent rollout.
                    history.extend([{'role': 'user', 'content': query},
                                    {'role': 'assistant', 'content': str(values[0])}])
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('calibration', 'evaluation', 'profile'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--sessions', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.sessions < 1 or args.steps < 2:raise ValueError('Positive sessions and >=2 steps required')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = generate(tokenizer, args.split, args.seed, args.sessions, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(json.dumps({'requests': len(rows), 'sessions': len({r['session_id'] for r in rows}),
                      'origin': 'synthetic', 'split': args.split, 'seed': args.seed}))
