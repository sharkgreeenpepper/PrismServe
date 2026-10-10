"""Readonly adapters for Bumi_Mem v0.6 structured snapshots; no SDK/store init."""
import hashlib
import json
from pathlib import Path
import sqlite3


def convert_payload(*, request, response, agent_id, stream_id, system, split, origin, target=None):
    if set(response) != {'turn_id', 'current_user', 'long_term_memory', 'short_term_memory'}:
        raise ValueError('Expected current four-field Bumi context response')
    query = request.get('query')
    if query is None:
        return None  # Flush is not an inference request.
    if response['turn_id'] != request['turn_id'] or response['current_user'] != request['user_id']:
        raise ValueError('Request/response identity mismatch')
    short = response['short_term_memory']
    if not short or short[-1]['role'] != 'user' or short[-1]['content'] != query or short[-1]['speaker_id'] != response['current_user']:
        raise ValueError('Current Query must be the final matching short-term message')
    l1 = []
    for item in short[:-1]:
        # Preserve actor and original speech time as supplied data. Never merge
        # the current Query with a previous equal question.
        metadata = {k: item[k] for k in ('speaker_id', 'time')}
        l1.append({'role': item['role'], 'content': json.dumps(metadata, ensure_ascii=False, sort_keys=True) + '\n' + item['content']})
    memories = []
    for item in response['long_term_memory']:
        if set(item) != {'user_id', 'content', 'time'}:
            raise ValueError('Unexpected public memory fields')
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        identity = hashlib.sha256(payload.encode()).hexdigest()
        memories.append({'id': 'public-content:' + identity, 'version': 'content-addressed; source-version-unavailable',
                         'text': item['content'], 'display_metadata': {'user_id': item['user_id'], 'time': item['time']}})
    return {'request_id': request['turn_id'], 'session_id': stream_id, 'l1_scope_id': agent_id,
            'system': system, 'l1': l1, 'l3': memories, 'query': query,
            'query_metadata': {k: short[-1][k] for k in ('speaker_id', 'time')},
            'split': split, 'origin': origin, 'target': target,
            'memory_view': 'bumi_public_context', 'mem0_top5_observed_separately': False,
            'memory_identity_origin': 'public content/owner/event-time; not Mem0 id/version'}


def cached_contexts_readonly(database):
    """Read existing cached responses without constructing migratory SessionStore."""
    path = Path(database).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        conn.execute('PRAGMA query_only=ON')
        conn.row_factory = sqlite3.Row
        columns = {r['name'] for r in conn.execute('PRAGMA table_info(context_requests)')}
        required = {'request_id', 'agent_id', 'user_id', 'query', 'response_json', 'created_at'}
        if not required <= columns:
            raise ValueError('Database has no compatible structured context cache')
        for row in conn.execute('SELECT request_id,agent_id,user_id,query,response_json,created_at '
                                'FROM context_requests WHERE response_json IS NOT NULL ORDER BY created_at,request_id'):
            response = json.loads(row['response_json'])
            if isinstance(response, dict) and set(response) == {'turn_id', 'current_user', 'long_term_memory', 'short_term_memory'}:
                yield {'request': {'turn_id': response['turn_id'], 'user_id': row['user_id'], 'query': row['query']},
                       'response': response, 'agent_id': row['agent_id'], 'created_at': row['created_at']}
    finally:
        conn.close()
