import copy
import unittest
from contracts import Geometry, check_split_sessions, validate_trace
from timing import interval_union_ns, summarize_host
from bumi_adapter import convert_payload, cached_contexts_readonly
import json
import sqlite3
import tempfile
from pathlib import Path


def trace():
    return {'request_id': 'r', 'session_id': 's', 'system': 'Use current facts',
            'l1': [], 'l3': [], 'query': 'What is current?', 'split': 'evaluation', 'origin': 'synthetic'}


class ContractTests(unittest.TestCase):
    def test_hybrid_state_is_not_treated_as_full_kv(self):
        config = {'text_config': {'model_type': 'qwen3_5_text', 'num_hidden_layers': 32,
                  'num_attention_heads': 16, 'num_key_value_heads': 4, 'head_dim': 256,
                  'layer_types': ['linear_attention'] * 3 + ['full_attention']}}
        config['text_config']['layer_types'] *= 8
        g = Geometry.from_config(config)
        self.assertEqual(g.full_attention_layers, 8)
        self.assertEqual(g.full_kv_bytes_per_token(), 32768)
        with self.assertRaises(ValueError):g.require_blend()

    def test_qwen3_geometry_does_not_reuse_7b_constants(self):
        g = Geometry.from_config({'model_type': 'qwen3', 'num_hidden_layers': 36,
                                 'hidden_size': 4096, 'num_attention_heads': 32, 'num_key_value_heads': 8})
        self.assertEqual(g.full_kv_bytes_per_token(), 147456)
        g.require_blend()

    def test_model_generated_reference_cannot_be_quality_truth(self):
        r = trace();r['target'] = {'type': 'exact', 'value': '1', 'provenance': 'full output', 'derived_from_model_output': True}
        with self.assertRaises(ValueError):validate_trace(r)
        self.assertIsNone(validate_trace(trace()).get('target'))

    def test_calibration_and_evaluation_do_not_share_sessions(self):
        a = trace();b = copy.deepcopy(a);b['split'] = 'calibration'
        with self.assertRaises(ValueError):check_split_sessions([a, b])

    def test_overlap_does_not_inflate_ttft_attribution(self):
        events = [{'name': 'kv_transfer', 'clock': 'host_monotonic', 'start_ns': 0, 'end_ns': 8_000_000},
                  {'name': 'selective_recompute', 'clock': 'host_monotonic', 'start_ns': 3_000_000, 'end_ns': 10_000_000},
                  {'name': 'first_decode_forward', 'clock': 'host_monotonic', 'start_ns': 10_000_000, 'end_ns': 12_000_000}]
        result = summarize_host(events, 0, 10_000_000)
        self.assertEqual(result['instrumented_union_host_ms'], 10)
        self.assertEqual(result['ttft_ms'], 10)
        self.assertIsNone(result['gpu_compute_ms'])

    def test_reversed_interval_is_invalid(self):
        with self.assertRaises(ValueError):interval_union_ns([{'start_ns': 2, 'end_ns': 1}])

    def test_bumi_current_query_is_split_once_without_dropping_old_equal_question(self):
        query = '同一个问题'
        response = {'turn_id': 't2', 'current_user': 'A', 'long_term_memory': [],
                    'short_term_memory': [{'role': 'user', 'speaker_id': 'A', 'content': query, 'time': 'old'},
                                          {'role': 'assistant', 'speaker_id': 'bot', 'content': '旧回答', 'time': 'old'},
                                          {'role': 'user', 'speaker_id': 'A', 'content': query, 'time': 'now'}]}
        r = convert_payload(request={'turn_id': 't2', 'user_id': 'A', 'query': query}, response=response,
                            agent_id='bot', stream_id='capture', system='system', split='profile', origin='synthetic')
        self.assertEqual(len(r['l1']), 2)
        self.assertIn(query, r['l1'][0]['content'])
        self.assertEqual(r['query'], query)
        self.assertEqual(r['l1_scope_id'], 'bot')

    def test_public_long_term_context_is_not_silently_truncated_to_five(self):
        response = {'turn_id': 't', 'current_user': 'A',
                    'long_term_memory': [{'user_id': 'A', 'content': str(i), 'time': []} for i in range(7)],
                    'short_term_memory': [{'role': 'user', 'speaker_id': 'A', 'content': '?', 'time': 'now'}]}
        r = convert_payload(request={'turn_id': 't', 'user_id': 'A', 'query': '?'}, response=response,
                            agent_id='bot', stream_id='capture', system='system', split='profile', origin='synthetic')
        validate_trace(r)
        self.assertEqual(len(r['l3']), 7)
        self.assertFalse(r['mem0_top5_observed_separately'])
        self.assertIn('source-version-unavailable', r['l3'][0]['version'])

    def test_sqlite_export_does_not_initialize_or_migrate_source(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'memory.sqlite'
            conn = sqlite3.connect(path)
            conn.execute('CREATE TABLE context_requests(request_id,agent_id,user_id,query,response_json,created_at)')
            payload = {'turn_id': 't', 'current_user': 'A', 'long_term_memory': [], 'short_term_memory': []}
            conn.execute('INSERT INTO context_requests VALUES(?,?,?,?,?,?)', ('r', 'bot', 'A', '?', json.dumps(payload), 1))
            conn.commit();conn.close()
            before = path.read_bytes()
            self.assertEqual(len(list(cached_contexts_readonly(path))), 1)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':unittest.main()
