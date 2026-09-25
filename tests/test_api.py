import io
import json
import unittest
from unittest.mock import patch

from memsearch.backends import ChatBackend


class ApiTests(unittest.TestCase):
    def test_usage_accounting_and_environment_auth(self):
        result = {"choices": [{"message": {"content": '{"answer":"x"}'}}],
                  "usage": {"prompt_tokens": 123, "completion_tokens": 9}}
        backend = ChatBackend({"model": "test", "base_url": "http://localhost:8000/v1",
                               "api_key_env": "MEMSEARCH_TEST_SECRET"})
        with patch.dict("os.environ", {"MEMSEARCH_TEST_SECRET": "fixture-not-a-real-secret"}):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(result).encode())) as request:
                backend.complete([{"role": "user", "content": "q"}], "reader")
                sent = request.call_args.args[0]
                self.assertEqual(sent.get_header("Authorization"), "Bearer fixture-not-a-real-secret")
        self.assertEqual(backend.calls[0].input_tokens, 123)
        self.assertEqual(backend.calls[0].output_tokens, 9)
        self.assertNotIn("fixture-not-a-real-secret", str(backend.calls))

    def test_url_credentials_are_rejected(self):
        backend = ChatBackend({"model": "test", "base_url": "https://user:password@example.invalid/v1"})
        with self.assertRaises(ValueError):
            backend.complete([], "reader")

    def test_reader_schema_is_role_scoped_and_records_truncation(self):
        result = {"choices": [{"message": {"content": '{"answer":'}, "finish_reason": "length"}]}
        backend = ChatBackend({"model": "test", "base_url": "http://localhost:8000/v1", "reader_output": "json_schema"})
        for role in ('reader', 'memory', 'controller'):
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(result).encode())) as request:
                text = backend.complete([], role)
                body = json.loads(request.call_args.args[0].data)
                self.assertEqual('structured_outputs' in body, role == 'reader')
                if role == 'reader':
                    schema = body['structured_outputs']['json']
                    self.assertEqual(schema['required'], ['answer', 'evidence_ids'])
                    self.assertFalse(schema['additionalProperties'])
                    self.assertEqual(backend.calls[-1].decoding_mode, 'reader_json_schema_v1')
                else:
                    self.assertEqual(backend.calls[-1].decoding_mode, 'text')
                self.assertEqual(text, '{"answer":')
                self.assertEqual(backend.calls[-1].finish_reason, 'length')

    def test_schema_is_opt_in(self):
        result = {'choices': [{'message': {'content': 'answer: yes'}}]}
        backend = ChatBackend({'model': 'test', 'base_url': 'http://localhost:8000/v1'})
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(result).encode())) as request:
            backend.complete([], 'reader')
            self.assertNotIn('structured_outputs', json.loads(request.call_args.args[0].data))
            self.assertEqual(backend.calls[-1].response, 'answer: yes')

    def test_memory_api_uses_same_schema_as_local_sampling(self):
        from memsearch.structured import memory_schema
        messages = [{"role": "user", "content": json.dumps({"VALID_MEMORY_IDS": [], "max_memory_ops": 16})}]
        result = {"choices": [{"message": {"content": '{"operations":[{"op":"NOOP"}]}'}}]}
        backend = ChatBackend({"model":"test", "base_url":"http://localhost:8000/v1", "memory_output":"json_schema"})
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(result).encode())) as request:
            backend.complete(messages, 'memory')
            self.assertEqual(json.loads(request.call_args.args[0].data)['structured_outputs']['json'], memory_schema(messages))
            self.assertEqual(backend.calls[-1].decoding_mode, 'memory_json_schema_v1')
