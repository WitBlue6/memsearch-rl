import importlib.util
import json
import unittest

from memsearch.structured import memory_schema

HAS_TORCH = importlib.util.find_spec('torch') is not None
HAS_XGRAMMAR = importlib.util.find_spec('xgrammar') is not None


class SchemaTests(unittest.TestCase):
    def test_schema_is_shared_without_gold_or_state_repair(self):
        messages = [{'content': json.dumps({'VALID_MEMORY_IDS': [], 'max_memory_ops': 16})}]
        schema = memory_schema(messages)
        self.assertEqual(schema['required'], ['operations'])
        self.assertEqual(schema['properties']['operations']['maxItems'], 16)
        variants = schema['properties']['operations']['items']['anyOf']
        self.assertEqual([v['properties']['op']['const'] for v in variants], ['ADD','UPDATE','DELETE','NOOP'])
        # Bad IDs remain an environment error, not silently replaced or remapped.
        self.assertNotIn('enum', variants[1]['properties']['target_id'])


@unittest.skipUnless(HAS_TORCH, 'torch required')
class MaskTests(unittest.TestCase):
    def test_mask_normalizes_and_blocks_invalid_gradients(self):
        import torch
        from memsearch.structured import mask_logits
        scores = torch.tensor([[1.,2.,3.,4.]], requires_grad=True)
        masked = mask_logits(scores, torch.tensor([[5]], dtype=torch.int32))
        loss = -masked.log_softmax(-1)[0,0]
        self.assertTrue(torch.allclose(loss, torch.logsumexp(scores[0,[0,2]],0)-scores[0,0]))
        loss.backward()
        self.assertEqual(scores.grad[0,1].item(),0)
        self.assertEqual(scores.grad[0,3].item(),0)
        self.assertTrue(torch.isfinite(scores.grad).all())


@unittest.skipUnless(HAS_TORCH and HAS_XGRAMMAR, 'Linux structured extra required')
class GrammarReplayTests(unittest.TestCase):
    def test_online_masks_match_replay_including_eos(self):
        import torch
        import xgrammar as xgr
        from memsearch.structured import Grammar, mask_logits
        grammar = Grammar.__new__(Grammar)
        grammar.xgr, grammar.vocab_size, grammar.cache = xgr, 4, {}
        grammar.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo([b'"', b'a', b'b', b'<eos>'], stop_token_ids=[3]))
        schema = {'type':'string','enum':['a','b']}
        completion = [0,1,0,3]
        processor = grammar.processor(schema)
        scores = torch.tensor([[0.,1.,2.,3.]])
        online = []
        for i, token in enumerate(completion):
            logits = processor(torch.tensor([[0]+completion[:i]]), scores.clone())
            online.append(logits.log_softmax(-1)[0,token])
        replay = mask_logits(scores.repeat(4,1), grammar.masks(schema,completion))
        expected = replay.log_softmax(-1)[torch.arange(4),torch.tensor(completion)]
        self.assertTrue(torch.allclose(torch.stack(online),expected))
        self.assertEqual(expected[-1].item(),0.)
        with self.assertRaises(ValueError):
            grammar.masks(schema,[0,3])
