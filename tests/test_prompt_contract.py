import copy
import json
import unittest
from dataclasses import asdict

from memsearch.budget import Counter
from memsearch.memory_actions import apply_operations
from memsearch.prompts import messages
from memsearch.types import Evidence, MemoryItem, MemoryState


class PromptContractTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [Evidence('actual-hash:4', 'City', 4, 'Ada was born in Luma.')]

    def observation(self, state):
        return {'QUESTION': 'Where was Ada born?', 'OLD_MEMORY': state.payload(),
                'NEW_EVIDENCE': [asdict(e) for e in self.evidence],
                'memory_budget': 500, 'max_memory_ops': 16}

    def test_empty_memory_examples_are_executable_and_do_not_invent_targets(self):
        old = MemoryState()
        obs = self.observation(old)
        before = copy.deepcopy(obs)
        prompt = messages('memory_decision', obs)
        visible = json.loads(prompt[1]['content'])
        self.assertEqual(visible['VALID_MEMORY_IDS'], [])
        self.assertEqual(visible['ALLOWED_OPERATIONS'], ['ADD', 'NOOP'])
        self.assertNotIn('m0', json.dumps(prompt))
        for proposal in visible['FORMAT_EXAMPLES_NOT_RECOMMENDATIONS']:
            apply_operations(old, self.evidence, proposal, 500, Counter())
        self.assertEqual(obs, before)

    def test_existing_memory_examples_use_only_live_ids_and_sources(self):
        old = MemoryState([MemoryItem('Luma is in Norvia.', ('kept:2',), 'm7')], [], 8)
        prompt = messages('memory_decision', self.observation(old))
        visible = json.loads(prompt[1]['content'])
        self.assertEqual(visible['VALID_MEMORY_IDS'], ['m7'])
        self.assertEqual(set(visible['VALID_SOURCE_IDS']), {'kept:2', 'actual-hash:4'})
        for proposal in visible['FORMAT_EXAMPLES_NOT_RECOMMENDATIONS']:
            apply_operations(old, self.evidence, proposal, 500, Counter())
        self.assertNotIn('m0', json.dumps(prompt))

    def test_summary_examples_require_real_sources_and_no_placeholder(self):
        for role in ('summary', 'memory'):
            prompt = messages(role, self.observation(MemoryState()))
            visible = json.loads(prompt[1]['content'])
            self.assertNotIn('document:sentence', json.dumps(prompt))
            for example in visible['FORMAT_EXAMPLES_NOT_RECOMMENDATIONS']:
                for item in example['items']:
                    self.assertEqual(item['source_ids'], ['actual-hash:4'])
                    self.assertEqual(item['text'], self.evidence[0].text)

    def test_reader_cannot_cite_forgotten_sources(self):
        obs = {'QUESTION': 'q', 'memory': MemoryState().payload()}
        for role in ('reader', 'controller'):
            visible = json.loads(messages(role, obs)[1]['content'])
            self.assertEqual(visible['VALID_CITATION_IDS'], [])


if __name__ == '__main__':
    unittest.main()
