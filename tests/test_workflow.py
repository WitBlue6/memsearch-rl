import importlib.util
import json
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('workflow', Path(__file__).resolve().parents[1]/'scripts/experiment.py')
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)

class WorkflowTests(unittest.TestCase):
    def test_generated_configs_escape_paths_and_preserve_roles(self):
        env = dict(EXPERIMENT_ID='test', READER_REPO='org/reader', READER_MODEL='/tmp/model "reader"', POLICY_REPO='org/policy', POLICY_MODEL='/tmp/policy', BUDGET_TOKENIZER='/tmp/tokenizer "fixed"', MAX_NEW='512', READER_PORT='8123', MEMORY_PORT='8124')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, env), patch('sys.argv', ['experiment.py', 'config']):
            old = os.getcwd()
            try:
                os.chdir(directory)
                workflow.main()
                root = Path('configs/generated/test')
                before = tomllib.loads((root/'before.toml').read_text())
                after = tomllib.loads((root/'after.toml').read_text())
                summary = tomllib.loads((root/'summary.toml').read_text())
                self.assertEqual(before['budget']['tokenizer'], env['BUDGET_TOKENIZER'])
                self.assertEqual(before['backend'], after['backend'])
                self.assertEqual(before['agent']['policy'], 'fixed')
                self.assertEqual(summary['agent']['memory'], 'summary')
                self.assertEqual(after['memory_backend']['base_url'], 'http://127.0.0.1:8124/v1')
                with self.assertRaises(FileExistsError):
                    workflow.main()
            finally:
                os.chdir(old)

    def test_model_directory_is_passed_as_one_argument(self):
        env = dict(EXPERIMENT_ID='test', READER_MODEL='/tmp/model with spaces', READER_REPO='org/custom')
        with patch.dict(os.environ, env), patch('sys.argv', ['experiment.py','download']), patch.object(workflow, 'run') as run:
            workflow.main()
            self.assertEqual(run.call_args.args[-3:], ('org/custom', '--local-dir', '/tmp/model with spaces'))
