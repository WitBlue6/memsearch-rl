import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_TRAIN = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "accelerate"))


@unittest.skipUnless(HAS_TRAIN, "Install .[train] for numerical training checks")
class TrainingMathTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.torch = torch
        self.model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=64, n_embd=16,
                                               n_layer=1, n_head=2, resid_pdrop=0,
                                               embd_pdrop=0, attn_pdrop=0))
        self.model.eval()

    def test_generated_token_loss_matches_masked_causal_loss(self):
        from memsearch.training import token_logprobs
        torch = self.torch
        prompt, completion = [1, 2, 3], [4, 5, 6]
        calculated = -token_logprobs(self.model, prompt, completion, "cpu").mean()
        ids = torch.tensor([prompt + completion])
        labels = torch.tensor([[-100, -100, -100] + completion])
        expected = self.model(input_ids=ids, labels=labels).loss
        self.assertTrue(torch.allclose(calculated, expected, atol=1e-6))
        calculated.backward()
        self.assertGreater(self.model.transformer.wte.weight.grad.abs().sum().item(), 0)

    def test_clipping_both_advantage_signs(self):
        from memsearch.training import clipped_objective
        torch = self.torch
        old = torch.zeros(2)
        new = torch.log(torch.tensor([1.5, 0.5]))
        self.assertTrue(torch.allclose(clipped_objective(new, old, 1), torch.tensor([-1.2, -0.5])))
        self.assertTrue(torch.allclose(clipped_objective(new, old, -1), torch.tensor([1.5, 0.8])))

    def test_policy_update_increases_rewarded_completion_probability(self):
        from memsearch.training import clipped_objective, token_logprobs
        torch = self.torch
        old = token_logprobs(self.model, [1, 2], [3, 4], "cpu").detach()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        new = token_logprobs(self.model, [1, 2], [3, 4], "cpu")
        loss = clipped_objective(new, old, 1).mean()
        loss.backward()
        optimizer.step()
        after = token_logprobs(self.model, [1, 2], [3, 4], "cpu").detach()
        self.assertGreater(after.mean().item(), old.mean().item())

    def test_sampling_records_exact_ids_and_logprobs(self):
        from memsearch.training import TrainBackend, token_logprobs
        torch = self.torch

        class Tokenizer:
            eos_token_id, pad_token_id = 31, 0

            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3]

            def decode(self, ids, **kwargs):
                return "fixture response"

        backend = TrainBackend(self.model, Tokenizer(), "cpu", max_new=4)
        backend.complete([{"role": "user", "content": "not a label"}], "controller")
        sample = backend.samples[0]
        expected = token_logprobs(self.model, sample.prompt_ids, sample.completion_ids, "cpu").detach()
        self.assertTrue(torch.allclose(expected, torch.tensor(sample.old_logprobs), atol=1e-6))
        self.assertEqual(backend.calls[0].output_tokens, len(sample.completion_ids))

    def test_constrained_logprobs_match_masked_model_and_backpropagate(self):
        from memsearch.training import token_logprobs
        torch = self.torch
        class Grammar:
            def masks(self, schema, completion):
                return torch.tensor([[24]] * len(completion), dtype=torch.int32)
        prompt, completion = [1, 2], [3, 4]
        calculated = token_logprobs(self.model, prompt, completion, "cpu", Grammar(), {"test": True})
        logits = self.model(input_ids=torch.tensor([prompt + completion])).logits[0, 1:-1].float()
        expected = logits[:, [3, 4]].log_softmax(-1)[torch.arange(2), torch.tensor([0, 1])]
        self.assertTrue(torch.allclose(calculated, expected, atol=1e-6))
        (-calculated.mean()).backward()
        self.assertTrue(torch.isfinite(self.model.transformer.wte.weight.grad).all())

    def test_lora_saves_reloadable_adapter_after_real_update(self):
        from peft import LoraConfig, PeftModel, get_peft_model
        from memsearch.training import token_logprobs
        torch = self.torch
        policy = get_peft_model(self.model, LoraConfig(r=2, lora_alpha=4, target_modules="all-linear",
                                                      task_type="CAUSAL_LM"))
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.01)
        before = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}
        (-token_logprobs(policy, [1, 2], [3, 4], "cpu").mean()).backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[n], p) for n, p in policy.named_parameters() if p.requires_grad))
        with tempfile.TemporaryDirectory() as temp:
            policy.save_pretrained(temp)
            self.assertTrue((Path(temp) / "adapter_model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
