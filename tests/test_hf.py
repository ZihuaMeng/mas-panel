"""Offline parity check for the decoder and its optional instrumentation."""

import math
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM
except ModuleNotFoundError:
    torch = None

from mas_panel.hf import HFBackend, TEMPERATURE


@unittest.skipIf(torch is None, "Requires the optional HF dependencies")
class CaptureTest(unittest.TestCase):
    def test_capture_matches_forward_pass_and_leaves_sampling_unchanged(self):
        for config_class, model_class in ((Qwen2Config, Qwen2ForCausalLM), (Qwen3Config, Qwen3ForCausalLM)):
            with self.subTest(model=model_class.__name__):
                self.check_capture(config_class, model_class)

    def check_capture(self, config_class, model_class):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.make_checkpoint(folder, config_class, model_class)
            baseline = HFBackend(folder, None, local_files_only=True)
            backend = HFBackend(folder, None, layers=(0,), logits=True, local_files_only=True)
            hashes = backend.metadata["local_files_sha256"]
            self.assertIn("model.safetensors", hashes)
            self.assertIn("tokenizer.json", hashes)
            self.assertEqual(hashes, baseline.metadata["local_files_sha256"])
            request = {
                "messages": [{"role": "user", "content": "Review"}],
                "seed": 29,
                "max_tokens": 4,
            }
            expected = baseline.generate(request)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(71)
                actual = backend.generate(request)
            capture = actual.pop("capture")
            self.assertEqual(actual, expected)
            self.assertEqual(actual["prompt_token_ids"], [3, 4, 1, 5])
            self.assertEqual(capture["anchor_index"], 3)
            self.assertEqual(capture["anchor_token_id"], 5)
            self.assertEqual(capture["layers"][0]["module"], "model.layers.0")
            self.assertFalse(backend.model.model.layers[0]._forward_hooks)

            prompt = torch.tensor([actual["prompt_token_ids"]])
            with torch.inference_mode():
                reference = backend.model(prompt, output_hidden_states=True, use_cache=False)
            reference_logits = reference.logits[0, -1]
            torch.testing.assert_close(torch.tensor(capture["logits"]), reference_logits)
            torch.testing.assert_close(
                torch.tensor(capture["layers"][0]["values"]), reference.hidden_states[1][0, -1],
            )
            for index, sampled in enumerate(actual["token_ids"]):
                full_prefix = torch.tensor([actual["prompt_token_ids"] + actual["token_ids"][:index]])
                with torch.inference_mode():
                    scores = backend.model(full_prefix, use_cache=False, logits_to_keep=1).logits[0, -1]
                probability = torch.softmax(scores / TEMPERATURE, dim=-1)[sampled].item()
                self.assertAlmostEqual(actual["sampled_token_logprobs"][index], math.log(probability), places=5)

            with self.assertRaisesRegex(ValueError, "zero-based"):
                HFBackend(folder, None, layers=(2,), local_files_only=True)

    @staticmethod
    def make_checkpoint(folder, config_class, model_class):
        vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "u": 3, "R": 4, "a": 5}
        tokenizer = Qwen2Tokenizer(
            vocab=vocab, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message.role[0] }}{{ message.content[0] }}[EOS]"
            "{% endfor %}{% if add_generation_prompt %}a{% endif %}"
        )
        tokenizer.save_pretrained(folder)
        config = config_class(
            vocab_size=len(vocab), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
            pad_token_id=0, eos_token_id=1, bos_token_id=None,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model_class(config).save_pretrained(folder)
