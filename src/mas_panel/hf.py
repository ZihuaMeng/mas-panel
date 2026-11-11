"""Single-device Hugging Face inference with optional prefill capture."""

from hashlib import file_digest
import math
from pathlib import Path
import platform
import re


TEMPERATURE = 0.3
MODEL_TYPES = {"qwen2", "qwen3"}
DTYPES = {"float32", "float16", "bfloat16"}


def local_file_hashes(directory):
    metadata_names = {
        "config.json", "generation_config.json", "model.safetensors.index.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "vocab.json", "merges.txt", "chat_template.jinja",
    }
    files = sorted({
        *directory.glob("*.safetensors"),
        *directory.glob("chat_templates/*.jinja"),
        *(directory / name for name in metadata_names if (directory / name).is_file()),
    })
    hashes = {}
    for file in files:
        with file.open("rb") as stream:
            hashes[file.relative_to(directory).as_posix()] = file_digest(stream, "sha256").hexdigest()
    return hashes


class HFBackend:
    """Sample from Qwen2 or Qwen3 with an independent generator per request."""

    def __init__(
        self,
        model_id,
        revision,
        device="cpu",
        dtype="float32",
        layers=(),
        logits=False,
        local_files_only=False,
    ):
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_id = str(model_id)
        directory = Path(model_id)
        is_local = directory.is_dir()
        if not is_local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("Use a full Hub commit SHA or a local model directory")
        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("This backend supports CPU and CUDA devices")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())

        options = {
            "revision": revision,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
            "token": False,
        }
        config = AutoConfig.from_pretrained(model_id, **options)
        if config.model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        self.layers = tuple(layers)
        if any(type(i) is not int or not 0 <= i < config.num_hidden_layers for i in self.layers):
            raise ValueError("Layer indices must be zero-based decoder block indices")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("Layer indices must be unique")
        self.capture_logits = logits
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, dtype=getattr(torch, dtype), use_safetensors=True, **options
        ).to(self.device).eval()

        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos]) - {None}
        self.metadata = {
            "backend": "hf",
            "model_id": model_id,
            "revision": revision,
            "model_type": config.model_type,
            "dtype": str(self.model.dtype).removeprefix("torch."),
            "device": str(self.device),
            "device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda" else platform.machine()
            ),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "tokenizer_class": type(self.tokenizer).__name__,
            "temperature": TEMPERATURE,
            "sampling": "full_vocabulary_multinomial",
            "use_cache": True,
            "eos_token_ids": sorted(self.eos),
            "capture_layers": list(self.layers),
            "capture_logits": self.capture_logits,
        }
        if is_local:
            self.metadata["local_files_sha256"] = local_file_hashes(directory)

    def _record_values(self, tensor):
        values = tensor.detach().cpu()
        if not self.torch.isfinite(values).all().item():
            raise FloatingPointError("Nonfinite prefill capture")
        return values.tolist()

    def _prefill(self, prompt, capture):
        hooks = []

        def record_layer(index):
            def record(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.ndim != 3 or hidden.shape[:2] != prompt.shape:
                    raise RuntimeError(f"Unexpected prefill shape in block {index}")
                vector = hidden[0, -1].detach()
                capture["layers"].append({
                    "index": index,
                    "module": f"model.layers.{index}",
                    "dtype": str(vector.dtype).removeprefix("torch."),
                    "values": self._record_values(vector),
                })
            return record

        try:
            for index in self.layers:
                layer = self.model.model.layers[index]
                hooks.append(layer.register_forward_hook(record_layer(index)))
            result = self.model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        finally:
            for hook in hooks:
                hook.remove()

        if capture is not None:
            if len(capture["layers"]) != len(self.layers):
                raise RuntimeError("Missing prefill activation")
            if self.capture_logits:
                logits = result.logits[0, -1].detach()
                capture["logits"] = self._record_values(logits)
                capture["logits_dtype"] = str(logits.dtype).removeprefix("torch.")
        return result

    def generate(self, request):
        torch = self.torch
        cap = request["max_tokens"]
        seed = request["seed"]
        if type(cap) is not int or cap < 1:
            raise ValueError("max_tokens must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")

        encoded = self.tokenizer.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        prompt = encoded["input_ids"].to(self.device)
        if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] < 1:
            raise ValueError("Expected one nonempty, unpadded token sequence")
        if "attention_mask" in encoded and not encoded["attention_mask"].all().item():
            raise ValueError("Padded prompts are unsupported")
        prompt_ids = prompt[0].tolist()
        if len(prompt_ids) + cap > self.model.config.max_position_embeddings:
            raise ValueError("Prompt plus generation cap exceeds the model context")
        capture = None
        if self.layers or self.capture_logits:
            capture = {
                "anchor_index": len(prompt_ids) - 1,
                "anchor_token_id": prompt_ids[-1],
                "layers": [],
            }

        rng = torch.Generator(device=self.device).manual_seed(seed)
        tokens, logprobs = [], []
        stop = "length"
        with torch.inference_mode():
            result = self._prefill(prompt, capture)
            for step in range(cap):
                raw_logits = result.logits[0, -1]
                probabilities = torch.softmax(raw_logits.float() / TEMPERATURE, dim=-1)
                sampled = torch.multinomial(probabilities, 1, generator=rng)
                token = sampled.item()
                probability = probabilities[token].item()
                if not 0.0 < probability <= 1.0:
                    raise FloatingPointError("Invalid sampled token probability")
                tokens.append(token)
                logprobs.append(math.log(probability))
                if token in self.eos:
                    stop = "eos"
                    break
                if step + 1 < cap:
                    result = self.model(
                        input_ids=sampled.view(1, 1), past_key_values=result.past_key_values,
                        use_cache=True, logits_to_keep=1,
                    )

        response = {
            "text": self.tokenizer.decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "token_ids": tokens,
            "prompt_token_ids": prompt_ids,
            "stop_reason": stop,
            "sampled_token_logprobs": logprobs,
        }
        if capture is not None:
            response["capture"] = capture
        return response
