# mas-panel

I use this setup to see what happens to a correct answer during multi-agent review. Four agents judge the same claim about a Python function on a fixed input. Each agent answers on its own first. I save those answers and start two reviews from the same history and decoding seed. In one review the agent reconsiders privately. In the other it sees the three peers’ initial answers.

I check who changed their answer, whether anyone still had the right answer, and what the group picked by a three-vote majority. In Bohdan’s coding experiment, reviewers rate deployability while a generator revises the program. Here I keep the program fixed and use its execution result to score the verdict. [Bohdan’s repo](https://github.com/BogdanTurbal/steering_agents/tree/79305914b5fc7dc7bbf0be2479829e8c6f69bd52) · [protocol and trace format](docs/protocol.md)

## quick start

Run this example from the repo root with Python 3.11 or later.

```sh
python -m pip install -e .
mas-panel demo --dataset examples/prefix_sum.json --panel-seed 29 \
  --initialization complementary --output runs/demo
```

The demo writes 16 scripted call records for one four-agent panel. The peer branch keeps one correct `No` and the group votes `Yes` three to one. The demo needs no GPU or model weights.

## run a model

Install PyTorch for your device and the Transformers extra. This example uses Qwen 3B on CUDA. Give each run a new output directory.

```sh
python -m pip install -e '.[hf]'
mas-panel run --dataset examples/prefix_sum.json --panel-seed 29 \
  --initialization complementary --output runs/qwen3b \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 \
  --device cuda --dtype float16 --layers 17 --logits
```

`trace.jsonl` has the messages, token IDs, decoded text, seed, stop reason, parse outcome, and sampled token log probabilities. `activations.jsonl` has the requested block outputs and first-token logits at the last prompt token. Match the files by `run_id`, `call_id`, and `attempt`. The manifest records the model revision. Omit `--logits` if you only need the vectors.

To see what sampling changes, rerun one saved prompt with `mas-panel inspect --source … --call-id … --seeds 11 21 31`, the same model flags, and a new output directory.

The Colab case includes the original prompts, all four reviewers’ answers, the capture-on/off check, and the results of changing only the decoding seed. [case and rerun commands](examples/nsmallest_phase25/RUN.md)

## replay the earlier run

If you have the lab’s Phase 25 snapshot, use it to rebuild and check all 768 saved replies without running the model.

```sh
mas-panel replay-phase25 /path/to/mas_setup_phase25/evidence
```

The earlier Colab run used batches and saved verdicts without activations. New capture runs generate one request at a time.

## tests

```sh
python -m unittest discover -s tests -v
```

The base tests require no model weights. With the HF extra installed, the capture test uses a locally initialized tiny Qwen model and checks output parity with capture on and off.
