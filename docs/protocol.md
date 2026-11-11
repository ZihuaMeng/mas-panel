# Fixed-code protocol

## Panel

A panel is one item, checkpoint, panel seed, and prompt setting. Each of four agents receives the same code and input. In the neutral setting all four receive the simulation question; in the complementary setting each gets one of trace, boundary, invariant, or simulation. The check response enters that agent's history before the initial verdict. The two review branches start from that exact history and initial response. The peer branch adds the three other parsed initial labels; it does not add their reasoning, check results, or gold labels. All agents see the frozen initial panel, never another agent's revision.

`Yes`, `No`, and `Unknown` are valid verdicts. Other strings remain invalid. Three matching Yes or No verdicts are required to decide. Report correct→wrong, wrong→correct, Unknown and invalid transitions, alongside panel coverage, loss, retention by the same reviewer, turnover, and vote correctness. The private/peer contrast estimates the effect of adding these peer labels in this prompt. It also changes prompt length and framing. It is not a clean estimate of majority direction.

The Phase 25 layout uses panel seeds 17 and 29, with reversed display order and a role rotation. Keep this linkage for exact historical replay. For a seed-only diagnostic, replay one literal request and vary the per-call seed with `inspect`. Do not count seeds or alternate prompts as independent items. Freeze development and held-out sets by source family.

## Data and execution

A dataset is a JSON list following [the example](../examples/prefix_sum.json). Every item has a unique ID, source family, split, displayed function, exact input, observed return, claim, gold verdict, and four check instructions with typed reference arrays. Establish reference values outside this runner. The runner reads the code as prompt text and never executes it. Keep gold labels and expected checks out of prompt messages. Use `--split dev` to report check accuracy on new development data. It only runs the check and initial verdict stages. The historical threshold is verified by `replay-phase25`; freeze a separate criterion for new data.

The original development gates were not met (3B: 7/64 exact checks; 4B: 2/64). The saved held-out run was a predeclared descriptive continuation. Treat prompt names as assignments, not evidence that an agent carried out the check.

## Trace contract

| File | Contents |
| --- | --- |
| `manifest.json` | Run ID, source hashes, backend and checkpoint metadata, inputs and settings. |
| `trace.jsonl` | One attempt per call. The request includes panel identity, stage, literal chat messages, cap and seed. The response includes actual prompt and output token IDs, unparsed decoded text (special tokens removed) and stop reason. Parse status is separate. Execution errors are logged and terminate the run. |
| `activations.jsonl` | Optional read-only observations indexed by run, call and attempt, with prompt token hash and anchor. |
| `panels.json`, `summary.json` | Completed panel views and counts. Review metrics use the original private/peer pairing. |

The historical snapshot kept requests and output token IDs but no rendered input token IDs or hidden states. New runs record them directly. Models run separately; do not join calls from different checkpoints by call ID alone. A completed run has a summary; inspect partial records if generation aborts. The runner never retries with a new seed.

To add a backend, implement `generate(request)` and a serializable `metadata` mapping. The runner owns prompt construction, branch pairing and scoring. An adapter to Bohdan's vLLM engine would have to pass the explicit request seed and token cap through, keep all output token IDs, and preserve failure attempts. The default phase-derived seeds and parser retries in that runner are not equivalent to this protocol.

## Representations

Capture is implemented for the Hugging Face Qwen2 and Qwen3 decoder backends. It stores the output of a declared block at the last prompt token during prefill, before any answer token is generated. Raw next-token logits at that location are optional. Sampled-token log probabilities use the actual sampling distribution, including temperature; they are different from the raw logits. A label that tokenizes to more than one token requires a sequence likelihood if the question is label preference. Do not use raw cosine distances to compare two checkpoints' coordinate systems.

For a new experiment, first verify capture-off and capture-on token parity in the intended runtime. Compare the answer boundary and the same checkpoint within a paired panel. A difference in a vector or in logits can reflect the extra text in the peer prompt, so it is a diagnostic observation. An intervention needs a held-out behavioral test with paired sampling and a matched control.
