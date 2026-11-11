"""Rerun the Phase 25 nsmallest panel and measure fixed-prompt seed sensitivity."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import struct
from time import perf_counter

from mas_panel.hf import HFBackend, TEMPERATURE
from mas_panel.protocol import decision, digest, make_panels
from mas_panel.runner import STAGES, inspect_request, load_dataset, read_jsonl, run_experiment, run_panels, write_json


MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"
ITEM = "real-code-cpython-nsmallest-2"
PANEL_SEED = 29
INITIALIZATION = "complementary"
LAYER = 17
REVIEWER = 3
SWEEP_SEEDS = (11, 21, 31, 41, 51, 61, 71)
LABELS = ("Yes", "No", "Unknown", " Yes", " No", " Unknown")


def file_sha256(path):
    return sha256(path.read_bytes()).hexdigest()


def values_sha256(values):
    """Hash the captured values as little-endian IEEE float32."""
    return sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


def historical_panel(dataset, history):
    items = load_dataset(dataset)
    if set(items) != {ITEM}:
        raise ValueError("Expected the single archived nsmallest item")
    panel, = (p for p in make_panels(items.values(), "main")
              if p["panel_seed"] == PANEL_SEED and p["initialization"] == INITIALIZATION)
    rows = list(read_jsonl(history))
    index = {r["call_id"]: r for r in rows}
    if len(rows) != 16 or len(index) != 16:
        raise ValueError("Expected 16 distinct historical calls")

    def lookup(request):
        row = index.pop(request["call_id"])
        if row["request"] != request:
            raise ValueError(f"Archived request differs: {request['call_id']}")
        return row

    run_panels(items, [panel], STAGES, lookup)
    if index:
        raise ValueError("Unused historical calls")
    return panel, rows


def observed_run(path, expected_calls, expected_layers):
    rows = list(read_jsonl(path / "trace.jsonl"))
    captures = list(read_jsonl(path / "activations.jsonl"))
    if len(rows) != expected_calls or len(captures) != expected_calls:
        raise ValueError(f"Incomplete run: {path.name}")
    for row, capture in zip(rows, captures, strict=True):
        key = {k: capture[k] for k in ("run_id", "call_id", "attempt")}
        if key != row["capture_key"] or row["status"] != "ok":
            raise ValueError("Capture does not join to its completed call")
        tokens = row["response"]["prompt_token_ids"]
        if capture["prompt_tokens_sha256"] != digest(tokens):
            raise ValueError("Capture prompt hash differs")
        if (capture["anchor_index"], capture["anchor_token_id"]) != (len(tokens) - 1, tokens[-1]):
            raise ValueError("Capture anchor differs")
        if [layer["index"] for layer in capture["layers"]] != expected_layers:
            raise ValueError("Capture layer selection differs")
    return rows, captures


def max_difference(left, right):
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def parity(off, on):
    off_rows, off_captures = off
    on_rows, on_captures = on
    checks = []
    for a, b, x, y in zip(off_rows, on_rows, off_captures, on_captures, strict=True):
        same_request = a["request"] == b["request"]
        checks.append({
            "stage": a["request"]["stage"],
            "agent_id": a["request"]["agent_id"],
            "call_id": a["call_id"],
            "request_equal": same_request,
            "prompt_ids_equal": a["response"]["prompt_token_ids"] == b["response"]["prompt_token_ids"],
            "output_ids_equal": a["response"]["token_ids"] == b["response"]["token_ids"],
            "text_equal": a["response"]["text"] == b["response"]["text"],
            "response_equal": a["response"] == b["response"],
            "parsed_equal": a["parsed"] == b["parsed"],
            "logits_equal": x["logits"] == y["logits"],
            "logits_max_abs_delta": max_difference(x["logits"], y["logits"]),
        })
    counts = {field: sum(row[field] for row in checks) for field in checks[0] if field.endswith("_equal")}
    return {"calls": len(checks), "equal_counts": counts, "by_call": checks}


def score_tokens(backend, row, capture):
    tokenizer = backend.tokenizer
    response = row["response"]
    prompt = tokenizer.apply_chat_template(row["request"]["messages"], tokenize=False, add_generation_prompt=True)
    ids = response["prompt_token_ids"]
    logits = backend.torch.tensor(capture["logits"], dtype=backend.torch.float32)
    probabilities = (logits / TEMPERATURE).softmax(dim=-1)

    def token_score(token):
        return {
            "token_id": token,
            "logit": logits[token].item(),
            "probability": probabilities[token].item(),
            "rank": (logits > logits[token]).sum().item() + 1,
        }

    labels = {}
    for label in LABELS:
        extended = tokenizer.encode(prompt + label, add_special_tokens=False)
        if extended[:-1] != ids or len(extended) != len(ids) + 1:
            raise ValueError(f"Label is not a single appended token: {label!r}")
        labels[label] = token_score(extended[-1])
    top = [
        {**token_score(token), "text": tokenizer.decode([token], skip_special_tokens=False)}
        for token in logits.topk(3).indices.tolist()
    ]
    return {
        "labels": labels,
        "top3": top,
        "yes_minus_no_logit": labels["Yes"]["logit"] - labels["No"]["logit"],
        "first_generated_token": token_score(response["token_ids"][0]),
    }


def compact_calls(backend, run):
    rows, captures = run
    return [
        {
            "call_id": row["call_id"],
            "stage": row["request"]["stage"],
            "agent_id": row["request"]["agent_id"],
            "role": row["request"]["role"],
            "seed": row["request"]["seed"],
            "request_sha256": row["request_sha256"],
            "prompt_tokens": len(row["response"]["prompt_token_ids"]),
            "prompt_tokens_sha256": cap["prompt_tokens_sha256"],
            "anchor_token_id": cap["anchor_token_id"],
            "text": row["response"]["text"],
            "parsed": row["parsed"],
            "generated_token_ids": row["response"]["token_ids"],
            "stop_reason": row["response"]["stop_reason"],
            "sampled_token_logprobs": row["response"]["sampled_token_logprobs"],
            "logits_sha256": values_sha256(cap["logits"]),
            "layer_vector_sha256": values_sha256(cap["layers"][0]["values"]) if cap["layers"] else None,
            "token_scores": score_tokens(backend, row, cap) if row["request"]["stage"] != "step" else None,
        }
        for row, cap in zip(rows, captures, strict=True)
    ]


def seed_sweep(backend, history, row, output, timings):
    request = row["request"]
    seeds = [request["seed"], *SWEEP_SEEDS]
    start = perf_counter()
    inspect_request(history, row["call_id"], output, backend, seeds)
    timings[output.name] = perf_counter() - start
    run = observed_run(output, len(seeds), [LAYER])
    new_rows, captures = run
    if [r["request"]["seed"] for r in new_rows] != seeds:
        raise ValueError("Unexpected seeds in replay")
    for record in new_rows:
        other = record["request"]
        if {k: v for k, v in other.items() if k not in {"seed", "call_id", "source_call_id"}} != {
            k: v for k, v in request.items() if k not in {"seed", "call_id"}
        }:
            raise ValueError("Fixed-prompt request changed")
    compact = compact_calls(backend, run)
    logits = captures[0]["logits"]
    vector = captures[0]["layers"][0]["values"]
    return {
        "historical_call_id": row["call_id"],
        "historical_text": row["text"],
        "historical_request_sha256": digest(request),
        "seeds": seeds,
        "prompt_ids_identical": all(
            r["response"]["prompt_token_ids"] == new_rows[0]["response"]["prompt_token_ids"] for r in new_rows
        ),
        "answer_counts": dict(Counter(call["parsed"] or "invalid" for call in compact)),
        "logits_max_abs_delta": max(max_difference(logits, c["logits"]) for c in captures),
        "layer_vector_max_abs_delta": max(max_difference(vector, c["layers"][0]["values"]) for c in captures),
        "calls": compact,
    }


def panel_summary(panel):
    gold = panel["gold"]
    branches = {}
    for stage in STAGES[1:]:
        labels = [decision(a[stage]["text"]) for a in panel["agents"]]
        counts = Counter(labels)
        winner = next((label for label in ("yes", "no") if counts[label] >= 3), None)
        branches[stage] = {
            "labels": labels,
            "correct_agent_ids": [i for i, label in enumerate(labels) if label == gold],
            "winner": winner,
            "vote_correct": winner == gold if winner is not None else None,
        }
    return {"gold": gold, "agents": panel["agents"], "branches": branches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "examples" / "nsmallest_phase25")
    parser.add_argument("--output", type=Path, required=True, help="New directory for all run artifacts")
    parser.add_argument("--local-files-only", action="store_true", help="Use the pinned checkpoint from the local Hub cache")
    args = parser.parse_args()
    dataset = args.case_dir / "dataset.json"
    history = args.case_dir / "historical.jsonl"
    historical, historical_rows = historical_panel(dataset, history)
    args.output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    backend = HFBackend(MODEL, REVISION, device="cuda", dtype="float16", logits=True,
                        local_files_only=args.local_files_only)
    backend.torch.cuda.reset_peak_memory_stats(backend.device)
    timings = {"model_load": perf_counter() - start}
    panels = {}
    runs = {}
    for mode, layers in (("off", ()), ("on", (LAYER,))):
        backend.layers = layers
        backend.metadata["capture_layers"] = list(layers)
        run_path = args.output / mode
        started = perf_counter()
        run_experiment(dataset, run_path, backend, "main", ITEM, PANEL_SEED, INITIALIZATION)
        timings[mode] = perf_counter() - started
        panels[mode], = json.loads((run_path / "panels.json").read_text(encoding="utf-8"))
        runs[mode] = observed_run(run_path, 16, list(layers))
    checks = parity(runs["off"], runs["on"])
    checks["panels_equal"] = panels["off"] == panels["on"]
    checks["summaries_equal"] = (args.output / "off" / "summary.json").read_bytes() == (
        args.output / "on" / "summary.json").read_bytes()
    sweeps = {}
    for stage in ("private", "peer"):
        source, = (r for r in historical_rows if r["request"]["stage"] == stage
                   and r["request"]["agent_id"] == REVIEWER)
        sweeps[stage] = seed_sweep(backend, history, source, args.output / f"inspect_{stage}", timings)
    report = {
        "item_id": ITEM,
        "panel_seed": PANEL_SEED,
        "initialization": INITIALIZATION,
        "dataset_sha256": file_sha256(dataset),
        "historical_sha256": file_sha256(history),
        "runner_sha256": file_sha256(Path(__file__)),
        "backend": backend.metadata,
        "gpu_total_bytes": backend.torch.cuda.get_device_properties(backend.device).total_memory,
        "gpu_peak_allocated_bytes": backend.torch.cuda.max_memory_allocated(backend.device),
        "timing_seconds": {**timings, "total_including_load": perf_counter() - start},
        "model_calls": 32 + sum(len(sweep["seeds"]) for sweep in sweeps.values()),
        "capture_control": "First-token logits recorded in both modes. Block 17 hook enabled only in on mode.",
        "score_definition": "Full-vocabulary first-token softmax at decoding temperature. Labels are exact single appended tokens.",
        "historical_panel": panel_summary(historical),
        "fresh_panels": {mode: panel_summary(panel) for mode, panel in panels.items()},
        "parity": checks,
        "fresh_calls": {mode: compact_calls(backend, run) for mode, run in runs.items()},
        "historical_prompt_seed_sweeps": sweeps,
    }
    write_json(args.output / "report.json", report)
    print(json.dumps({
        "model_calls": report["model_calls"],
        "parity": checks["equal_counts"],
        "seed_sweeps": {stage: sweep["answer_counts"] for stage, sweep in sweeps.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
