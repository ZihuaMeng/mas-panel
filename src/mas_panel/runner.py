"""Run fixed panels and preserve the evidence needed to score them."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import uuid

from .analysis import summarize
from .protocol import CAPS, ROLES, SEEDS, decision, digest, make_panels, make_request, parse_array, summarize_gate

STAGES = ("step", "initial", "private", "peer")
SCHEMA = "mas-panel/1"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_dataset(path):
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Expected a nonempty list of items")
    ids, splits = set(), {}
    for item in items:
        item_id, family, split = (item[key] for key in ("item_id", "source_family", "split"))
        if not all(isinstance(key, str) and key for key in (item_id, family)):
            raise ValueError("Item IDs and source families must be nonempty strings")
        if item_id in ids or split not in ("dev", "main") or item["gold"] not in ("yes", "no"):
            raise ValueError(f"Invalid ID, split, or gold label: {item_id}")
        if splits.setdefault(family, split) != split:
            raise ValueError(f"Source family crosses the development and main splits: {family}")
        if not isinstance(item["artifact"], str) or not isinstance(item["claim"], str):
            raise ValueError(f"Expected code and claim strings: {item_id}")
        entrypoint = item["source"].get("entrypoint") or item["source"].get("function_name")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise ValueError(f"Missing source entrypoint: {item_id}")
        item["diagnostic_trace"]["input"]
        for role in ROLES:
            check = item["role_checks"][role]
            if not isinstance(check["instruction"], str) or not isinstance(check["expected"], list):
                raise ValueError(f"Invalid check for {item_id}/{role}")
        ids.add(item_id)
    return {item["item_id"]: item for item in items}


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from exc


def append_row(stream, row):
    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def run_panels(items, panels, stages, generate):
    for stage in stages:
        for panel in panels:
            for agent in panel["agents"]:
                request = make_request(items[panel["item_id"]], panel, agent, stage)
                result = generate(request)
                agent[stage] = {"text": result["text"], "call_id": request["call_id"]}
    return panels


def new_run(output, backend, kind, inputs):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA,
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "backend": backend.metadata,
        "inputs": inputs,
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
        "retry_policy": "none",
    }
    write_json(output / "manifest.json", manifest)
    return manifest["run_id"]


def record_request(run_id, request, backend, trace, captures):
    key = {"run_id": run_id, "call_id": request["call_id"], "attempt": 1}
    row = {**key, "request_sha256": digest(request), "request": request}
    try:
        response = dict(backend.generate(request))
        capture = response.pop("capture", None)
        text = response["text"]
        parsed = parse_array(text) if request["stage"] == "step" else decision(text)
        token_ids = response["prompt_token_ids"]
        if capture is not None:
            index = capture["anchor_index"]
            if not token_ids or index != len(token_ids) - 1 or token_ids[index] != capture["anchor_token_id"]:
                raise ValueError("Capture does not match the final prompt token")
            row["capture_key"] = key
            append_row(captures, {**capture, **key, "prompt_tokens_sha256": digest(token_ids)})
        row.update(status="ok", parse_status="valid" if parsed is not None else "invalid",
                   parsed=parsed, response=response)
    except Exception as exc:
        append_row(trace, {**row, "status": "error", "error": type(exc).__name__})
        raise
    append_row(trace, row)
    return response


def run_experiment(dataset, output, backend, split, item_id=None, seed=None, initialization=None):
    items = load_dataset(dataset)
    panels = [p for p in make_panels(items.values(), split)
              if (item_id is None or p["item_id"] == item_id)
              and (seed is None or p["panel_seed"] == seed)
              and (initialization is None or p["initialization"] == initialization)]
    if not panels:
        raise ValueError("No panels match the selection")
    run_id = new_run(output, backend, "panel", {
        "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "split": split, "item_id": item_id, "panel_seed": seed, "initialization": initialization,
        "protocol_panel_seeds": list(SEEDS), "token_caps": CAPS,
    })
    (output / "dataset.json").write_bytes(dataset.read_bytes())
    stages = STAGES if split == "main" else STAGES[:2]
    with (output / "trace.jsonl").open("w", encoding="utf-8") as trace, \
         (output / "activations.jsonl").open("w", encoding="utf-8") as captures:
        run_panels(items, panels, stages, lambda req: record_request(run_id, req, backend, trace, captures))
    write_json(output / "panels.json", panels)
    result = summarize(panels) if split == "main" else summarize_gate(panels, items)
    if split == "dev":
        result.pop("pass")
        result["panels"] = len(panels)
    write_json(output / "summary.json", result)
    return result


def inspect_request(source, call_id, output, backend, seeds):
    rows = [row for row in read_jsonl(source) if row["call_id"] == call_id]
    if len(rows) != 1:
        raise ValueError("Expected exactly one source row for the call ID")
    original = rows[0]["request"]
    if original["call_id"] != call_id:
        raise ValueError("Source row and request IDs differ")
    seeds = [original["seed"]] if seeds is None else seeds
    if len(set(seeds)) != len(seeds) or any(not 0 <= s < 2**63 for s in seeds):
        raise ValueError("Decoding seeds must be unique integers in [0, 2^63)")
    run_id = new_run(output, backend, "request_replay", {
        "source_trace_sha256": sha256(source.read_bytes()).hexdigest(),
        "source_call_id": call_id, "source_request_sha256": digest(original), "decoding_seeds": seeds,
    })
    with (output / "trace.jsonl").open("w", encoding="utf-8") as trace, \
         (output / "activations.jsonl").open("w", encoding="utf-8") as captures:
        for seed in seeds:
            request = {**original, "seed": seed, "source_call_id": call_id,
                       "call_id": digest([call_id, "request_replay", seed])}
            record_request(run_id, request, backend, trace, captures)
    write_json(output / "summary.json", {"replayed_requests": len(seeds)})
    return len(seeds)


def replay_phase25(archive):
    items = load_dataset(archive / "dataset.json")
    reports = {}
    for model in ("3b", "4b"):
        folder = archive / model
        rows = {}
        for row in read_jsonl(folder / "raw.jsonl"):
            if row["call_id"] in rows:
                raise ValueError(f"{model}: duplicate call ID")
            rows[row["call_id"]] = row
        count = len(rows)

        def replay(request):
            row = rows.pop(request["call_id"], None)
            if row is None or row["request"] != request:
                raise ValueError(f"{model}: replay request differs: {request['call_id']}")
            return row

        for split, stages in (("dev", STAGES[:2]), ("main", STAGES)):
            panels = run_panels(items, make_panels(items.values(), split), stages, replay)
            if panels != json.loads((folder / f"{split}_rows.json").read_text(encoding="utf-8")):
                raise ValueError(f"{model}: replayed {split} panels differ")
            if split == "main":
                reports[model] = {"replayed_calls": count, "main": summarize(panels)}
            elif summarize_gate(panels, items) != json.loads((folder / "gate.json").read_text(encoding="utf-8")):
                raise ValueError(f"{model}: development gate differs")
        if rows:
            raise ValueError(f"{model}: unconsumed trace entries")
    return reports
