"""Command line entry points for the panel protocol."""

import argparse
import json
from pathlib import Path

from .protocol import SEEDS
from .runner import inspect_request, replay_phase25, run_experiment


class DemoBackend:
    metadata = {"backend": "demo", "source": "scripted outputs; no model or token IDs"}

    def generate(self, request):
        stage, agent = request["stage"], request["agent_id"]
        if stage == "step":
            text = "[]"
        elif stage == "private":
            text = "Yes"
        else:
            text = "No" if agent == 3 else "Yes"
        return {"text": text, "token_ids": None, "prompt_token_ids": None, "stop_reason": "scripted"}


def output_arg(parser):
    parser.add_argument("--output", type=Path, required=True, help="New directory for the run")


def panel_args(parser):
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("main", "dev"), default="main")
    parser.add_argument("--item")
    parser.add_argument("--panel-seed", type=int, choices=SEEDS)
    parser.add_argument("--initialization", choices=("neutral", "complementary"))
    output_arg(parser)


def model_args(parser):
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local model directory")
    parser.add_argument("--revision", help="Full commit SHA for a Hub model")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--layers", type=int, nargs="+", default=(), help="Zero-based block indices")
    parser.add_argument("--logits", action="store_true", help="Save full raw logits for the first generated token")
    parser.add_argument("--local-files-only", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired multi-agent code review")
    commands = parser.add_subparsers(dest="command", required=True)
    panel_args(commands.add_parser("demo", help="Run the protocol using scripted answers"))
    live = commands.add_parser("run", help="Generate all stages from a fixed dataset")
    panel_args(live)
    model_args(live)
    replay = commands.add_parser("replay-phase25", help="Audit the original Phase 25 panel reconstruction offline")
    replay.add_argument("archive", type=Path, help="Evidence directory from the team snapshot")
    inspect = commands.add_parser("inspect", help="Regenerate one saved request with fixed history")
    inspect.add_argument("--source", required=True, type=Path, help="A model's raw.jsonl or trace.jsonl")
    inspect.add_argument("--call-id", required=True)
    inspect.add_argument("--seeds", type=int, nargs="+", help="Override only the per-call sampling seed")
    output_arg(inspect)
    model_args(inspect)
    args = parser.parse_args(argv)

    if args.command == "replay-phase25":
        result = replay_phase25(args.archive)
    else:
        if args.output.exists():
            parser.error("--output must not exist")
        if args.command == "demo":
            backend = DemoBackend()
        else:
            from .hf import HFBackend

            backend = HFBackend(args.model, args.revision, args.device, args.dtype, args.layers, args.logits,
                                args.local_files_only)
        if args.command == "inspect":
            result = {"replayed_requests": inspect_request(args.source, args.call_id, args.output, backend, args.seeds)}
        else:
            result = run_experiment(args.dataset, args.output, backend, args.split, args.item, args.panel_seed,
                                    args.initialization)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
