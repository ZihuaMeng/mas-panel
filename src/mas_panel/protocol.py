"""The fixed-code protocol used by the Phase 25 pilot."""

from collections import Counter
import hashlib
import json
import re


ROLES = ("trace", "boundary", "invariant", "simulation")
SEEDS = (17, 29)
CAPS = {"step": 48, "initial": 8, "private": 8, "peer": 8}
TEMPERATURE = 0.3
SYSTEM = "You are an independent Python code reviewer. Follow the requested output format."
REVIEW = "Reconsider your initial answer using the code and the exact input. Return your own final verdict."


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def parse_array(text):
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\[.*\])\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    try:
        value = json.loads(fenced[1] if fenced else stripped)
    except ValueError:
        return None
    return value if isinstance(value, list) else None


def decision(text):
    value = text.strip().casefold()
    return value if value in {"yes", "no", "unknown"} else None


def output_format(order):
    return f"Reply with exactly one word: {' or '.join(order)} or Unknown."


def message(role, content):
    return {"role": role, "content": content}


def make_panels(items, split):
    panels = []
    for item in items:
        if item["split"] != split:
            continue
        for seed_index, seed in enumerate(SEEDS):
            order = ("Yes", "No") if seed_index == 0 else ("No", "Yes")
            initializations = ("complementary",) if split == "dev" else ("neutral", "complementary")
            for initialization in initializations:
                agents = []
                for slot in range(4):
                    check = ROLES[(slot + seed_index) % 4] if initialization == "complementary" else "simulation"
                    agents.append({"agent_id": slot, "role": check if initialization == "complementary" else "neutral",
                                   "check_role": check})
                panels.append({"item_id": item["item_id"], "source_family": item["source_family"], "gold": item["gold"],
                               "panel_seed": seed, "initialization": initialization, "answer_order": list(order),
                               "agents": agents})
    return panels


def stage_messages(item, panel, agent, stage):
    call_input = canonical(item["diagnostic_trace"]["input"])
    check = item["role_checks"][agent["check_role"]]["instruction"]
    entry = item["source"].get("entrypoint") or item["source"]["function_name"]
    context = [
        message("system", SYSTEM),
        message("user", f"Evaluate the displayed Python function `{entry}` on the exact input below. "
                f"String values are JSON-escaped.\n\n```python\n{item['artifact']}\n```\n"
                f"\nInput: {call_input}\n\n{check}\nReply with only a JSON array."),
    ]
    if stage == "step":
        return context
    fmt = output_format(panel["answer_order"])
    context += [message("assistant", agent["step"]["text"]),
                message("user", f"Claim: {item['claim']}\n{fmt}")]
    if stage == "initial":
        return context
    review = REVIEW
    if stage == "peer":
        other = [peer for peer in panel["agents"] if peer["agent_id"] != agent["agent_id"]]
        if panel["panel_seed"] == SEEDS[1]:
            other.reverse()
        labels = [decision(peer["initial"]["text"]) for peer in other]
        peer_text = "\n".join(f"Peer {i}: {label.title() if label else 'Unparseable'}"
                              for i, label in enumerate(labels, 1))
        review = f"The other three reviewers gave these initial answers:\n{peer_text}\n\n{review}"
    return context + [message("assistant", agent["initial"]["text"]), message("user", f"{review}\n{fmt}")]


def make_request(item, panel, agent, stage):
    revision_stage = "review" if stage in {"private", "peer"} else stage
    state = [item["item_id"], panel["panel_seed"], agent["agent_id"]]
    seed = int(digest(state + [revision_stage])[:15], 16)
    return {
        "call_id": digest(state + [panel["initialization"], stage]),
        "item_id": item["item_id"],
        "panel_seed": panel["panel_seed"],
        "agent_id": agent["agent_id"],
        "initialization": panel["initialization"],
        "role": agent["role"],
        "check_role": agent["check_role"],
        "stage": stage,
        "seed": seed,
        "max_tokens": CAPS[stage],
        "messages": stage_messages(item, panel, agent, stage),
    }


def summarize_gate(panels, items):
    correct = Counter()
    verified_panels = 0
    parsed = 0
    for panel in panels:
        roles = set()
        for agent in panel["agents"]:
            role = agent["check_role"]
            expected = items[panel["item_id"]]["role_checks"][role]["expected"]
            if canonical(parse_array(agent["step"]["text"])) == canonical(expected):
                correct[role] += 1
                roles.add(role)
            parsed += decision(agent["initial"]["text"]) is not None
        verified_panels += len(roles) >= 3
    counts = {role: correct[role] for role in ROLES}
    return {"exact_by_role": counts, "panels_at_least_three": verified_panels, "parseable_initial": parsed,
            "pass": min(counts.values()) >= 8 and verified_panels >= 8 and parsed >= 58}
