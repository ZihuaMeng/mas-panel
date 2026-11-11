"""Summarize matched judgments and votes from completed fixed-code panels."""

from collections import Counter, defaultdict

from .protocol import decision


STAGES = ("initial", "private", "peer")
OUTCOMES = ("correct", "wrong", "unknown", "unparseable")
VOTES = ("correct", "wrong", "no_decision")
AVAILABILITY = ("available", "unavailable")


def _outcome(text, gold):
    label = decision(text) if isinstance(text, str) else None
    if label is None:
        return "unparseable"
    if label == "unknown":
        return "unknown"
    return "correct" if label == gold else "wrong"


def _vote(outcomes):
    for label in ("correct", "wrong"):
        if outcomes.count(label) >= 3:
            return label
    return "no_decision"


def _availability(outcomes):
    return "available" if "correct" in outcomes else "unavailable"


def _counts(values, labels):
    counts = Counter(values)
    return {label: counts[label] for label in labels}


def _cross_table(pairs, labels):
    counts = Counter(pairs)
    return {
        left: {right: counts[left, right] for right in labels}
        for left in labels
    }


def _review(panels, stage):
    transitions = Counter()
    changes = Counter()
    for panel in panels:
        initial, reviewed = panel["initial"], panel[stage]
        pairs = tuple(zip(initial, reviewed, strict=True))
        transitions.update(pairs)
        if "correct" in initial:
            if "correct" in reviewed:
                changes["retained_correct"] += 1
                if ("correct", "correct") in pairs:
                    changes["same_correct_agent_retained"] += 1
                else:
                    changes["correct_agent_turnover"] += 1
                if _vote(reviewed) == "wrong":
                    changes["wrong_vote_with_retained_candidate"] += 1
            else:
                changes["last_correct_loss"] += 1
        elif "correct" in reviewed:
            changes["new_correct"] += 1

    return {
        "agent_outcomes": _counts(
            (outcome for panel in panels for outcome in panel[stage]), OUTCOMES
        ),
        "strict_vote_outcomes": _counts((_vote(panel[stage]) for panel in panels), VOTES),
        "correct_available": changes["retained_correct"] + changes["new_correct"],
        "last_correct_loss": changes["last_correct_loss"],
        "retained_correct": changes["retained_correct"],
        "new_correct": changes["new_correct"],
        "same_correct_agent_retained": changes["same_correct_agent_retained"],
        "correct_agent_turnover": changes["correct_agent_turnover"],
        "wrong_vote_with_retained_candidate": changes["wrong_vote_with_retained_candidate"],
        "initial_to_review": {
            before: {after: transitions[before, after] for after in OUTCOMES}
            for before in OUTCOMES
        },
    }


def _summarize_group(panels):
    return {
        "panels": len(panels),
        "initial": {
            "correct_available": sum("correct" in panel["initial"] for panel in panels),
            "agent_outcomes": _counts(
                (outcome for panel in panels for outcome in panel["initial"]), OUTCOMES
            ),
            "strict_vote_outcomes": _counts((_vote(panel["initial"]) for panel in panels), VOTES),
        },
        "private": _review(panels, "private"),
        "peer": _review(panels, "peer"),
        "paired_private_to_peer": {
            "strict_vote": _cross_table(
                ((_vote(panel["private"]), _vote(panel["peer"])) for panel in panels),
                VOTES,
            ),
            "correct_availability": _cross_table(
                ((_availability(panel["private"]), _availability(panel["peer"])) for panel in panels),
                AVAILABILITY,
            ),
            "agent_outcome": _cross_table(
                (pair for panel in panels for pair in zip(panel["private"], panel["peer"], strict=True)),
                OUTCOMES,
            ),
        },
    }


def summarize(panels: list[dict]) -> dict:
    """Count outcomes overall and by initialization. Cross-table rows are the first condition."""
    by_initialization = defaultdict(list)
    all_outcomes = []
    for panel in panels:
        gold = panel["gold"]
        agents = panel["agents"]
        if gold not in ("yes", "no"):
            raise ValueError(f"Invalid gold verdict: {gold!r}")
        if len(agents) != 4 or len({agent["agent_id"] for agent in agents}) != 4:
            raise ValueError("A panel must contain four distinct agents")
        outcomes = {
            stage: tuple(_outcome(agent[stage]["text"], gold) for agent in agents)
            for stage in STAGES
        }
        all_outcomes.append(outcomes)
        by_initialization[panel["initialization"]].append(outcomes)
    return {
        "overall": _summarize_group(all_outcomes),
        "by_initialization": {
            initialization: _summarize_group(by_initialization[initialization])
            for initialization in sorted(by_initialization)
        },
    }
