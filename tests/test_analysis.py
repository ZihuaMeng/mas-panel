import unittest

from mas_panel.analysis import summarize


def panel(initial, private, peer, *, gold="yes", initialization="neutral"):
    return {
        "gold": gold,
        "initialization": initialization,
        "agents": [
            {
                "agent_id": index,
                "initial": {"text": before},
                "private": {"text": own},
                "peer": {"text": others},
            }
            for index, (before, own, others) in enumerate(zip(initial, private, peer, strict=True))
        ],
    }


class SummarizeTests(unittest.TestCase):
    def test_panel_availability_agent_turnover_and_majority_are_distinct(self):
        result = summarize([
            panel(
                ("yes", "no", "Unknown", "bad"),
                ("no", "no", "yes", "no"),
                ("yes", "yes", "yes", "unknown"),
            ),
            panel(
                ("no", "yes", "no", "no"),
                ("no", "no", "no", "no"),
                ("no", "yes", "no", "no"),
            ),
            panel(
                ("no", "no", "unknown", "bad"),
                ("yes", "yes", "unknown", "bad"),
                ("no", "no", "no", "no"),
                initialization="complementary",
            ),
        ])
        overall = result["overall"]
        self.assertEqual(overall["panels"], 3)
        self.assertEqual(overall["initial"]["correct_available"], 2)
        private, peer = overall["private"], overall["peer"]
        self.assertEqual(private["correct_available"], 2)
        self.assertEqual(private["last_correct_loss"], 1)
        self.assertEqual(private["retained_correct"], 1)
        self.assertEqual(private["new_correct"], 1)
        self.assertEqual(private["same_correct_agent_retained"], 0)
        self.assertEqual(private["correct_agent_turnover"], 1)
        self.assertEqual(private["wrong_vote_with_retained_candidate"], 1)
        self.assertEqual(peer["retained_correct"], 2)
        self.assertEqual(peer["same_correct_agent_retained"], 2)
        self.assertEqual(peer["correct_agent_turnover"], 0)
        self.assertEqual(peer["wrong_vote_with_retained_candidate"], 1)
        self.assertEqual(private["strict_vote_outcomes"], {"correct": 0, "wrong": 2, "no_decision": 1})
        self.assertEqual(peer["strict_vote_outcomes"], {"correct": 1, "wrong": 2, "no_decision": 0})
        self.assertEqual(private["initial_to_review"]["unknown"]["correct"], 1)
        self.assertEqual(private["initial_to_review"]["unparseable"]["unparseable"], 1)
        self.assertEqual(result["by_initialization"]["neutral"]["panels"], 2)
        self.assertEqual(result["by_initialization"]["complementary"]["private"]["new_correct"], 1)

        paired = overall["paired_private_to_peer"]
        self.assertEqual(paired["strict_vote"]["wrong"]["correct"], 1)
        self.assertEqual(paired["strict_vote"]["no_decision"]["wrong"], 1)
        self.assertEqual(paired["correct_availability"]["unavailable"]["available"], 1)
        self.assertEqual(paired["correct_availability"]["available"]["unavailable"], 1)
        self.assertEqual(paired["agent_outcome"]["unknown"]["wrong"], 1)
        self.assertEqual(paired["agent_outcome"]["unparseable"]["wrong"], 1)

    def test_no_is_correct_when_gold_is_no(self):
        result = summarize([
            panel(
                ("No", "NO", "yes", "unknown"),
                ("no", "yes", "no", "unknown"),
                (" no ", "no", "no", "No."),
                gold="no",
            ),
        ])["overall"]
        self.assertEqual(result["private"]["strict_vote_outcomes"]["no_decision"], 1)
        self.assertEqual(result["peer"]["strict_vote_outcomes"]["correct"], 1)
        self.assertEqual(result["peer"]["agent_outcomes"]["unparseable"], 1)

    def test_empty_result_has_zero_denominators(self):
        result = summarize([])
        self.assertEqual(result["overall"]["panels"], 0)
        self.assertEqual(result["by_initialization"], {})


if __name__ == "__main__":
    unittest.main()
