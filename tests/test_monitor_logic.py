import unittest

from src.monitoring.monitor_logic import (
    decide_cat_name,
    pick_best_detection,
    should_confirm_exit,
)


class MonitorLogicTests(unittest.TestCase):
    def test_pick_best_detection_prefers_identity_classes(self):
        detections = [
            {"class_name": "cat", "confidence": 0.95},
            {"class_name": "bagel", "confidence": 0.80},
            {"class_name": "kurumi", "confidence": 0.85},
        ]

        best = pick_best_detection(detections, identity_classes=["bagel", "kurumi"])
        self.assertEqual(best["class_name"], "kurumi")

    def test_decide_cat_name_uses_max_vote(self):
        votes = {"bagel": 0.9, "kurumi": 0.6}
        self.assertEqual(decide_cat_name(votes), "BAGEL")

    def test_should_confirm_exit_requires_empty_state(self):
        self.assertFalse(
            should_confirm_exit(
                empty_probability=0.30,
                empty_threshold=0.70,
                empty_duration=3.0,
                empty_confirm_seconds=2.0,
            )
        )
        self.assertTrue(
            should_confirm_exit(
                empty_probability=0.90,
                empty_threshold=0.70,
                empty_duration=2.5,
                empty_confirm_seconds=2.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
