"""No-network three-sleeve freeze and release audit tests."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_three_portfolio_readiness import assess


class ThreeSleeveReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = json.loads((ROOT / "config/three_portfolio_release_manifest.json").read_text())

    def test_scope_exactly_fast_equities_metals(self):
        self.assertEqual(set(self.m["portfolios"]), {"FAST", "EQUITIES", "METALS"})
        self.assertEqual(len(self.m["portfolios"]["METALS"]["research_universe"]), 10)
        self.assertFalse(assess(self.m)["operational_release_approved"])

    def test_never_silently_assign_outer_account_weights(self):
        self.assertIsNone(self.m["account_outer_allocations"])
        self.assertIn("ALLOCATION.unfrozen_or_invalid_three_sleeve_outer_weights",
                      assess(self.m)["paper_and_live_blockers"])

    def test_historical_replay_not_broker_release(self):
        self.assertTrue(self.m["portfolios"]["METALS"]["historical_reference_reproduced"])
        self.assertIn("METALS.instrument_mapping_unresolved", assess(self.m)["paper_and_live_blockers"])

    def test_original_metal_alpha_weights_immutable(self):
        v = copy.deepcopy(self.m)
        v["portfolios"]["METALS"]["frozen_internal_component_weights"]["METALS_RAW_H2"] = .3
        with self.assertRaisesRegex(ValueError, "alpha weights"):
            assess(v)

    def test_omit_any_portfolio_fails_closed(self):
        v = copy.deepcopy(self.m)
        del v["portfolios"]["METALS"]
        with self.assertRaisesRegex(ValueError, "three complete"):
            assess(v)

    def test_bad_or_missing_metal_mapping_cannot_be_implicitly_solved(self):
        v = copy.deepcopy(self.m)
        v["portfolios"]["METALS"]["production_execution_instrument_map"] = None
        self.assertIn("METALS.instrument_mapping_unresolved", assess(v)["paper_and_live_blockers"])

    def test_paper_and_live_never_authorized_by_manifest(self):
        self.assertFalse(assess(self.m, "PAPER")["operational_release_approved"])
        self.assertFalse(assess(self.m, "LIVE")["operational_release_approved"])
        self.assertIn("AUTHORIZATION.live_disabled", assess(self.m, "LIVE")["paper_and_live_blockers"])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            assess(self.m, "AUTO")


if __name__ == "__main__":
    unittest.main()
