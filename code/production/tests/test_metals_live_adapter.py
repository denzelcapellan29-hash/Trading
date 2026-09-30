import importlib.util
import json
import math
from pathlib import Path
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("adapter", ROOT/"tools"/"metals_live_signal_adapter.py")
adapter=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(adapter)

class MetalsLiveAdapterTests(unittest.TestCase):
    def test_epoch_seconds_accepts_ms_and_seconds(self):
        self.assertEqual(adapter._epoch_seconds(1788555600000),1788555600)
        self.assertEqual(adapter._epoch_seconds(1788555600),1788555600)

    def test_frame_requires_confirmed_and_named_plots(self):
        p={"$time":100.0,"E":200000.0,"C":1.0,"X":3.5}
        f=adapter._frame([p],("X",),"C","E")
        self.assertEqual(len(f),1); self.assertEqual(float(f.iloc[0]["X"]),3.5)

    def test_row_hash_is_deterministic_and_contains_no_raw_fixture(self):
        row=pd.Series({k:1.25 for k in adapter.HASH_FIELDS})
        a=adapter._row_hash(123000,row); b=adapter._row_hash(123000,row)
        self.assertEqual(a,b); self.assertEqual(len(a),64)
        fixture=json.loads((ROOT/"config"/"metals_live_input_hash_fixture.json").read_text())
        self.assertFalse(fixture["raw_values_included"]); self.assertEqual(fixture["row_count"],400)

    def test_frozen_overlap_passes_exact_synthetic_hash(self):
        row={"_end_epoch":123.0, **{k:1.0 for k in adapter.HASH_FIELDS}}
        w=pd.DataFrame([row],index=[pd.Timestamp("2026-01-05")])
        h=adapter._row_hash(123000,w.iloc[0])
        p=ROOT/"tests"/"_tmp_hash_fixture.json"
        p.write_text(json.dumps({"source_freeze":adapter.VERSION,"raw_values_included":False,"quantization":".12g","fields":list(adapter.HASH_FIELDS),"row_hashes":{"123000":h},"overall_sha256":"x"}))
        try:
            r=adapter.validate_frozen_overlap(w,p); self.assertTrue(r["passed"])
        finally: p.unlink(missing_ok=True)

if __name__=="__main__": unittest.main()
