import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from replay_frozen_metals_parity import compare_panels, PANELS, SOURCE_SHA, INPUT_SHA


class MetalReferenceParityUnitTests(unittest.TestCase):
    def setUp(self):
        self.a = pd.DataFrame({'Gold':[0.25, float('nan')], 'Silver':[-0.25,0.0]}, index=['a','b'])
    def test_checklist_has_all_eight_frozen_panels(self):
        self.assertEqual(len(PANELS),8)
        self.assertEqual(len(SOURCE_SHA),64)
        self.assertEqual(len(INPUT_SHA),64)
    def test_identical_finite_and_nan_masks(self):
        self.assertEqual(compare_panels(self.a,self.a)['max_abs_diff'],0.0)
    def test_wrong_nan_pattern_rejected(self):
        b=self.a.copy();b.loc['b','Gold']=1
        with self.assertRaisesRegex(ValueError,'MISSING_VALUE'):
            compare_panels(self.a,b)
    def test_wrong_column_rejected(self):
        with self.assertRaisesRegex(ValueError,'SCHEMA'):
            compare_panels(self.a,self.a.rename(columns={'Gold':'Gold2'}))
    def test_alpha_mismatch_rejected(self):
        b=self.a.copy();b.loc['a','Gold']+=.01
        with self.assertRaisesRegex(ValueError,'NUMERIC'):
            compare_panels(self.a,b)
    def test_small_numeric_tolerance(self):
        b=self.a.copy();b.loc['a','Gold']+=1e-14
        self.assertLess(compare_panels(self.a,b)['max_abs_diff'],1e-12)

if __name__=='__main__':unittest.main()
