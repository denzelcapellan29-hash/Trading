import json
import tempfile
import unittest
from datetime import date,datetime,timezone
from pathlib import Path
from fast_monday_cutoff_shadow import cutoff_for, audit

class CutoffTests(unittest.TestCase):
 def test_europe_london_dst_is_not_fixed_utc(self):
  self.assertEqual(cutoff_for(date(2026,9,28)).hour,8)  # BST
  self.assertEqual(cutoff_for(date(2026,11,2)).hour,9)  # GMT
 def test_reject_non_monday(self):
  with self.assertRaisesRegex(ValueError,'MONDAY'):cutoff_for(date(2026,9,29))
 def test_no_chance_to_backdate_captures(self):
  # An empty DB and full actual manifest must not manufacture 'ready'.
  from fast_event_time_ledger import connect
  with tempfile.TemporaryDirectory() as tmp:
   home=Path(tmp);db=home/'db.sqlite3';c=connect(db);c.close()
   symbols={'freeze_id':'FX-FAST-2026-08-28',
            'fxcm_model_spot':[f'FX:T{i}' for i in range(31)],
            'factor_symbols':[f'TVC:F{i}' for i in range(66)]}
   manifest=home/'manifest.json';manifest.write_text(json.dumps(symbols))
   r=audit(db,manifest,home/'result.json',date(2026,9,28),
           now=datetime(2026,9,29,16,tzinfo=timezone.utc))
   self.assertEqual(r['ready_series'],0)
   self.assertEqual(r['blocked_series'],194)
   self.assertFalse(r['orders_enabled'])
 def test_future_monday_rejected(self):
  with self.assertRaisesRegex(ValueError,'CUTOFF_NOT_YET_REACHED'):
   audit(Path('/nonexistent'),Path('/nonexistent'),Path('/nonexistent'),
         date(2026,11,2),now=datetime(2026,9,29,16,tzinfo=timezone.utc))
if __name__=='__main__':unittest.main(verbosity=2)
