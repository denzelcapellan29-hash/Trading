#!/usr/bin/env python3
"""Run-to-run fail-closed and immutable-hash tests; no network or credentials."""
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from fast_event_time_ledger import connect, ingest_snapshot, select_completed
from fast_snapshot_continuity import validate_database, restore

class SourceContinuityTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
  self.old=self.base/'prior.sqlite3';self.now=time.time()
  c=connect(self.old)
  ingest_snapshot(c,observed_epoch=self.now-100,source='observed',
     series={('FX:EURUSD','W'):[{'time':self.now-14*86400,'close':1.10},
                                  {'time':self.now-7*86400,'close':1.11}]},batch_id='batch-A')
  c.close()
 def tearDown(self):self.tmp.cleanup()
 def test_valid_prior_and_version_append(self):
  self.assertEqual(validate_database(self.old)['observations'],2)
  dst=self.base/'new.sqlite3';r=restore(self.old,dst,self.base/'receipt.json')
  self.assertEqual(r['batches'],1)
  c=connect(dst)
  ingest_snapshot(c,observed_epoch=self.now-20,source='observed',
    series={('FX:EURUSD','W'):[{'time':self.now-14*86400,'close':1.10},
                                 {'time':self.now-7*86400,'close':1.115}]},batch_id='batch-B')
  before=select_completed(c,symbol='FX:EURUSD',timeframe='W',cutoff_epoch=self.now-80,
                          max_age_seconds=30*86400)
  after=select_completed(c,symbol='FX:EURUSD',timeframe='W',cutoff_epoch=self.now,
                         max_age_seconds=30*86400)
  self.assertEqual(before['batch_id'],'batch-A')
  self.assertEqual(after['batch_id'],'batch-B')
  self.assertEqual(after['start_epoch'],before['start_epoch'])
  c.close()
  self.assertEqual(validate_database(dst)['batches'],2)
 def test_missing_prior_is_hard_block(self):
  with self.assertRaisesRegex(ValueError,'PREDECESSOR_REQUIRED'):
   restore(self.base/'missing.sqlite3',self.base/'target.sqlite3',self.base/'receipt.json')
 def test_corrupted_value_detected_even_if_sqlite_valid(self):
  c=sqlite3.connect(self.old)
  c.execute('UPDATE observations SET close=99 WHERE batch_id=?',('batch-A',));c.commit();c.close()
  with self.assertRaisesRegex(ValueError,'MODIFIED_BAR_VERSION'):
   validate_database(self.old)
 def test_explicit_bootstrap_only(self):
  x=restore(self.base/'missing.sqlite3',self.base/'target.sqlite3',self.base/'receipt.json',bootstrap=True)
  self.assertEqual(x['status'],'EXPLICIT_BOOTSTRAP_ONLY')
  self.assertFalse((self.base/'target.sqlite3').exists())
 def test_never_overwrite_existing_destination(self):
  with self.assertRaisesRegex(ValueError,'DESTINATION_ALREADY_EXISTS'):
   restore(self.old,self.old,self.base/'receipt.json')
if __name__=='__main__':unittest.main(verbosity=2)
