#!/usr/bin/env python3
"""Safety invariants; synthetic clocks intentionally avoid historical backdating."""
import tempfile
import time
import unittest
from pathlib import Path
from fast_event_time_ledger import connect, ingest_snapshot, select_completed, decide, certify_calendar_close, UnsafeSource

class EventTimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=connect(Path(self.tmp.name)/'event.sqlite3')
        self.now=time.time()
        self.first=self.now-480
        self.second=self.now-300
        self.decision=self.now-200
        self.oldstart=self.now-1209600
        self.newstart=self.now-604800
    def tearDown(self):self.db.close();self.tmp.cleanup()
    def feed(self,symbol='TEST:ABC',t=None,vals=None,batch=None):
        v=vals if vals is not None else [10.,11.]
        bars=[{'time':self.oldstart,'close':v[0]},{'time':self.newstart,'close':v[1]}]
        return ingest_snapshot(self.db,observed_epoch=self.first if t is None else t,
             source='signed-source',series={(symbol,'W'):bars},batch_id=batch)
    def select(self,at=None,symbol='TEST:ABC'):
        return select_completed(self.db,symbol=symbol,timeframe='W',
                       cutoff_epoch=self.decision if at is None else at,max_age_seconds=20*86400)
    def test_001_complete_only_on_observed_successor(self):
        self.feed();result=self.select()
        self.assertEqual(result['status'],'OK')
        self.assertEqual(result['close'],10.)
        self.assertEqual(result['start_epoch'],self.oldstart)
        self.assertEqual(result['proof'],'NEXT_BAR_OBSERVED')
        self.assertEqual(result['history_proven_count'],1)
    def test_002_missing_late_capture(self):
        self.feed(t=self.now-100)
        self.assertEqual(self.select()['reason'],'NO_OBSERVED_BARS')
    def test_003_idempotent_and_versioned_revisions(self):
        a=self.feed(batch='one');b=self.feed(batch='one')
        self.assertTrue(b['replayed']);self.assertEqual(a['batch_id'],'one')
        with self.assertRaises(UnsafeSource):self.feed(vals=[20,11],batch='one')
        before=self.select()
        self.feed(t=self.now-180,vals=[21.,11.],batch='two')
        self.assertEqual(before['close'],10.)
        self.assertEqual(self.select()['close'],10.) # later version did not exist at original cutoff
        self.assertEqual(self.select(at=self.now-120)['close'],21.)
    def test_004_unverified_current_week_blocked(self):
        self.feed()
        cert=certify_calendar_close(self.db,symbol='TEST:ABC',timeframe='W',
            bar_start_epoch=self.newstart,verified_close_epoch=self.first-5,
            issued_epoch=self.first+1,source='calendar-attestor',evidence_id='test-close-1')
        before=self.select(at=self.first)
        self.assertEqual(before['start_epoch'],self.oldstart)
        after=self.select(at=self.now+1)
        self.assertEqual(after['start_epoch'],self.newstart)
        self.assertTrue(after['proof'].startswith('VERIFIED_CALENDAR:'))
    def test_005_reject_bad_or_future_certificate(self):
        with self.assertRaises(UnsafeSource):
            certify_calendar_close(self.db,symbol='TEST:ABC',timeframe='W',
                bar_start_epoch=self.newstart,verified_close_epoch=self.now+100,
                issued_epoch=self.now+200,source='claim',evidence_id='future')
    def test_006_missing_source_fails_entire_cohort(self):
        self.feed()
        r=decide(self.db,decision_id='test-1',symbols=['TEST:ABC','TEST:MISSING'],
                 timeframe='W',cutoff_epoch=self.decision,max_age_seconds=20*86400)
        self.assertEqual(r['status'],'BLOCK');self.assertFalse(r['orders_enabled'])
        self.assertEqual(len(self.db.execute('SELECT * FROM decisions').fetchall()),2)
        with self.assertRaises(UnsafeSource):
            decide(self.db,decision_id='test-1',symbols=['TEST:ABC'],timeframe='W',
                   cutoff_epoch=self.decision,max_age_seconds=20*86400)
    def test_007_prevent_stale_reuse(self):
        self.feed()
        r=select_completed(self.db,symbol='TEST:ABC',timeframe='W',
                           cutoff_epoch=self.decision,max_age_seconds=60)
        self.assertEqual(r['reason'],'STALE_COMPLETED_SOURCE')
    def test_010_old_import_is_not_fresh_even_if_observed_now(self):
        old_a=self.now-42*86400;old_b=self.now-35*86400
        ingest_snapshot(self.db,observed_epoch=self.first,source='archive-ingestion',
            series={('TEST:OLD','W'):[{'time':old_a,'close':100},
                     {'time':old_b,'close':101}]})
        result=select_completed(self.db,symbol='TEST:OLD',timeframe='W',
            cutoff_epoch=self.decision,max_age_seconds=12*86400)
        self.assertEqual(result['reason'],'STALE_COMPLETED_SOURCE')
    def test_011_certificate_insertion_not_retroactively_visible(self):
        self.feed()
        certify_calendar_close(self.db,symbol='TEST:ABC',timeframe='W',
           bar_start_epoch=self.newstart,verified_close_epoch=self.first-5,
           issued_epoch=self.first+1,source='calendar-attestor',evidence_id='late-backdate')
        self.assertEqual(self.select()['start_epoch'],self.oldstart)
        self.assertEqual(self.select(self.now+1)['start_epoch'],self.newstart)
    def test_009_signed_macro_factor_but_positive_fx_spot(self):
        row={'time':self.oldstart,'close':-0.25}
        ingest_snapshot(self.db,observed_epoch=self.first,source='signed-source',
              series={('TVC:BTPBUND','W'):[row]})
        with self.assertRaises(UnsafeSource):
            ingest_snapshot(self.db,observed_epoch=self.first,source='signed-source',
                series={('FX:EURUSD','W'):[row]})
    def test_008_no_hindsight_certificate(self):
        self.feed()
        certify_calendar_close(self.db,symbol='TEST:ABC',timeframe='W',
            bar_start_epoch=self.newstart,verified_close_epoch=self.first-5,
            issued_epoch=self.now-100,source='calendar-attestor',evidence_id='late')
        self.assertEqual(self.select()['start_epoch'],self.oldstart)
        self.assertEqual(self.select(self.now+1)['start_epoch'],self.newstart)

if __name__=='__main__':unittest.main(verbosity=2)
