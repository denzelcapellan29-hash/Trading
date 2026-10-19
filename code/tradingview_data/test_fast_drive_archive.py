#!/usr/bin/env python3
"""No-credential source archive invariants; fake Drive persists file bytes only."""
import json
import shutil
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from fast_event_time_ledger import connect
from fast_snapshot_continuity import canonical_hash,validate_database,sha256
from fast_drive_archive import (ArchiveBlocked, parse_chain,filename,make_delta,
    apply_delta,logical_digest,seed_archive,restore_archive,verify_parent,publish_delta,reconcile_github)

class FakeDrive:
    def __init__(self,folder:Path):self.folder=folder;self.folder.mkdir(parents=True)
    def list(self):return [{'name':p.name,'id':p.name} for p in self.folder.iterdir() if p.is_file()]
    def download(self,file_id,path):shutil.copyfile(self.folder/file_id,path)
    def create(self,name,path):
        target=self.folder/name
        if target.exists():raise ArchiveBlocked('DUPLICATE_IMMUTABLE_OBJECT')
        shutil.copyfile(path,target)
        return name

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.prior=self.base/'prior.sqlite3';self.current=self.base/'current.sqlite3'
        self.store=FakeDrive(self.base/'fake-drive')
        self.now=time.time()
        self._record(self.prior,'B1',1.1000)
    def tearDown(self):self.tmp.cleanup()
    def _record(self,path,batch_id,close):
        # Use exactly the live ledger schema and original hash canonicalization.
        c=connect(path)
        original=self.now-14*86400
        px={'open':None,'high':None,'low':None,'close':close,'volume':None}
        symbol='FX:EURUSD';tf='W'
        ver=canonical_hash([symbol,tf,original,px])
        payload=[(symbol,tf,original,px,ver)]
        observed=self.now-2 if batch_id=='B1' else self.now-1
        with c:
            c.execute('INSERT INTO batches VALUES (?,?,?,?,?)',
                (batch_id,observed,canonical_hash(['signed-source',payload]),'signed-source',observed))
            c.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?)',
                (batch_id,symbol,tf,original,None,None,None,close,None,ver))
        c.close()
    def _next(self):shutil.copyfile(self.prior,self.current);self._record(self.current,'B2',1.1025)
    def _seed(self):return seed_archive(self.store,self.prior,sha256(self.prior))
    def test_seed_delta_restore_and_verifiable_parents(self):
        self._next();seed=self._seed()
        self.assertEqual(seed['status'],'SEED_PUBLISHED_VERIFIED')
        self.assertEqual(verify_parent(self.store,self.prior)['batch_count'],1)
        p=publish_delta(self.store,self.prior,self.current,seed['head_sha256'])
        self.assertEqual(p['batch_count'],2)
        self.assertEqual(p['delta_observations'],1)
        dest=self.base/'restored.sqlite3'
        restored=restore_archive(self.store,dest)
        self.assertEqual(restored['batch_count'],2)
        self.assertEqual(logical_digest(dest),logical_digest(self.current))
    def test_cannot_bootstrap_without_exact_approved_sha(self):
        with self.assertRaisesRegex(ArchiveBlocked,'SUPERVISED_BOOTSTRAP_SHA_MISMATCH'):
            seed_archive(self.store,self.prior,'0'*64)
        self.assertFalse(self.store.list())
    def test_no_fallback_when_missing_seed(self):
        with self.assertRaisesRegex(ArchiveBlocked,'NO_REMOTE_SEED'):
            restore_archive(self.store,self.base/'dest.sqlite3')
    def test_remote_corruption_halts_before_merge(self):
        self._seed()
        remote=next(self.store.folder.iterdir());remote.write_bytes(b'not SQLite')
        with self.assertRaisesRegex(ArchiveBlocked,'REMOTE_SEGMENT_SHA256_MISMATCH'):
            restore_archive(self.store,self.base/'dest.sqlite3')
    def test_fork_or_gap_is_blocked(self):
        self._seed()
        entry=parse_chain(self.store.list())[0]
        rogue=filename(0,1,'root','b'*64)
        (self.store.folder/rogue).write_bytes(b'rogue')
        with self.assertRaisesRegex(ArchiveBlocked,'ARCHIVE_SEQUENCE_GAP_OR_FORK'):
            restore_archive(self.store,self.base/'dest.sqlite3')
        self.assertEqual(entry['seq'],0)
    def test_prior_certificate_modified_even_if_bar_hash_still_valid(self):
        self._next()
        c=sqlite3.connect(self.prior)
        with c:
            c.execute('INSERT INTO close_certificates VALUES (?,?,?,?,?,?,?,?,?)',
               ('cert-a','FX:EURUSD','W',1.,2.,2.,2.,'calendar','evidence'))
        c.close()
        with self.assertRaisesRegex(ArchiveBlocked,'PRIOR_LEDGER_ROW_MUTATED:close_certificates'):
            make_delta(self.prior,self.current,self.base/'delta.sqlite3')
    def test_unverified_parent_and_wrong_head_block(self):
        self._next();seed=self._seed()
        with self.assertRaisesRegex(ArchiveBlocked,'REMOTE_HEAD_CHANGED_BETWEEN_STEPS'):
            publish_delta(self.store,self.prior,self.current,'0'*64)
        with self.assertRaisesRegex(ArchiveBlocked,'GITHUB_PREDECESSOR_DIFFERS_FROM_PERMANENT_ARCHIVE'):
            verify_parent(self.store,self.current)
    def test_github_ahead_of_drive_fails_closed(self):
        self._next();self._seed()
        with self.assertRaisesRegex(ArchiveBlocked,'GITHUB_OBSERVED_LEDGER_AHEAD_OF_DRIVE'):
            reconcile_github(self.store,self.current,self.prior)
    def test_missing_github_artifact_allows_independent_durable_restore(self):
        self._seed()
        r=reconcile_github(self.store,None,self.prior)
        self.assertEqual(r['status'],'NO_GITHUB_COMPARATOR_PERMANENT_PROOF_ONLY')
    def test_no_equal_batchcount_delta(self):
        shutil.copyfile(self.prior,self.current)
        with self.assertRaisesRegex(ArchiveBlocked,'EXPECTED_EXACTLY_ONE_NEW_BATCH'):
            make_delta(self.prior,self.current,self.base/'delta.sqlite3')
    def test_duplicate_prior_or_extra_unrelated_archive_name_blocks(self):
        self._seed();(self.store.folder/'fast-source-archive-v1-seg-bad').write_bytes(b'bad')
        with self.assertRaisesRegex(ArchiveBlocked,'UNRECOGNIZED_ARCHIVE_OBJECT'):
            restore_archive(self.store,self.base/'dest.sqlite3')

if __name__=='__main__':unittest.main(verbosity=2)
