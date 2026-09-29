#!/usr/bin/env python3
"""Fail-closed FAST *shadow* ledger archive: immutable Drive seed + linear deltas.

Never backdates source observations; it transfers SQLite rows *as recorded*.
No credentials, prices, or raw bars are logged. No broker interface exists.
An archive built by this tool is operationally append-only but Google Drive is
NOT WORM storage. A published SHA anchor should be retained independently.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from fast_snapshot_continuity import validate_database, sha256

PREFIX = 'fast-source-archive-v1-seg-'
NAME = re.compile(r'^fast-source-archive-v1-seg-(\d{6})-b(\d{3})-(root|[0-9a-f]{64})-([0-9a-f]{64})\.sqlite3$')
TABLES = ('batches', 'observations', 'close_certificates', 'decisions')
MAX_SEGMENTS = 500  # require reviewed compaction instead of unbounded restoration

class ArchiveBlocked(ValueError): pass

def filename(seq: int, batches: int, parent: str, digest: str) -> str:
    if seq < 0 or batches < 1 or not re.fullmatch(r'(root|[0-9a-f]{64})',parent):
        raise ArchiveBlocked('INVALID_ARCHIVE_NAME')
    return f'{PREFIX}{seq:06d}-b{batches:03d}-{parent}-{digest}.sqlite3'

def parse_chain(files: list[dict]) -> list[dict]:
    parsed=[]
    for f in files:
        name=f.get('name','')
        if not name.startswith(PREFIX):continue
        m=NAME.fullmatch(name)
        if not m:raise ArchiveBlocked('UNRECOGNIZED_ARCHIVE_OBJECT')
        seq,batches,parent,digest=m.groups()
        parsed.append(dict(seq=int(seq),batches=int(batches),parent=parent,
                           digest=digest,id=f['id'],name=name))
    parsed.sort(key=lambda x:x['seq'])
    if len(parsed)>MAX_SEGMENTS:raise ArchiveBlocked('ARCHIVE_REQUIRES_REVIEWED_COMPACTION')
    for i,f in enumerate(parsed):
        if f['seq']!=i:raise ArchiveBlocked('ARCHIVE_SEQUENCE_GAP_OR_FORK')
        if i==0:
            if f['parent']!='root' or f['batches']<1:raise ArchiveBlocked('INVALID_ARCHIVE_SEED')
        elif f['parent']!=parsed[i-1]['digest'] or f['batches']!=parsed[i-1]['batches']+1:
            raise ArchiveBlocked('ARCHIVE_PARENT_OR_BATCH_FORK')
    return parsed

def sql_connect(path: Path, *, readonly: bool) -> sqlite3.Connection:
    con=sqlite3.connect(f'file:{path.resolve()}?mode={"ro" if readonly else "rw"}',uri=True)
    con.execute('PRAGMA foreign_keys=ON')
    return con

def logical_digest(path: Path) -> str:
    """Stable digest of all material row values, independent of SQLite page layout."""
    validate_database(path)
    h=hashlib.sha256(b'FAST_ARCHIVE_LOGICAL_V1\n')
    con=sql_connect(path,readonly=True)
    try:
        for table in TABLES:
            h.update((table+'\n').encode())
            cols=[x[1] for x in con.execute('PRAGMA table_info('+table+')')]
            order=','.join(f'"{x}"' for x in cols)  # canonical total ordering
            for row in con.execute(f'SELECT * FROM {table} ORDER BY {order}'):
                h.update(json.dumps(row,allow_nan=False,separators=(',',':'),ensure_ascii=True).encode()+b'\n')
    finally:con.close()
    return h.hexdigest()

def _table_rows(con: sqlite3.Connection, table: str, db: str) -> int:
    return int(con.execute(f'SELECT COUNT(*) FROM {db}.{table}').fetchone()[0])

def make_delta(previous: Path, current: Path, output: Path) -> dict:
    """Export only the newly recorded batch and ancillary rows; reject ANY old-row edit."""
    validate_database(previous);validate_database(current)
    if output.exists():raise ArchiveBlocked('DELTA_DESTINATION_EXISTS')
    output.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(':memory:')
    try:
        con.execute('ATTACH DATABASE ? AS old',(str(previous.resolve()),))
        con.execute('ATTACH DATABASE ? AS new',(str(current.resolve()),))
        for table in TABLES:
            bad=con.execute(f'SELECT 1 FROM (SELECT * FROM old.{table} EXCEPT SELECT * FROM new.{table}) LIMIT 1').fetchone()
            if bad:raise ArchiveBlocked('PRIOR_LEDGER_ROW_MUTATED:'+table)
        old_batch=_table_rows(con,'batches','old')
        new_batch=_table_rows(con,'batches','new')
        if new_batch!=old_batch+1:raise ArchiveBlocked('EXPECTED_EXACTLY_ONE_NEW_BATCH')
        new_ids=[r[0] for r in con.execute('SELECT batch_id FROM new.batches EXCEPT SELECT batch_id FROM old.batches')]
        if len(new_ids)!=1:raise ArchiveBlocked('BATCH_ID_CHANGE_DETECTED')
        # Create an empty ledger using schema copied by sqlite3 backup of original
        # empty schema rather than reintroducing synthetic capture timestamps.
        from fast_event_time_ledger import connect
        empty=connect(output);empty.close()
        dest=sqlite3.connect(str(output));dest.execute('PRAGMA foreign_keys=ON')
        try:
            dest.execute('ATTACH DATABASE ? AS source',(str(current.resolve()),))
            with dest:
                for table in TABLES:
                    # Only rows absent in the predecessor; exact row subtraction
                    # is safe because an old-row mutation has already blocked.
                    columns=[v[1] for v in dest.execute(f'PRAGMA source.table_info({table})')]
                    col=','.join('"'+x+'"' for x in columns)
                    pk={'batches':['batch_id'],'observations':['batch_id','symbol','timeframe','bar_start_epoch'],
                        'close_certificates':['certificate_id'],
                        'decisions':['decision_id','symbol','timeframe']}[table]
                    where=' AND '.join(f'o."{c}"=n."{c}"' for c in pk)
                    dest.execute('ATTACH DATABASE ? AS predecessor',(str(previous.resolve()),)) if table==TABLES[0] else None
                    dest.execute(f'INSERT INTO main.{table} ({col}) SELECT {col} FROM source.{table} n '
                                 f'WHERE NOT EXISTS (SELECT 1 FROM predecessor.{table} o WHERE {where})')
        finally:dest.close()
    except Exception:
        if output.exists():output.unlink()
        raise
    finally:con.close()
    meta=validate_database(output)
    if meta['batches']!=1:output.unlink();raise ArchiveBlocked('DELTA_BATCH_COUNT_INVALID')
    return {'old_batches':old_batch,'new_batches':new_batch,'delta_observations':meta['observations'],
            'delta_sha256':meta['sha256']}

def apply_delta(root: Path, delta: Path) -> None:
    """Preserve original batch observation epochs without calling live-only ingestion."""
    validate_database(root)
    info=validate_database(delta)
    if info['batches']!=1:raise ArchiveBlocked('DELTA_NOT_ONE_BATCH')
    con=sql_connect(root,readonly=False)
    try:
        con.execute('ATTACH DATABASE ? AS patch',(str(delta.resolve()),))
        with con:
            for table in TABLES:
                # No silent overwrites of recorded versions/decisions.
                con.execute(f'INSERT INTO main.{table} SELECT * FROM patch.{table}')
    finally:con.close()
    validate_database(root)

class DriveStore:
    """Dedicated private Trading Drive data/ folder via OAuth user consent.

    Requires an already-authorized refresh token for Google Drive scope and
    an explicitly configured exact folder ID. No token values are printed.
    """
    def __init__(self,client_id,client_secret,refresh_token,folder_id):
        if not all((client_id,client_secret,refresh_token,folder_id)):
            raise ArchiveBlocked('DRIVE_OAUTH_OR_FOLDER_NOT_CONFIGURED')
        self.folder=folder_id
        body=urllib.parse.urlencode({'client_id':client_id,'client_secret':client_secret,
                        'refresh_token':refresh_token,'grant_type':'refresh_token'}).encode()
        req=urllib.request.Request('https://oauth2.googleapis.com/token',data=body,method='POST')
        try:
            with urllib.request.urlopen(req,timeout=25) as resp:token=json.load(resp).get('access_token')
        except Exception as exc:
            raise ArchiveBlocked('DRIVE_OAUTH_TOKEN_EXCHANGE_FAILED') from None
        if not token:raise ArchiveBlocked('DRIVE_OAUTH_ACCESS_TOKEN_MISSING')
        self._token=token
    def _request(self,url,*,data=None,content_type=None):
        headers={'Authorization':'Bearer '+self._token}
        if content_type:headers['Content-Type']=content_type
        request=urllib.request.Request(url,data=data,headers=headers,method='POST' if data is not None else 'GET')
        try:
            with urllib.request.urlopen(request,timeout=100) as res:return res.read()
        except Exception:
            raise ArchiveBlocked('DRIVE_ARCHIVE_IO_FAILED') from None
    def list(self):
        records=[];page=None
        while True:
            q=f"'{self.folder}' in parents and trashed = false and name contains '{PREFIX}'"
            params={'q':q,'fields':'nextPageToken,files(id,name)','pageSize':'1000',
                    'supportsAllDrives':'true','includeItemsFromAllDrives':'true'}
            if page:params['pageToken']=page
            raw=self._request('https://www.googleapis.com/drive/v3/files?'+urllib.parse.urlencode(params))
            doc=json.loads(raw)
            records+=doc.get('files',[]);page=doc.get('nextPageToken')
            if not page:break
        return records
    def download(self,file_id,path:Path):
        data=self._request('https://www.googleapis.com/drive/v3/files/'+urllib.parse.quote(file_id,safe='')+'?alt=media&supportsAllDrives=true')
        path.write_bytes(data)
    def create(self,name:str,path:Path):
        meta=json.dumps({'name':name,'parents':[self.folder]},separators=(',',':')).encode()
        boundary='FastArchiveCheckedV1_918281'
        body=(b'--'+boundary.encode()+b'\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n'+meta+
              b'\r\n--'+boundary.encode()+b'\r\nContent-Type: application/x-sqlite3\r\n\r\n'+
              path.read_bytes()+b'\r\n--'+boundary.encode()+b'--\r\n')
        raw=self._request('https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name',
                     data=body,content_type='multipart/related; boundary='+boundary)
        result=json.loads(raw)
        if result.get('name')!=name or not result.get('id'):raise ArchiveBlocked('DRIVE_UPLOAD_ACK_MISMATCH')
        return result['id']

def list_checked(store):return parse_chain(store.list())

def read_verified(store, entry, target:Path):
    store.download(entry['id'],target)
    if sha256(target)!=entry['digest']:
        raise ArchiveBlocked('REMOTE_SEGMENT_SHA256_MISMATCH')
    validate_database(target)

def restore_archive(store,output:Path)->dict:
    if output.exists():raise ArchiveBlocked('RESTORE_WOULD_OVERWRITE')
    chain=list_checked(store)
    if not chain:raise ArchiveBlocked('NO_REMOTE_SEED')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        try:
            for entry in chain:
                seg=work/f'{entry["seq"]:06d}.sqlite3'
                read_verified(store,entry,seg)
                if entry['seq']==0:shutil.copyfile(seg,output)
                else:apply_delta(output,seg)
            info=validate_database(output)
            if info['batches']!=chain[-1]['batches']:
                raise ArchiveBlocked('REMOTE_RESTORED_BATCHCOUNT_MISMATCH')
        except Exception:
            if output.exists():output.unlink()
            raise
    return {'status':'REMOTE_RESTORED_VERIFIED','head_sha256':chain[-1]['digest'],
            'segments':len(chain),'batch_count':info['batches'],
            'logical_sha256':logical_digest(output),'orders_enabled':False}


def reconcile_github(store,github_predecessor:Path|None,remote:Path)->dict:
    """Never abandon a later GH-observed batch when switching to permanent Drive.

    If the optional latest complete GH artifact is available, every retained row
    must be a subset of the independently restored Drive archive. If it is not
    available (expired), the verified Drive chain remains an independent proof.
    """
    validate_database(remote)
    if github_predecessor is None or not github_predecessor.exists():
        return {'status':'NO_GITHUB_COMPARATOR_PERMANENT_PROOF_ONLY','orders_enabled':False}
    g=validate_database(github_predecessor)
    d=validate_database(remote)
    if g['batches']>d['batches']:
        raise ArchiveBlocked('GITHUB_OBSERVED_LEDGER_AHEAD_OF_DRIVE')
    con=sqlite3.connect(':memory:')
    try:
        con.execute('ATTACH DATABASE ? AS gh',(str(github_predecessor.resolve()),))
        con.execute('ATTACH DATABASE ? AS permanent',(str(remote.resolve()),))
        for table in TABLES:
            missing=con.execute(f'SELECT 1 FROM (SELECT * FROM gh.{table} EXCEPT SELECT * FROM permanent.{table}) LIMIT 1').fetchone()
            if missing:raise ArchiveBlocked('GITHUB_AND_PERMANENT_LEDGER_DIVERGED:'+table)
    finally:con.close()
    return {'status':'GITHUB_PREDECESSOR_IS_VERIFIED_PERMANENT_PREFIX',
            'github_batches':g['batches'],'permanent_batches':d['batches'],
            'orders_enabled':False}

def _check_unique(store,name,expected_sha):
    matches=[x for x in store.list() if x.get('name')==name]
    if len(matches)>1:raise ArchiveBlocked('DUPLICATE_REMOTE_IMMUTABLE_NAME')
    if len(matches)==1:
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'existing.sqlite3';store.download(matches[0]['id'],target)
            if sha256(target)!=expected_sha:raise ArchiveBlocked('IMMUTABLE_REMOTE_NAME_REPLACED')
        return matches[0]['id']
    return None

def _put_checked(store,name,path,sha):
    file_id=_check_unique(store,name,sha)
    if file_id is None:file_id=store.create(name,path)
    with tempfile.TemporaryDirectory() as tmp:
        target=Path(tmp)/'readback.sqlite3';store.download(file_id,target)
        if sha256(target)!=sha:raise ArchiveBlocked('UPLOAD_READBACK_CHECKSUM_FAILED')
        validate_database(target)
    return file_id

def seed_archive(store,db:Path,approved_sha:str)->dict:
    info=validate_database(db)
    if not re.fullmatch('[0-9a-f]{64}',approved_sha) or info['sha256']!=approved_sha:
        raise ArchiveBlocked('SUPERVISED_BOOTSTRAP_SHA_MISMATCH')
    if list_checked(store):raise ArchiveBlocked('REMOTE_ARCHIVE_ALREADY_EXISTS')
    name=filename(0,info['batches'],'root',approved_sha)
    file_id=_put_checked(store,name,db,approved_sha)
    if len(list_checked(store))!=1:raise ArchiveBlocked('SEED_POST_UPLOAD_CHAIN_INVALID')
    return {'status':'SEED_PUBLISHED_VERIFIED','object_id':file_id,'head_sha256':approved_sha,
            'batch_count':info['batches'],'orders_enabled':False}

def verify_parent(store,local:Path)->dict:
    info=validate_database(local)
    with tempfile.TemporaryDirectory() as tmp:
        remote=Path(tmp)/'restored.sqlite3'
        r=restore_archive(store,remote)
        if r['batch_count']!=info['batches'] or logical_digest(local)!=r['logical_sha256']:
            raise ArchiveBlocked('GITHUB_PREDECESSOR_DIFFERS_FROM_PERMANENT_ARCHIVE')
    return {'status':'PERMANENT_ARCHIVE_MATCHES_RUN_PREDECESSOR','head_sha256':r['head_sha256'],
            'batch_count':r['batch_count'],'segments':r['segments'],'orders_enabled':False}

def publish_delta(store,previous:Path,current:Path,expected_head:str)->dict:
    chain=list_checked(store)
    if not chain:raise ArchiveBlocked('NO_REMOTE_SEED')
    if chain[-1]['digest']!=expected_head:raise ArchiveBlocked('REMOTE_HEAD_CHANGED_BETWEEN_STEPS')
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);remote=tmp/'archive.sqlite3'
        restore_archive(store,remote)
        if logical_digest(remote)!=logical_digest(previous):
            raise ArchiveBlocked('PARENT_NOT_EQUIVALENT_TO_REMOTE_ARCHIVE')
        delta=tmp/'delta.sqlite3';meta=make_delta(previous,current,delta)
        if meta['new_batches']!=chain[-1]['batches']+1:raise ArchiveBlocked('REMOTE_BATCHCOUNT_NOT_SUCCESSOR')
        check=tmp/'rebuilt.sqlite3';shutil.copyfile(remote,check);apply_delta(check,delta)
        if logical_digest(check)!=logical_digest(current):raise ArchiveBlocked('DELTA_RECONSTRUCTION_MISMATCH')
        # Re-read listing immediately before upload. Repository Actions runs
        # additionally need concurrency serialized by their GitHub workflow.
        if list_checked(store)[-1]['digest']!=expected_head:raise ArchiveBlocked('REMOTE_HEAD_CHANGED_PRE_UPLOAD')
        name=filename(chain[-1]['seq']+1,meta['new_batches'],expected_head,meta['delta_sha256'])
        file_id=_put_checked(store,name,delta,meta['delta_sha256'])
    verified=list_checked(store)
    if verified[-1]['digest']!=meta['delta_sha256'] or len(verified)!=len(chain)+1:
        raise ArchiveBlocked('POST_PUBLISH_CHAIN_INVALID')
    return {'status':'PERMANENT_DELTA_PUBLISHED_VERIFIED','object_id':file_id,
            'head_sha256':meta['delta_sha256'],'batch_count':meta['new_batches'],
            'delta_observations':meta['delta_observations'],'orders_enabled':False}

def get_store():
    return DriveStore(*(os.environ.get(k,'') for k in ('FAST_DRIVE_CLIENT_ID',
        'FAST_DRIVE_CLIENT_SECRET','FAST_DRIVE_REFRESH_TOKEN','FAST_DRIVE_FOLDER_ID')))

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('seed','verify-parent','publish','restore','reconcile-github'))
    p.add_argument('--db',type=Path);p.add_argument('--prev',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--receipt',type=Path)
    p.add_argument('--approved-sha');p.add_argument('--expected-head')
    args=p.parse_args(argv)
    try:
        store=get_store()
        if args.operation=='seed': result=seed_archive(store,args.db,args.approved_sha or '')
        elif args.operation=='verify-parent':result=verify_parent(store,args.db)
        elif args.operation=='restore':result=restore_archive(store,args.out)
        elif args.operation=='reconcile-github':result=reconcile_github(store,args.prev,args.db)
        else:result=publish_delta(store,args.prev,args.db,args.expected_head or '')
        if args.receipt:
            args.receipt.parent.mkdir(parents=True,exist_ok=True)
            args.receipt.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,sort_keys=True))
        return 0
    except (ArchiveBlocked,sqlite3.Error,OSError,ValueError) as exc:
        # No raw network error, token, filenames or data in logs.
        print('BLOCK: '+str(exc) if isinstance(exc,ArchiveBlocked) else 'BLOCK: ARCHIVE_CHECK_FAILED',file=sys.stderr)
        return 4

if __name__=='__main__':sys.exit(main())
