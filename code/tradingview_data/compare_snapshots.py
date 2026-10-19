#!/usr/bin/env python3
"""Compare two actually observed, immutable FAST API snapshots without rewriting either."""
import argparse,io,json,sqlite3,tempfile,zipfile
from pathlib import Path
import pandas as pd

def snapshot(path):
    with zipfile.ZipFile(path) as z:
        meta=json.loads(z.read('status.json'))
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'snapshot.db';db.write_bytes(z.read('source_versions.sqlite3'))
            con=sqlite3.connect(db)
            rows=pd.read_sql_query('select symbol,timeframe,bar_start_epoch,open,high,low,close,volume,version_sha256 from observations',con)
            con.close()
    return meta,rows

def compare(older,newer,out):
    out.mkdir(parents=True,exist_ok=True)
    a,x=snapshot(older);b,y=snapshot(newer)
    joined=x.merge(y,on=['symbol','timeframe','bar_start_epoch'],suffixes=('_old','_new'),how='outer',indicator=True)
    matched=joined[joined._merge=='both'].copy()
    changed=matched[matched.version_sha256_old!=matched.version_sha256_new].copy()
    maxstarts=matched.groupby(['symbol','timeframe']).bar_start_epoch.max().rename('newest_start')
    changed=changed.join(maxstarts,on=['symbol','timeframe']);changed['is_latest_chart_bar']=changed.bar_start_epoch==changed.newest_start
    for field in ('open','high','low','close','volume'):
        changed[field+'_changed']=((changed[field+'_old'].fillna(0)-changed[field+'_new'].fillna(0)).abs()>1e-12)
    changed.to_csv(out/'snapshot_version_diffs.csv',index=False)
    result={'older_captured_utc':a['observed_utc'],'newer_captured_utc':b['observed_utc'],
       'older_series':a['captured_series'],'newer_series':b['captured_series'],
       'overlapping_bars':len(matched),'version_changes':len(changed),
       'changed_latest_chart_bars':int(changed.is_latest_chart_bar.sum()),
       'changed_nonlatest_chart_bars':int((~changed.is_latest_chart_bar).sum()),
       'changed_nonlatest_close':int((changed.loc[~changed.is_latest_chart_bar,'close_changed']).sum()),
       'field_change_counts':{field:{'latest':int(changed.loc[changed.is_latest_chart_bar,field+'_changed'].sum()),
                                'nonlatest':int(changed.loc[~changed.is_latest_chart_bar,field+'_changed'].sum())}
                              for field in ('open','high','low','close','volume')},
       'nonlatest_close_source_symbols':sorted(changed.loc[(~changed.is_latest_chart_bar)&changed.close_changed,'symbol'].unique().tolist()),
       'only_old':int((joined._merge=='left_only').sum()),'only_new':int((joined._merge=='right_only').sum()),
       'note':'Changes in latest chart bar likely reflect incomplete intraperiod updates; older-bar changes require independent investigation. Comparisons do not prove publication-time parity.'}
    (out/'snapshot_comparison.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('older',type=Path);p.add_argument('newer',type=Path);p.add_argument('--out',type=Path,default=Path('artifacts/version_comparison'))
    args=p.parse_args();print(json.dumps(compare(args.older,args.newer,args.out),indent=2))
