#!/usr/bin/env python3
"""Audit/infer TradingView SP500 78-minute multi-stock batch exports.

The exporter does not put its Batch input into CSV output. This utility identifies
which batch each CSV contains by comparing one completed session close against
the frozen 503-stock daily reference panel. It then reports duplicate payloads,
missing batches, and slot-level OHLCV coverage.

It does NOT alter the raw exports. Use the resulting mapping as the canonical
staging manifest for downstream liquidity/node research.
"""
from pathlib import Path
import argparse, zipfile, pandas as pd, numpy as np, re, io, hashlib, json


def parse_universe(pine_path: Path):
    s=pine_path.read_text(encoding='utf-8')
    m=re.search(r'universe = str\.split\("([^"]+)"',s)
    if not m: raise ValueError('Universe array not found in Pine exporter')
    u=m.group(1).split(',')
    if len(u)!=503: raise ValueError(f'Expected 503 symbols, found {len(u)}')
    return u


def daily_closes(daily_zip: Path, refdate: str):
    out={}
    with zipfile.ZipFile(daily_zip) as z:
        for n in z.namelist():
            m=re.search(r'/stocks/BATS_(.*), 1D\.csv$',n)
            if not m: continue
            tk=m.group(1); data=z.read(n); lines=data.splitlines()
            header=lines[0].decode().split(','); ci=header.index('close'); val=np.nan
            for line in reversed(lines):
                if line.startswith((refdate+',').encode()):
                    try: val=float(line.decode().split(',')[ci])
                    except Exception: pass
                    break
            out[tk]=val
    return out


def run(args):
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    universe=parse_universe(Path(args.pine))
    close=daily_closes(Path(args.daily_reference_zip),args.reference_date)
    expected=[]
    for b in range(1,43):
        xs=universe[(b-1)*12:b*12]
        vals=np.array([close.get(t,np.nan) for t in xs]+[np.nan]*(12-len(xs)),float)[:12]
        expected.append((b,xs,vals))

    filemap=[]
    with zipfile.ZipFile(args.intraday_zip) as z:
        for n in z.namelist():
            raw=z.read(n)
            df=pd.read_csv(io.BytesIO(raw),usecols=['time']+[f'S{i:02d}_C' for i in range(1,13)])
            ds=df.time.astype(str).str[:10]; obs=[]
            for slot in range(1,13):
                x=pd.to_numeric(df.loc[ds==args.reference_date,f'S{slot:02d}_C'],errors='coerce').dropna()
                obs.append(x.iloc[-1] if len(x) else np.nan)
            O=np.array(obs,float); scores=[]
            for b,xs,E in expected:
                mask=np.isfinite(O)&np.isfinite(E)&(np.abs(E)>1e-12)
                score=999.0 if mask.sum()<6 else float(np.sqrt(np.mean(((O[mask]-E[mask])/E[mask])**2)))
                scores.append((score,b,int(mask.sum())))
            scores.sort()
            filemap.append({'file':n,'batch':scores[0][1],'match_rmse':scores[0][0],
                            'matched_slots':scores[0][2],'second_batch':scores[1][1],
                            'second_rmse':scores[1][0],
                            'sha256':hashlib.sha256(raw).hexdigest()})
    fm=pd.DataFrame(filemap)
    fm['is_duplicate_payload']=fm.duplicated('sha256',keep='first')
    fm.to_csv(out/'source_file_batch_inference.csv',index=False)
    chosen=fm[~fm.is_duplicate_payload].sort_values(['batch','match_rmse']).drop_duplicates('batch')
    missing=sorted(set(range(1,43))-set(chosen.batch.astype(int)))

    rows=[]
    with zipfile.ZipFile(args.intraday_zip) as z:
        for _,r in chosen.iterrows():
            b=int(r.batch); n=r.file; df=pd.read_csv(io.BytesIO(z.read(n))); times=df.time.astype(str)
            for slot,tk in enumerate(universe[(b-1)*12:b*12],1):
                base=f'S{slot:02d}_'; o,h,l,c,v=[base+x for x in 'OHLCV']
                num=df[[o,h,l,c,v]].apply(pd.to_numeric,errors='coerce')
                valid=num[[o,h,l,c]].notna().all(axis=1); x=num.loc[valid]
                if valid.any():
                    ii=np.flatnonzero(valid.to_numpy()); start=times.iloc[ii[0]]; end=times.iloc[ii[-1]]
                    invalid=int(((x[h]<x[[o,c,l]].max(axis=1))|(x[l]>x[[o,c,h]].min(axis=1))).sum())
                    vol=x[v]
                else:
                    start=end=''; invalid=0; vol=pd.Series(dtype=float)
                rows.append({'ticker':tk,'batch':b,'slot':slot,'source_file':n,
                             'valid_ohlc_rows':int(valid.sum()),'start':start,'end':end,
                             'volume_nonmissing_frac':float(vol.notna().mean()) if len(vol) else 0.0,
                             'volume_positive_frac':float((vol>0).mean()) if len(vol) else 0.0,
                             'invalid_ohlc_rows':invalid})
    audit=pd.DataFrame(rows); audit.to_csv(out/'stock_coverage_audit.csv',index=False)

    man=[]
    for i,tk in enumerate(universe):
        b=i//12+1; slot=i%12+1
        man.append({'rank':i+1,'ticker':tk,'batch':b,'slot':slot,'batch_available':b not in missing})
    pd.DataFrame(man).to_csv(out/'full_503_batch_manifest_and_coverage.csv',index=False)

    summary={'csv_exports_received':len(fm),'unique_payloads':int((~fm.is_duplicate_payload).sum()),
             'unique_batches_present':int(chosen.batch.nunique()),'missing_batches':missing,
             'expected_tickers_in_present_batches':len(audit),'tickers_with_data':int((audit.valid_ohlc_rows>0).sum()),
             'zero_data_tickers':audit.loc[audit.valid_ohlc_rows==0,'ticker'].tolist(),
             'median_valid_rows':float(audit.loc[audit.valid_ohlc_rows>0,'valid_ohlc_rows'].median())}
    (out/'PROCESSING_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--intraday-zip',required=True)
    ap.add_argument('--pine',required=True)
    ap.add_argument('--daily-reference-zip',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--reference-date',default='2026-08-07')
    run(ap.parse_args())
