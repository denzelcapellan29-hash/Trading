#!/usr/bin/env python3
from __future__ import annotations
import argparse, zipfile, json
from pathlib import Path
import numpy as np
import pandas as pd

CORRIDOR_WEIGHTS=[0.0,0.10,0.15,0.20]
FX_WEIGHTS=[0.25,1/3,0.40,0.50,0.60]

def perf(r: pd.Series)->dict:
    r=pd.Series(r).dropna().astype(float)
    eq=(1+r).cumprod(); years=(r.index[-1]-r.index[0]).days/365.25
    dd=eq/eq.cummax()-1; vol=r.std(ddof=1)*np.sqrt(52)
    down=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52)
    q=r.quantile(.05)
    return {
        'weeks':len(r),'start':str(r.index.min().date()),'end':str(r.index.max().date()),
        'CAGR':eq.iloc[-1]**(1/years)-1,'ann_arithmetic_return':r.mean()*52,
        'vol':vol,'Sharpe':r.mean()/r.std(ddof=1)*np.sqrt(52),
        'Sortino_downside_RMS':r.mean()*52/down if down>0 else np.nan,
        'maxDD':dd.min(),'Ulcer':np.sqrt(np.mean((dd*100)**2)),
        'CVaR5_weekly':r[r<=q].mean(),'end_multiple':eq.iloc[-1],
    }

def load_phase6(path: Path)->pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        n=next(n for n in z.namelist() if n.endswith('13_weekly_frozen_plus_corridor_accept.csv'))
        x=pd.read_csv(z.open(n),parse_dates=['period_end']).set_index('period_end').sort_index()
    x.index=x.index+pd.Timedelta(days=7); x.index.name='period_end'
    return x

def load_old_equity(path: Path)->tuple[pd.Series,pd.Series]:
    with zipfile.ZipFile(path) as z:
        n=next(n for n in z.namelist() if n.endswith('data/source_weekly_equity_fx.csv'))
        x=pd.read_csv(z.open(n),parse_dates=['date']).set_index('date').sort_index()
    signal=x['Barbell50_Agreement25_PCA25'].rename('preferred_equity_signal_labeled')
    realized=signal.copy(); realized.index=realized.index+pd.Timedelta(days=7); realized.name='preferred_equity'
    return realized, x['FX_65FAST_35ALT'].rename('old_joined_ALT')

def load_fast(path: Path)->pd.Series:
    x=pd.read_csv(path,parse_dates=['period_end']).set_index('period_end').sort_index()
    return x['net_portfolio_return'].rename('FAST')

def load_alt(path: Path)->pd.Series:
    x=pd.read_csv(path); x['date']=pd.to_datetime(x['date'],utc=True).dt.tz_convert(None)
    return x.set_index('date').sort_index()['FX_65FAST_35ALT'].rename('FX_65FAST_35ALT')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase6-data',required=True)
    ap.add_argument('--old-leverage-zip',required=True)
    ap.add_argument('--fast-weekly',required=True)
    ap.add_argument('--alt-weekly',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    phase6=load_phase6(Path(args.phase6_data)); old_eq, old_alt_joined=load_old_equity(Path(args.old_leverage_zip))
    fast=load_fast(Path(args.fast_weekly)); alt=load_alt(Path(args.alt_weekly))

    p11=pd.concat([old_eq,fast,alt],axis=1,join='inner').dropna()
    p11.to_csv(out/'01_2011plus_aligned_preferred_equity_fx.csv')
    rows=[]
    for fx in ['FAST','FX_65FAST_35ALT']:
        rows.append({'sample':'2011plus','equity':'preferred_no_put','fx':fx,'fx_weight':0.0,**perf(p11['preferred_equity'])})
        for w in FX_WEIGHTS:
            r=(1-w)*p11['preferred_equity']+w*p11[fx]
            rows.append({'sample':'2011plus','equity':'preferred_no_put','fx':fx,'fx_weight':w,**perf(r)})
    pd.DataFrame(rows).to_csv(out/'02_2011plus_no_put_grid.csv',index=False)

    p13=phase6.join(fast,how='inner').join(alt,how='inner')
    for c in CORRIDOR_WEIGHTS:
        p13[f'EQ_C{int(c*100):02d}']=(1-c)*p13['bar50_ag25_pca25']+c*p13['corridor_accept']
    p13.to_csv(out/'03_2013plus_aligned_corridor_fx.csv')
    rows=[]
    for c in CORRIDOR_WEIGHTS:
        e=p13[f'EQ_C{int(c*100):02d}']
        rows.append({'sample':'2013plus','corridor_weight':c,'fx':'NONE','fx_weight':0.0,**perf(e)})
        for fx in ['FAST','FX_65FAST_35ALT']:
            for w in FX_WEIGHTS:
                rows.append({'sample':'2013plus','corridor_weight':c,'fx':fx,'fx_weight':w,**perf((1-w)*e+w*p13[fx])})
    pd.DataFrame(rows).to_csv(out/'04_2013plus_no_put_corridor_grid.csv',index=False)

    with zipfile.ZipFile(Path(args.old_leverage_zip)) as z:
        n=next(n for n in z.namelist() if n.endswith('data/source_weekly_equity_fx.csv'))
        old=pd.read_csv(z.open(n),parse_dates=['date']).set_index('date').sort_index()
    old_misaligned=.5*old['Barbell50_Agreement25_PCA25']+.5*old['FX_65FAST_35ALT']
    corrected_2011_alt=.5*p11['preferred_equity']+.5*p11['FX_65FAST_35ALT']
    corrected_2011_fast=.5*p11['preferred_equity']+.5*p11['FAST']
    e20=p13['EQ_C20']
    corrected_2013_fast=.5*e20+.5*p13['FAST']
    corrected_2013_alt=.5*e20+.5*p13['FX_65FAST_35ALT']
    audit=[]
    for name,r in [
        ('OLD_2011_50EQ_50ALT_MISALIGNED',old_misaligned),
        ('CORRECTED_2011_50EQ_50ALT',corrected_2011_alt),
        ('CORRECTED_2011_50EQ_50FAST',corrected_2011_fast),
        ('CORRECTED_2013_C20_50EQ_50FAST',corrected_2013_fast),
        ('CORRECTED_2013_C20_50EQ_50ALT',corrected_2013_alt),
    ]:
        audit.append({'construction':name,**perf(r)})
    pd.DataFrame(audit).to_csv(out/'05_alignment_correction_key_comparison.csv',index=False)

    rows=[]
    for label,a,b in [('2013_2016','2013-01-01','2016-12-31'),('2017_2021','2017-01-01','2021-12-31'),('2022_PLUS','2022-01-01','2026-12-31'),('FULL','2013-01-01','2026-12-31')]:
        z=p13.loc[a:b]; e=z['EQ_C20']
        for fx in ['FAST','FX_65FAST_35ALT']:
            rows.append({'period':label,'fx':fx,'weeks':len(z),'corr_equity_fx':e.corr(z[fx]),'corr_corridor_fx':z['corridor_accept'].corr(z[fx])})
    pd.DataFrame(rows).to_csv(out/'06_corrected_correlation_chronology.csv',index=False)

    full11=pd.date_range(p11.index.min(),p11.index.max(),freq='W-FRI')
    meta={
        'alignment_rule':'Equity signal-week labels shifted forward 7 calendar days to actual realization-week Friday. FAST/ALT actual-week Friday labels are not shifted.',
        '2011plus_sample':[str(p11.index.min().date()),str(p11.index.max().date()),len(p11)],
        '2011plus_missing_observed_fridays':len(full11.difference(p11.index)),
        '2013plus_corridor_sample':[str(p13.index.min().date()),str(p13.index.max().date()),len(p13)],
        'put_overlay':'excluded entirely from this study',
        'optimization':'none; fixed weights only',
    }
    (out/'00_methodology.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__': main()
