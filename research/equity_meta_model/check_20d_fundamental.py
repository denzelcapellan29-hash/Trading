import pandas as pd, numpy as np
from pathlib import Path
OUT=Path('/mnt/data/fund_meta/results_v2')
STOCK=Path('/mnt/data/ml_meta_research_inputs/ohlcv/SP500_Current_503_OHLCV_Savepoint_2026-08-20/stocks')
d=pd.read_csv(OUT/'walkforward_trade_panel_with_fundamentals.csv.gz',parse_dates=['signal_date','entry_date'])
d['ret20_fwd']=np.nan
for n,(t,idxs) in enumerate(d.groupby('ticker').groups.items(),1):
    p=STOCK/f'BATS_{t}, 1D.csv'
    if not p.exists(): continue
    x=pd.read_csv(p,usecols=['time','open']); x['time']=pd.to_datetime(x.time); x=x.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    pos={dt:i for i,dt in enumerate(x.time)}
    for j in idxs:
        dt=d.at[j,'entry_date']; k=pos.get(dt)
        if k is not None and k+20 < len(x) and pd.notna(x.at[k,'open']) and pd.notna(x.at[k+20,'open']):
            d.at[j,'ret20_fwd']=x.at[k+20,'open']/x.at[k,'open']-1
    if n%100==0: print('ticker',n,flush=True)
rows=[]
for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07')]:
    z=d[(d.signal_date>=a)&(d.signal_date<=b)&d.wf_fund_bucket.notna()&d.ret20_fwd.notna()]
    for bb,g in z.groupby('wf_fund_bucket'):
        rows.append({'sample':sample,'bucket':int(bb),'n':len(g),'mean_5d':g.gross_trade_ret_5d.mean(),'mean_20d':g.ret20_fwd.mean(),'win20':(g.ret20_fwd>0).mean(),'mean_extension':g.fund_extension.mean()})
pd.DataFrame(rows).to_csv(OUT/'fundamental_support_5d_vs_20d.csv',index=False)
d['wf_extension_half']=np.nan
for yr in range(2015,2027):
    tr=d[(d.signal_date.dt.year<yr)&d.fund_extension.notna()]; te=d[(d.signal_date.dt.year==yr)&d.fund_extension.notna()]
    if len(tr)<100: continue
    q=tr.fund_extension.median(); d.loc[te.index,'wf_extension_half']=np.where(te.fund_extension<=q,'low_extension','high_extension')
rows=[]
for sample,a,b in [('VALID','2015-01-01','2019-12-31'),('HOLDOUT','2020-01-01','2026-08-07'),('OOS2015+','2015-01-01','2026-08-07')]:
    z=d[(d.signal_date>=a)&(d.signal_date<=b)&d.wf_extension_half.notna()&d.ret20_fwd.notna()]
    for bb,g in z.groupby('wf_extension_half'):
        rows.append({'sample':sample,'extension_half':bb,'n':len(g),'mean_5d':g.gross_trade_ret_5d.mean(),'mean_20d':g.ret20_fwd.mean(),'win20':(g.ret20_fwd>0).mean(),'mean_extension':g.fund_extension.mean()})
pd.DataFrame(rows).to_csv(OUT/'fundamental_extension_halves_5d_vs_20d.csv',index=False)
d[['signal_date','entry_date','ticker','ret20_fwd','fund_extension','fund_support','wf_fund_bucket']].to_csv(OUT/'trade_20d_fundamental_diagnostic.csv.gz',index=False,compression='gzip')
print(pd.read_csv(OUT/'fundamental_support_5d_vs_20d.csv').query("sample=='OOS2015+'").to_string(index=False))
print('\nHalves')
print(pd.read_csv(OUT/'fundamental_extension_halves_5d_vs_20d.csv').to_string(index=False))
