"""
Momentum Barbell trade-panel reconstruction used for MetaModel v1.

This script is the exact research reconstruction used in the first pass.
It expects the project OHLCV savepoint and SPX ZIP paths shown below.
The resulting 10bp 2015+ portfolio was checked against the frozen Barbell
fingerprint before the trade-level ML analysis.

For a new machine, edit ROOT, SPX_ZIP, and OUT at the top of the file.
"""
from pathlib import Path
import pandas as pd, numpy as np, re, json, math, warnings
warnings.filterwarnings('ignore')
ROOT=Path('/mnt/data/ml_meta_research_inputs/ohlcv/SP500_Current_503_OHLCV_Savepoint_2026-08-20/stocks')
SPX_ZIP=Path('/mnt/data/c5b53fa5-63cc-4d3a-86f0-e5c6e1c52b79.zip')
OUT=Path('/mnt/data/ml_meta_research_inputs/barbell_recon')
OUT.mkdir(exist_ok=True)
# load stocks 2008+ to provide lookback
series={}
opens={}; highs={}; lows={}; vols={}; closes={}
for f in ROOT.glob('*.csv'):
    m=re.match(r'BATS_(.*), 1D\.csv', f.name)
    if not m: continue
    t=m.group(1)
    df=pd.read_csv(f, usecols=lambda c:c in ['time','open','high','low','close','Volume'])
    df['time']=pd.to_datetime(df['time'])
    df=df[(df.time>='2008-01-01')&(df.time<='2026-08-07')].drop_duplicates('time').set_index('time').sort_index()
    if len(df)<100: continue
    opens[t]=df['open']; highs[t]=df['high']; lows[t]=df['low']; closes[t]=df['close']; vols[t]=df['Volume']
print('loaded',len(closes))
O=pd.DataFrame(opens).sort_index(); H=pd.DataFrame(highs).sort_index(); L=pd.DataFrame(lows).sort_index(); C=pd.DataFrame(closes).sort_index(); V=pd.DataFrame(vols).sort_index()
# SPX
import zipfile, io
with zipfile.ZipFile(SPX_ZIP) as z:
    spx=pd.read_csv(z.open('TVC_SPX, 1D.csv'))
spx['time']=pd.to_datetime(spx['time']); spx=spx.set_index('time').sort_index()
spx=spx[(spx.index>='2008-01-01')&(spx.index<='2026-08-07')]
# Align to stock trading dates using SPX close changes on exact dates (ignore old monthly junk)
# Use dates after 2008 where there are many stock obs
R=C.pct_change(fill_method=None)
# availability and company exclusion
avail=R.notna()
sum_r=R.sum(axis=1, skipna=True); cnt=avail.sum(axis=1)
LOO=pd.DataFrame(index=R.index, columns=R.columns, dtype='float32')
# general leave-one-ticker first
for t in R.columns:
    num=sum_r-R[t].fillna(0)
    den=cnt-avail[t].astype(int)
    LOO[t]=(num/den.replace(0,np.nan)).astype('float32')
# alphabet leave-one-company
for t in ['GOOG','GOOGL']:
    if t in R.columns:
        other='GOOGL' if t=='GOOG' else 'GOOG'
        num=sum_r-R[t].fillna(0)-(R[other].fillna(0) if other in R else 0)
        den=cnt-avail[t].astype(int)-(avail[other].astype(int) if other in R else 0)
        LOO[t]=(num/den.replace(0,np.nan)).astype('float32')
# rolling alpha beta previous 252 observations, shifted 1
# pairwise rolling moments
minp=126
mean_x=LOO.rolling(252,min_periods=minp).mean().shift(1)
mean_y=R.rolling(252,min_periods=minp).mean().shift(1)
mean_xy=(LOO*R).rolling(252,min_periods=minp).mean().shift(1)
mean_x2=(LOO*LOO).rolling(252,min_periods=minp).mean().shift(1)
cov=mean_xy-mean_x*mean_y
var=mean_x2-mean_x*mean_x
beta=cov/var.replace(0,np.nan)
alpha=mean_y-beta*mean_x
resid=R-alpha-beta*LOO
# SPX returns on dates
spxret=spx['close'].pct_change(fill_method=None).reindex(R.index)
# fallback if exact missing no fill
neg=spxret<0
# D63 denominator residual std; numerator mean on spx down days only
resid_std=resid.rolling(63,min_periods=40).std(ddof=1)
num=resid.where(neg, np.nan).rolling(63,min_periods=10).mean()
D63=num/resid_std
# relative volume, current vs prior 20d average volume
vma=V.rolling(20,min_periods=15).mean().shift(1)
rvol=V/vma
# defiance day: SPX down + positive residual; avg rvol over those days
mask_def=pd.DataFrame(np.broadcast_to(neg.values[:,None],resid.shape), index=resid.index, columns=resid.columns) & (resid>0)
def_rvol=rvol.where(mask_def).rolling(63,min_periods=3).mean()
# momentum 12-1 close return: from t-252 to t-21
TS=C.shift(21)/C.shift(252)-1
CS=TS.rank(axis=1,pct=True,method='average')
# ancillary features causal
ret20=C/C.shift(20)-1; ret63=C/C.shift(63)-1
vol20=R.rolling(20,min_periods=15).std()*np.sqrt(252); vol63=R.rolling(63,min_periods=40).std()*np.sqrt(252)
# ATR14 normalized
prev=C.shift(1); tr=np.maximum(H-L,np.maximum((H-prev).abs(),(L-prev).abs()))
atr14=tr.rolling(14,min_periods=10).mean(); atrpct=atr14/C
# drawdown from 63d high
DD63=C/C.rolling(63,min_periods=40).max()-1
# rvol current and average20
rvol_now=rvol
# Market features
spxc=spx['close'].reindex(R.index)
spx20=spxc/spxc.shift(20)-1; spx63=spxc/spxc.shift(63)-1
spxvol20=spxret.rolling(20,min_periods=15).std()*np.sqrt(252); spxdd63=spxc/spxc.rolling(63,min_periods=40).max()-1
# weekly signal last trading date each W-FRI from 2010
signal_dates=pd.Series(R.index,index=R.index).resample('W-FRI').last().dropna()
rows=[]; weekly=[]
idx=C.index
for k,(wfr,sig) in enumerate(signal_dates.items()):
    if sig<pd.Timestamp('2010-01-01') or sig>pd.Timestamp('2026-08-07'): continue
    if sig not in D63.index: continue
    # eligible sufficient D and volume
    d=D63.loc[sig]; drv=def_rvol.loc[sig]; ts=TS.loc[sig]; cs=CS.loc[sig]
    valid=d.notna()
    if valid.sum()<50: continue
    d_pct=d.rank(pct=True)
    core=(d_pct>=0.80)&(drv>=1.20)
    bar=core & ((ts<0)|(cs>=2/3))
    cand=d[bar].dropna().sort_values(ascending=False).head(5)
    # find next open and open after 5 sessions for each ticker individually using global date positions
    pos=idx.get_indexer([sig])[0]
    if pos<0 or pos+6>=len(idx): continue
    entry_date=idx[pos+1]; exit_date=idx[pos+6]
    selected=[]
    for t in cand.index:
        if pd.isna(O.at[entry_date,t]) or pd.isna(O.at[exit_date,t]): continue
        gross=O.at[exit_date,t]/O.at[entry_date,t]-1
        loo5=(C.at[exit_date,t]/C.at[entry_date,t]-1) # placeholder
        row={
            'signal_date':sig,'week_friday':wfr,'entry_date':entry_date,'exit_date':exit_date,'ticker':t,
            'gross_trade_ret_5d':gross,'D63':d[t],'D63_pct':d_pct[t],'def_rvol63':drv[t],
            'tsmom_12_1':ts[t],'csmom_pct':cs[t],
            'state_turnaround':int(ts[t]<0),'state_leader':int((ts[t]>=0) and (cs[t]>=2/3)),
            'ret20':ret20.at[sig,t],'ret63':ret63.at[sig,t],'vol20':vol20.at[sig,t],'vol63':vol63.at[sig,t],
            'atrpct14':atrpct.at[sig,t],'dd63':DD63.at[sig,t],'rvol_now':rvol_now.at[sig,t],
            'spx_ret20':spx20.at[sig],'spx_ret63':spx63.at[sig],'spx_vol20':spxvol20.at[sig],'spx_dd63':spxdd63.at[sig],
        }
        rows.append(row); selected.append((t,gross))
    if selected:
        gross_week=np.mean([r for _,r in selected])
        # simple roundtrip cost 20bp every week; useful parity approximation
        net_week=gross_week-0.002
        weekly.append({'signal_date':sig,'entry_date':entry_date,'exit_date':exit_date,'n':len(selected),'gross':gross_week,'net_simple20bp':net_week,'tickers':'|'.join(t for t,_ in selected)})
trades=pd.DataFrame(rows); weeks=pd.DataFrame(weekly)
trades.to_csv(OUT/'barbell_trade_panel.csv.gz',index=False,compression='gzip'); weeks.to_csv(OUT/'barbell_weekly_recon.csv',index=False)
# Metrics weekly based on simple returns

def met(r):
    r=pd.Series(r).dropna(); eq=(1+r).cumprod(); yrs=len(r)/52
    cagr=eq.iloc[-1]**(1/yrs)-1; vol=r.std()*np.sqrt(52); sh=r.mean()/r.std()*np.sqrt(52)
    down=np.sqrt(np.mean(np.minimum(r,0)**2))*np.sqrt(52); sort=r.mean()*52/down
    dd=eq/eq.cummax()-1
    return dict(n=len(r),cagr=cagr,vol=vol,sharpe=sh,sortino=sort,maxdd=dd.min())
for a,b in [('2010-01-01','2014-12-31'),('2015-01-01','2019-12-31'),('2020-01-01','2026-08-07'),('2015-01-01','2026-08-07'),('2010-01-01','2026-08-07')]:
    z=weeks[(pd.to_datetime(weeks.signal_date)>=a)&(pd.to_datetime(weeks.signal_date)<=b)]
    print(a,b,'gross',met(z.gross),'net',met(z.net_simple20bp),'avg n',z.n.mean())
print('trades',len(trades),'weeks',len(weeks))
print(trades[['D63','def_rvol63','tsmom_12_1','csmom_pct','gross_trade_ret_5d']].describe())
