import pandas as pd, numpy as np, math
from pathlib import Path
OUT=Path('/mnt/data/fund_meta/results_v2')
tr=pd.read_csv(OUT/'walkforward_trade_panel_with_fundamentals.csv.gz',parse_dates=['signal_date'])
MULT={1:.75,2:.90,3:1.,4:1.10,5:1.25}

def build_port(d,bcol=None,cost_bp=10):
    rows=[]; prev={}
    for dt,g in d.sort_values(['signal_date','ticker']).groupby('signal_date'):
        if bcol is None: w=pd.Series(1/len(g),index=g.ticker.values)
        else:
            m=g.set_index('ticker')[bcol].astype(int).map(MULT).astype(float); w=m/m.sum()
        union=set(prev)|set(w.index); turn=sum(abs(float(w.get(t,0))-float(prev.get(t,0))) for t in union)
        rr=g.set_index('ticker').gross_trade_ret_5d; gross=float((w*rr).sum()); net=gross-turn*cost_bp/10000
        denom=1+gross; prev={t:float(w[t]*(1+rr[t])/denom) for t in w.index}
        rows.append((dt,gross,net,turn,len(g)))
    return pd.DataFrame(rows,columns=['date','gross','net','turnover','n']).set_index('date')

ports={}
for name,col in [('baseline',None),('base11_ev','base11_ev_bucket'),('fund12_ev','fund12_ev_bucket'),('base11_pwin','base11_pwin_bucket'),('fund12_pwin','fund12_pwin_bucket'),('base11_q50','base11_q50_bucket'),('fund12_q50','fund12_q50_bucket')]:
    z=tr if col is None else tr[tr[col].notna()].copy(); ports[name]=build_port(z,col)
wk=pd.concat({k:v.net for k,v in ports.items()},axis=1)
wk.to_csv(OUT/'weekly_return_comparison.csv')

def m(a):
    a=np.asarray(a,float); eq=np.cumprod(1+a); cagr=eq[-1]**(52/len(a))-1; vol=a.std(ddof=1)*np.sqrt(52); sh=a.mean()/a.std(ddof=1)*np.sqrt(52)
    down=np.sqrt(np.mean(np.minimum(a,0)**2))*np.sqrt(52); sort=a.mean()*52/down; dd=eq/np.maximum.accumulate(eq)-1; ulcer=np.sqrt(np.mean((dd*100)**2)); q=np.quantile(a,.05); cvar=a[a<=q].mean()
    return np.array([cagr,vol,sh,sort,dd.min(),ulcer,cvar])

def boot(x,y,B=5000,block=26,seed=20260825):
    x=np.asarray(x); y=np.asarray(y); n=len(x); rng=np.random.default_rng(seed); nb=math.ceil(n/block); A=np.empty((B,7))
    for i in range(B):
        starts=rng.integers(0,n,nb); ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]; A[i]=m(y[ix])-m(x[ix])
    names=['CAGR','vol','Sharpe','Sortino','maxDD','Ulcer','CVaR5']; rows=[]
    for j,k in enumerate(names):
        higher=k not in ['vol','Ulcer']; pim=(A[:,j]>0).mean() if higher else (A[:,j]<0).mean()
        rows.append({'metric':k,'median_delta':np.median(A[:,j]),'p_improve':pim,'lo95':np.quantile(A[:,j],.025),'hi95':np.quantile(A[:,j],.975)})
    return pd.DataFrame(rows)

z=wk.loc['2015-01-01':'2026-08-07'].dropna()
for score in ['ev','pwin','q50']:
    boot(z[f'base11_{score}'].values,z[f'fund12_{score}'].values).to_csv(OUT/f'bootstrap_fund12_{score}_vs_base11_{score}.csv',index=False)
print(wk.loc['2015-01-01':].shape)
print(pd.read_csv(OUT/'bootstrap_fund12_ev_vs_base11_ev.csv').to_string(index=False))
