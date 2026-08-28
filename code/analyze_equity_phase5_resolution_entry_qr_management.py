from __future__ import annotations
import argparse, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PERIODS=['TRAIN_2013_2016','VALID_2017_2021','HOLDOUT_2022PLUS']


def enrich_from_phase5(candidates: pd.DataFrame, paths: pd.DataFrame):
    t=candidates.copy(); p=paths.copy()
    den=t['target_dist_atr_from_confirmation']+t['qr']+0.5
    A=t['direction']*(t['target_price']-t['stop_price'])/den
    t['snapshot_atr_recovered']=A
    t['node_price']=t['stop_price']+t['direction']*0.5*A
    t['resolution_entry_price']=t['stop_price']+t['direction']*A
    t['confirmation_close']=t['stop_price']+t['direction']*(t['qr']+0.5)*A
    t['resolution_entry_gross_to_close']=t['direction']*(t['confirmation_close']-t['resolution_entry_price'])/t['resolution_entry_price']
    if not np.all(np.isfinite(A)&(A>0)): raise ValueError('Failed to recover positive snapshot ATR for all candidates')
    p['time']=pd.to_datetime(p.time,utc=True)
    lookup=t.set_index('trade_id')[['direction','entry_price','confirmation_close']]
    p=p.sort_values(['trade_id','time']).copy();p['rn']=p.groupby('trade_id').cumcount()
    p=p.merge(lookup,left_on='trade_id',right_index=True,how='left',validate='many_to_one')
    first=p.rn.eq(0)
    mark=p.loc[first,'entry_price']*(1+p.loc[first,'direction']*p.loc[first,'raw_return'])
    p.loc[first,'raw_return']=p.loc[first,'direction']*(mark-p.loc[first,'confirmation_close'])/p.loc[first,'confirmation_close']
    resrows=pd.DataFrame({'trade_id':t.trade_id,'time':pd.to_datetime(t.resolution_time,utc=True),'raw_return':t.resolution_entry_gross_to_close})
    retained=pd.concat([resrows,p[['trade_id','time','raw_return']]],ignore_index=True).sort_values(['trade_id','time'])
    return t,retained


def select(t,variant):
    x=t.copy()
    if variant=='RESOLUTION_ALL_RETAIN':x['retain']=True
    elif variant=='RESOLUTION_QR_MANAGED':x['retain']=x.qr_high.astype(bool)
    elif variant=='INTERACTED_Q2_QR_MANAGED':x=x[x.interacted_Q.eq(2)].copy();x['retain']=x.qr_high.astype(bool)
    elif variant=='TARGET_Q2_QR_MANAGED':x=x[x.target_Q.eq(2)].copy();x['retain']=x.qr_high.astype(bool)
    elif variant=='BOTH_Q2_QR_MANAGED':x=x[x.interacted_Q.eq(2)&x.target_Q.eq(2)].copy();x['retain']=x.qr_high.astype(bool)
    else:raise ValueError(variant)
    for c in ['snapshot_time','resolution_time','exit_time']:x[c]=pd.to_datetime(x[c],utc=True)
    x['entry_time']=x.resolution_time
    x=x.sort_values(['ticker','entry_time','snapshot_time']).drop_duplicates(['ticker','entry_time','direction'],keep='last')
    x.loc[~x.retain,'exit_time']=x.loc[~x.retain,'resolution_time']
    x.loc[~x.retain,'gross_return']=x.loc[~x.retain,'resolution_entry_gross_to_close']
    x.loc[~x.retain,'bars_held']=1
    x.loc[~x.retain,'exit_reason']='qr_exit_at_confirmation_close'
    rg=x.retain
    x.loc[rg,'gross_return']=x.loc[rg,'direction']*(x.loc[rg,'exit_price']-x.loc[rg,'resolution_entry_price'])/x.loc[rg,'resolution_entry_price']
    keep=[]
    for _,g in x.sort_values(['ticker','entry_time']).groupby('ticker',sort=False):
        last=pd.Timestamp.min.tz_localize('UTC')
        for i,r in g.iterrows():
            if r.entry_time>last:keep.append(i);last=r.exit_time
    return x.loc[keep].sort_values('entry_time').reset_index(drop=True)


def trade_stats(x,cost):
    if x.empty:return {}
    net=x.gross_return.to_numpy()-2*cost*1e-4;w=net[net>0];l=net[net<0]
    return {'n':len(x),'win_rate':(net>0).mean(),'mean_trade':net.mean(),'median_trade':np.median(net),
            'profit_factor':w.sum()/(-l.sum()) if len(l) else np.inf,'mean_bars_held':x.bars_held.mean(),
            'retain_share':x.retain.mean(),'target_share':x.exit_reason.str.startswith('target').mean()}


def paths_for(x,retained):
    rid=set(x.loc[x.retain,'trade_id']);eid=set(x.loc[~x.retain,'trade_id'])
    q=retained[retained.trade_id.isin(rid)].copy()
    e=x[x.trade_id.isin(eid)][['trade_id','resolution_time','resolution_entry_gross_to_close']].rename(columns={'resolution_time':'time','resolution_entry_gross_to_close':'raw_return'})
    return pd.concat([q,e[['trade_id','time','raw_return']]],ignore_index=True)


def book_metrics(x,retained,cost,calendar):
    if x.empty:return {}
    q=paths_for(x,retained);q['time']=pd.to_datetime(q.time,utc=True)
    q=q.merge(x[['trade_id','entry_time','exit_time']],on='trade_id',how='left',validate='many_to_one')
    cb=cost*1e-4;q['cost']=0.0;q.loc[q.time.eq(q.entry_time),'cost']+=cb;q.loc[q.time.eq(q.exit_time),'cost']+=cb;q['net_ret']=q.raw_return-q.cost
    br=q.groupby('time').agg(ret=('net_ret','mean'),active=('trade_id','nunique')).sort_index()
    idx=calendar[(calendar>=br.index.min())&(calendar<=br.index.max())];br=br.reindex(idx,fill_value=0.0)
    eq=(1+br.ret).cumprod();years=(br.index[-1]-br.index[0]).total_seconds()/(365.25*86400);cagr=eq.iloc[-1]**(1/years)-1
    ann=1260;sd=br.ret.std(ddof=1);dn=br.ret[br.ret<0];dd=eq/eq.cummax()-1;mdd=dd.min()
    return {'book_cagr':cagr,'book_vol':sd*np.sqrt(ann),'book_sharpe':br.ret.mean()/sd*np.sqrt(ann) if sd>0 else np.nan,
            'book_sortino':br.ret.mean()/dn.std(ddof=1)*np.sqrt(ann) if len(dn)>1 and dn.std(ddof=1)>0 else np.nan,
            'book_max_dd':mdd,'book_calmar':cagr/abs(mdd) if mdd<0 else np.nan,'end_multiple':eq.iloc[-1],
            'active_bar_share':(br.active>0).mean(),'mean_active_when_active':br.loc[br.active>0,'active'].mean(),'max_active':br.active.max()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidates',required=True);ap.add_argument('--paths',required=True);ap.add_argument('--raw-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    t,p=enrich_from_phase5(pd.read_csv(a.candidates),pd.read_csv(a.paths));t.to_csv(out/'resolution_entry_enriched_candidates.csv.gz',index=False,compression='gzip');p.to_csv(out/'resolution_entry_retained_paths.csv.gz',index=False,compression='gzip')
    ref=next(Path(a.raw_dir).glob('*.csv'));calendar=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ref,usecols=['time']).time,utc=True).dropna().drop_duplicates().sort_values())
    variants=['RESOLUTION_ALL_RETAIN','RESOLUTION_QR_MANAGED','INTERACTED_Q2_QR_MANAGED','TARGET_Q2_QR_MANAGED','BOTH_Q2_QR_MANAGED'];costs=[0,1,2,5,10];rows=[];subs=[]
    for v in variants:
        x=select(t,v);x.to_csv(out/f'trades_{v.lower()}.csv',index=False)
        for c in costs:
            s=trade_stats(x,c);s.update(book_metrics(x,p,c,calendar));s.update({'variant':v,'cost_bps_one_way':c,'sample':'ALL'});rows.append(s)
            for per in PERIODS:
                g=x[x.period.eq(per)];z=trade_stats(g,c);z.update({'variant':v,'cost_bps_one_way':c,'sample':per});subs.append(z)
    pd.DataFrame(rows).to_csv(out/'resolution_entry_portfolio_metrics.csv',index=False);pd.DataFrame(subs).to_csv(out/'resolution_entry_trade_metrics_by_period.csv',index=False)
    methodology={'concept':'Resolution-trigger entry with QR as causal post-entry management','entry':'first clean node +/-0.5 ATR hold/accept threshold','high_qr':'retain to exact corrected mapped target or structural invalidation','nonhigh_qr':'exit at resolution-bar close','time_exit':None,'cost_grid_one_way_bps':costs,'universe':'384-stock persisted-raw Phase5 pilot'}
    (out/'RESOLUTION_ENTRY_METHODOLOGY.json').write_text(json.dumps(methodology,indent=2))

if __name__=='__main__':main()
