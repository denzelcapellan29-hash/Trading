from __future__ import annotations
import argparse, json, zipfile, math, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_equity_fx_framework_phase5_standalone as p5

STATES=[('usable_side_mixed','hold'),('corridor','hold'),('corridor','accept'),('fragile_gateway_proxy','accept')]
PERIODS=p5.PERIODS


def process_stock_threshold(args):
    ticker,slot,frame,events=args
    cols=[f'S{slot:02d}_{x}' for x in ['O','H','L','C','V']]
    vals=frame[cols].apply(pd.to_numeric,errors='coerce'); mask=vals.iloc[:,:4].notna().all(axis=1).to_numpy()
    vals=vals.loc[mask].reset_index(drop=True)
    ts=pd.to_datetime(frame.loc[mask,'time'],errors='coerce',utc=True).dt.tz_convert('America/New_York').reset_index(drop=True)
    O,H,L,C,V=[vals.iloc[:,i].to_numpy(float) for i in range(5)]; n=len(C)
    if n<200 or events.empty:return [],[]
    prev=np.r_[np.nan,C[:-1]]; tr=np.maximum(H-L,np.maximum(np.abs(H-prev),np.abs(L-prev)))
    atr=pd.Series(tr).rolling(100,min_periods=50).mean().shift(1).to_numpy()
    p2=p5.make_pivots(H,L,C,2); p5m=p5.make_pivots(H,L,C,5); p10=p5.make_pivots(H,L,C,10)
    idxmap={int(x.value):i for i,x in enumerate(ts)}; node_cache={}; trades=[]; pathrows=[]
    for r in events.itertuples(index=False):
        snap=pd.Timestamp(r.snapshot_time); touch=pd.Timestamp(r.touch_time)
        si=idxmap.get(int(snap.value)); ti=idxmap.get(int(touch.value))
        if si is None or ti is None or ti>=n-1:continue
        a=atr[si]
        if not np.isfinite(a) or a<=0:continue
        if si not in node_cache:node_cache[si]=p5.build_nodes(p2,p5m,p10,si,C[si],a)
        nodes=node_cache[si]
        if not nodes:continue
        side_sign=1 if r.interacted_side=='upper' else -1
        expected=C[si]+side_sign*float(r.dist_atr)*a
        ni=int(np.argmin([abs(x['level']-expected) for x in nodes])); node=nodes[ni]
        if abs(node['level']-expected)>p5.CLUSTER_BAND_ATR*a+1e-9:continue
        lv=node['level']; upper=lv>C[si]
        hi=H[ti:min(n,ti+6)]; lo=L[ti:min(n,ti+6)]
        if upper:
            hh=np.flatnonzero(lo<=lv-p5.RESOLUTION_ATR*a); aa=np.flatnonzero(hi>=lv+p5.RESOLUTION_ATR*a)
        else:
            hh=np.flatnonzero(hi>=lv+p5.RESOLUTION_ATR*a); aa=np.flatnonzero(lo<=lv-p5.RESOLUTION_ATR*a)
        hd=int(hh[0]) if len(hh) else 99; ad=int(aa[0]) if len(aa) else 99
        if hd==ad or (hd==99 and ad==99):continue
        resolution='hold' if hd<ad else 'accept'
        if resolution!=r.resolution or (str(r.trust_family),resolution) not in STATES:continue
        ridx=ti+min(hd,ad)
        if ridx<=ti:continue
        direction=(-1 if upper else 1) if resolution=='hold' else (1 if upper else -1)
        stop=float(lv-direction*p5.RESOLUTION_ATR*a)
        threshold=float(lv+direction*p5.RESOLUTION_ATR*a)
        cand=[]
        for k,n2 in enumerate(nodes):
            if k==ni:continue
            lv2=float(n2['level']); d=direction*(lv2-threshold)
            if d>0:cand.append((d,k))
        if not cand:continue
        delta,ki=min(cand); target_node=nodes[ki]; target=float(target_node['level']); td=float(delta/a)
        op=float(O[ridx])
        if direction*(op-stop)<=0:continue
        entry=float(op if direction*(op-threshold)>0 else threshold)
        if direction*(target-entry)<=0 or direction*(entry-stop)<=0:continue
        exit_idx,exit_px,exit_reason,amb=p5.first_barrier_exit(O,H,L,ridx,direction,target,stop)
        if exit_idx is None:continue
        gross=float(direction*(exit_px-entry)/entry); risk=float(direction*(entry-stop)/entry); reward=float(direction*(target-entry)/entry)
        qr=float(direction*(C[ridx]-lv)/a)
        tid=f'{ticker}|STRICT|{ridx}|{direction}|{target:.8g}|{si}'
        t={'trade_id':tid,'ticker':ticker,'snapshot_time':r.snapshot_time,'touch_time':r.touch_time,'resolution_time':ts.iloc[ridx].isoformat(),
           'entry_time':ts.iloc[ridx].isoformat(),'exit_time':ts.iloc[exit_idx].isoformat(),'period':r.period,'trust_family':r.trust_family,'resolution':resolution,
           'direction':direction,'interacted_Q':int(r.Q_simple),'target_Q':int(target_node['Q']),'qr':qr,'target_dist_atr_from_trigger':td,
           'entry_price':entry,'threshold_price':threshold,'entry_bar_open':op,'gap_through_trigger':bool(direction*(op-threshold)>0),
           'target_price':target,'stop_price':stop,'exit_price':float(exit_px),'gross_return':gross,'risk_frac':risk,'reward_frac':reward,
           'reward_risk':reward/risk if risk>0 else np.nan,'bars_held':int(exit_idx-ridx+1),'exit_reason':exit_reason,'ambiguous_bar':amb}
        trades.append(t)
        for tm,rr in p5.trade_path(ts,O,C,ridx,exit_idx,entry,float(exit_px),direction):pathrows.append((tid,tm,rr))
    return trades,pathrows


def process_batch(args):
    b,raw_file,bmap,events=args
    use=['time']
    for rr in bmap.itertuples(index=False):use += [f'S{int(rr.slot):02d}_{x}' for x in ['O','H','L','C','V']]
    frame=pd.read_csv(raw_file,usecols=use); tr=[]; pa=[]
    for rr in bmap.itertuples(index=False):
        ev=events[events.ticker.astype(str)==str(rr.ticker)]
        if ev.empty:continue
        a,bp=process_stock_threshold((str(rr.ticker),int(rr.slot),frame,ev));tr.extend(a);pa.extend(bp)
    return b,tr,pa


def nonoverlap(x):
    if x.empty:return x
    y=x.copy()
    for c in ['snapshot_time','entry_time','exit_time','resolution_time']:y[c]=pd.to_datetime(y[c],utc=True)
    y=y.sort_values(['ticker','entry_time','snapshot_time']).drop_duplicates(['ticker','entry_time','direction'],keep='last')
    keep=[]
    for _,g in y.sort_values(['ticker','entry_time']).groupby('ticker',sort=False):
        last=pd.Timestamp.min.tz_localize('UTC')
        for i,r in g.iterrows():
            if r.entry_time>last:keep.append(i);last=r.exit_time
    return y.loc[keep].sort_values('entry_time').reset_index(drop=True)


def stats(x,cost):
    if x.empty:return {}
    net=x.gross_return.to_numpy()-2*cost*1e-4;w=net[net>0];l=net[net<0]
    return {'n':len(x),'win_rate':float((net>0).mean()),'mean_trade':float(net.mean()),'median_trade':float(np.median(net)),
            'profit_factor':float(w.sum()/(-l.sum())) if len(l) else np.inf,'mean_bars_held':float(x.bars_held.mean()),
            'mean_rr_at_entry':float(x.reward_risk.mean()),'gap_trigger_share':float(x.gap_through_trigger.mean()),'ambiguous_share':float(x.ambiguous_bar.mean())}


def book(x,paths,cost,calendar):
    if x.empty:return {}
    p=paths[paths.trade_id.isin(set(x.trade_id))].copy();p['time']=pd.to_datetime(p.time,utc=True)
    p=p.merge(x[['trade_id','entry_time','exit_time']],on='trade_id',how='left',validate='many_to_one');cb=cost*1e-4;p['cost']=0.0
    p.loc[p.time.eq(p.entry_time),'cost']+=cb;p.loc[p.time.eq(p.exit_time),'cost']+=cb;p['net_ret']=p.raw_return-p.cost
    br=p.groupby('time').agg(ret=('net_ret','mean'),active=('trade_id','nunique')).sort_index();idx=calendar[(calendar>=br.index.min())&(calendar<=br.index.max())];br=br.reindex(idx,fill_value=0.0)
    eq=(1+br.ret).cumprod();yrs=(br.index[-1]-br.index[0]).total_seconds()/(365.25*86400);cagr=eq.iloc[-1]**(1/yrs)-1;ann=1260;sd=br.ret.std(ddof=1);dn=br.ret[br.ret<0];dd=eq/eq.cummax()-1;mdd=dd.min()
    return {'book_cagr':float(cagr),'book_vol':float(sd*np.sqrt(ann)),'book_sharpe':float(br.ret.mean()/sd*np.sqrt(ann)) if sd>0 else np.nan,
            'book_sortino':float(br.ret.mean()/dn.std(ddof=1)*np.sqrt(ann)) if len(dn)>1 and dn.std(ddof=1)>0 else np.nan,'book_max_dd':float(mdd),
            'book_calmar':float(cagr/abs(mdd)) if mdd<0 else np.nan,'mean_active_when_active':float(br.loc[br.active>0,'active'].mean()),'max_active':int(br.active.max())}


def aggregate(out,raw):
    tf=sorted(out.glob('batch_*_trades.csv.gz'));pf=sorted(out.glob('batch_*_paths.csv.gz'))
    if len(tf)!=42 or len(pf)!=42:raise RuntimeError(f'checkpoints trades={len(tf)} paths={len(pf)}')
    t=pd.concat([pd.read_csv(f) for f in tf],ignore_index=True);paths=pd.concat([pd.read_csv(f) for f in pf],ignore_index=True)
    x=nonoverlap(t);x.to_csv(out/'causal_target_trades_all_states.csv.gz',index=False,compression='gzip');paths.to_csv(out/'causal_target_paths_all_states.csv.gz',index=False,compression='gzip')
    ref=next(raw.glob('*.csv'));cal=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ref,usecols=['time']).time,utc=True).dropna().drop_duplicates().sort_values())
    rows=[];subs=[]
    groups=[('ALL',None,None)]+[(f'{a}|{b}',a,b) for a,b in STATES]
    for name,fam,res in groups:
        g=x if fam is None else x[x.trust_family.eq(fam)&x.resolution.eq(res)]
        for c in [0,1,2,5,10]:
            s=stats(g,c);s.update(book(g,paths,c,cal));s.update({'variant':name,'one_way_cost_bps':c,'sample':'ALL'});rows.append(s)
            for per in PERIODS:
                z=stats(g[g.period.eq(per)],c);z.update({'variant':name,'one_way_cost_bps':c,'sample':per});subs.append(z)
    pd.DataFrame(rows).to_csv(out/'causal_target_portfolio_metrics.csv',index=False);pd.DataFrame(subs).to_csv(out/'causal_target_trade_metrics_by_period.csv',index=False)
    meth={'universe':503,'batches':42,'entry':'fully causal later-bar 0.5 ATR OCO resolution trigger; touch bar must precede resolution bar','gap_rule':'gap through trigger fills at resolution-bar open; gap beyond invalidation skips','same_bar_touch_resolution':'excluded','same_bar_stop_target':'stop-first conservative','target':'nearest frozen snapshot-map node ahead of pre-placed resolution trigger; no outcome-informed distance filter','states':STATES,'costs_one_way_bps':[0,1,2,5,10],'pnl_used_to_choose_parameters':False}
    (out/'CAUSAL_TARGET_METHODOLOGY.json').write_text(json.dumps(meth,indent=2))
    m=pd.DataFrame(rows);print(m[m.one_way_cost_bps.isin([0,1,2])][['variant','one_way_cost_bps','n','win_rate','mean_trade','profit_factor','gap_trigger_share','book_cagr','book_sharpe','book_sortino','book_max_dd']].to_string(index=False))
    s=pd.DataFrame(subs);print('\nPERIOD usable-side HOLD')
    print(s[s.variant.eq('usable_side_mixed|hold') & s.one_way_cost_bps.isin([0,1,2])][['one_way_cost_bps','sample','n','win_rate','mean_trade','profit_factor']].to_string(index=False))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--inference',required=True);ap.add_argument('--phase3-data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=min(16,cpu_count()));a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);raw=Path(a.raw_dir);mf=pd.read_csv(a.manifest);inf=pd.read_csv(a.inference)
    done={int(f.name.split('_')[1]) for f in out.glob('batch_*_trades.csv.gz') if (out/f.name.replace('_trades','_paths')).exists()};missing=[b for b in range(1,43) if b not in done]
    print('checkpointed',len(done),'missing',missing,flush=True)
    fmap={int(r.batch):raw/str(r.file) for r in inf.itertuples(index=False) if not bool(r.is_duplicate_payload) and (raw/str(r.file)).exists()}
    jobs=[]
    with zipfile.ZipFile(a.phase3_data) as z:
        for b in missing:
            ev=pd.read_csv(z.open(f'events_by_batch/batch_{b:02d}_lifecycle.csv.gz'),compression='gzip')
            ev=ev[ev.near_train_q1 & ev.clean_resolution.eq(1) & ev.trust_family.isin([x[0] for x in STATES])].copy();jobs.append((b,str(fmap[b]),mf[mf.batch.eq(b)].copy(),ev))
    with Pool(a.workers) as pool:
        for k,(b,tr,pa) in enumerate(pool.imap_unordered(process_batch,jobs),1):
            pd.DataFrame(tr).to_csv(out/f'batch_{b:02d}_trades.csv.gz',index=False,compression='gzip');pd.DataFrame(pa,columns=['trade_id','time','raw_return']).to_csv(out/f'batch_{b:02d}_paths.csv.gz',index=False,compression='gzip');print('checkpoint',k,'/',len(jobs),'batch',b,'trades',len(tr),flush=True)
    aggregate(out,raw)

if __name__=='__main__':main()
