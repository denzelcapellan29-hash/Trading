from __future__ import annotations
import argparse, math, zipfile, warnings, json
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

MAX_AGE=100
CLUSTER_BAND_ATR=.15
MAP_RADIUS_ATR=3.0
RESOLUTION_ATR=.50
PRIMARY_TARGET_MIN_ATR=.50
PRIMARY_TARGET_MAX_ATR=1.50
PROMOTED={
    ('usable_side_mixed','hold'),
    ('corridor','hold'),
    ('corridor','accept'),
    ('fragile_gateway_proxy','accept'),
}
PERIODS=['TRAIN_2013_2016','VALID_2017_2021','HOLDOUT_2022PLUS']


def make_pivots(H,L,C,k,max_age=MAX_AGE):
    n=len(C); out=[]
    for kind,arr in ((1,H),(-1,L)):
        s=pd.Series(arr)
        roll=(s.rolling(2*k+1,center=True,min_periods=2*k+1).max() if kind==1 else s.rolling(2*k+1,center=True,min_periods=2*k+1).min()).to_numpy()
        inds=np.flatnonzero(np.isfinite(arr)&np.isclose(arr,roll,rtol=0,atol=1e-12))
        conf=inds+k; keep=conf<n; inds=inds[keep]; conf=conf[keep]; levels=arr[inds]
        invalid=np.full(len(inds),n+1,dtype=int)
        for q,(cf,lv) in enumerate(zip(conf,levels)):
            end=min(n,cf+max_age+1)
            cond=C[cf+1:end]>lv if kind==1 else C[cf+1:end]<lv
            ix=np.flatnonzero(cond)
            if len(ix)>=2: invalid[q]=cf+1+ix[1]
        if len(levels):
            kd=np.full(len(levels),kind,dtype=int)
            out.extend(zip(conf,invalid,levels,kd))
    if not out:
        return tuple(np.array([]) for _ in range(4))
    x=np.array(out,float); order=np.argsort(x[:,0])
    return x[order,0].astype(int),x[order,1].astype(int),x[order,2],x[order,3].astype(int)


def active_pivots(piv,t,max_age=MAX_AGE):
    conf,invalid,level,kind=piv
    if len(conf)==0: return np.empty((0,4))
    mask=(conf<=t)&(invalid>t)&((t-conf)<=max_age)
    return np.column_stack([conf[mask],invalid[mask],level[mask],kind[mask]])


def active_levels(piv,t,max_age=MAX_AGE):
    x=active_pivots(piv,t,max_age)
    return x[:,2] if len(x) else np.array([])


def build_nodes(p2,p5,p10,t,px,a):
    fast=active_pivots(p2,t)
    if len(fast)==0:return []
    order=np.argsort(fast[:,2]); fast=fast[order]
    levels=fast[:,2]; band=CLUSTER_BAND_ATR*a
    gaps=np.flatnonzero(np.diff(levels)>band)+1; ends=np.r_[gaps,len(levels)]
    med=active_levels(p5,t); slow=active_levels(p10,t)
    nodes=[]; st=0
    for en in ends:
        grp=fast[st:en]; vv=grp[:,2]
        lv=float(vv.mean()); dist=(lv-px)/a
        if abs(dist)<=MAP_RADIUS_ATR and abs(dist)>=1e-12:
            members=len(vv)
            div=1+(int(np.any(np.abs(med-lv)<=band)) if len(med) else 0)+(int(np.any(np.abs(slow-lv)<=band)) if len(slow) else 0)
            q=int(members>=3)+int(div>=2)
            nodes.append({'level':lv,'members':members,'source_diversity':int(div),'Q':q,'dist':dist})
        st=en
    return nodes


def first_barrier_exit(O,H,L,start,direction,target,stop):
    n=len(O)
    for j in range(start,n):
        op=O[j]; hi=H[j]; lo=L[j]
        if direction>0:
            # Gap handling first. If both concepts crossed by an impossible pathological bar, invalidate conservatively.
            if op<=stop: return j,float(op),'stop_gap',0
            if op>=target: return j,float(op),'target_gap',0
            hit_stop=lo<=stop; hit_target=hi>=target
        else:
            if op>=stop: return j,float(op),'stop_gap',0
            if op<=target: return j,float(op),'target_gap',0
            hit_stop=hi>=stop; hit_target=lo<=target
        if hit_stop and hit_target:
            return j,float(stop),'ambiguous_stop_first',1
        if hit_stop:return j,float(stop),'stop',0
        if hit_target:return j,float(target),'target',0
    return None,np.nan,'censored',0


def trade_path(ts,O,C,entry_idx,exit_idx,entry_price,exit_price,direction):
    rows=[]
    prev=entry_price
    for j in range(entry_idx,exit_idx+1):
        mark=exit_price if j==exit_idx else C[j]
        r=direction*(mark-prev)/prev
        rows.append((ts.iloc[j].isoformat(),float(r)))
        prev=mark
    return rows


def process_stock(args):
    ticker,slot,frame,events,qr_cuts=args
    cols=[f'S{slot:02d}_{x}' for x in ['O','H','L','C','V']]
    vals=frame[cols].apply(pd.to_numeric,errors='coerce')
    mask=vals.iloc[:,:4].notna().all(axis=1).to_numpy()
    vals=vals.loc[mask].reset_index(drop=True)
    ts=pd.to_datetime(frame.loc[mask,'time'],errors='coerce',utc=True).dt.tz_convert('America/New_York').reset_index(drop=True)
    O,H,L,C,V=[vals.iloc[:,i].to_numpy(float) for i in range(5)]; n=len(C)
    if n<200 or events.empty:return [],[]
    prev=np.r_[np.nan,C[:-1]]; tr=np.maximum(H-L,np.maximum(np.abs(H-prev),np.abs(L-prev)))
    atr=pd.Series(tr).rolling(100,min_periods=50).mean().shift(1).to_numpy()
    p2=make_pivots(H,L,C,2); p5=make_pivots(H,L,C,5); p10=make_pivots(H,L,C,10)
    # timestamp index in nanoseconds is fast and DST-safe because all are tz-aware.
    idxmap={int(x.value):i for i,x in enumerate(ts)}
    node_cache={}; trades=[]; pathrows=[]
    for r in events.itertuples(index=False):
        snap=pd.Timestamp(r.snapshot_time); touch=pd.Timestamp(r.touch_time)
        si=idxmap.get(int(snap.value)); ti=idxmap.get(int(touch.value))
        if si is None or ti is None or ti>=n-1:continue
        a=atr[si]
        if not np.isfinite(a) or a<=0:continue
        if si not in node_cache:node_cache[si]=build_nodes(p2,p5,p10,si,C[si],a)
        nodes=node_cache[si]
        if not nodes:continue
        side_sign=1 if r.interacted_side=='upper' else -1
        expected=C[si]+side_sign*float(r.dist_atr)*a
        ni=int(np.argmin([abs(x['level']-expected) for x in nodes])); node=nodes[ni]
        # Reject any map reconstruction mismatch greater than the original cluster band.
        if abs(node['level']-expected) > CLUSTER_BAND_ATR*a+1e-9:continue
        lv=node['level']; upper=lv>C[si]
        hi=H[ti:min(n,ti+6)]; lo=L[ti:min(n,ti+6)]
        if upper:
            hh=np.flatnonzero(lo<=lv-RESOLUTION_ATR*a); aa=np.flatnonzero(hi>=lv+RESOLUTION_ATR*a)
        else:
            hh=np.flatnonzero(hi>=lv+RESOLUTION_ATR*a); aa=np.flatnonzero(lo<=lv-RESOLUTION_ATR*a)
        hd=int(hh[0]) if len(hh) else 99; ad=int(aa[0]) if len(aa) else 99
        if hd==ad or (hd==99 and ad==99):continue
        resolution='hold' if hd<ad else 'accept'
        if resolution!=r.resolution:continue
        ridx=ti+min(hd,ad)
        direction=(-1 if upper else 1) if resolution=='hold' else (1 if upper else -1)
        qr=float(direction*(C[ridx]-lv)/a)
        qkey=(str(r.trust_family),resolution)
        q67=qr_cuts.get(qkey,np.nan)
        qr_high=bool(np.isfinite(q67) and qr>=q67)
        promoted=qkey in PROMOTED
        # Phase-4 corrected target: the nearest *frozen map* node still ahead of confirmation close.
        cand=[]
        for k,n2 in enumerate(nodes):
            if k==ni:continue
            lv2=n2['level']
            if direction>0 and lv2>C[ridx]:cand.append((lv2-C[ridx],k))
            elif direction<0 and lv2<C[ridx]:cand.append((C[ridx]-lv2,k))
        if not cand:continue
        delta,ki=min(cand); target_node=nodes[ki]; target=target_node['level']; td=float(delta/a)
        if td<PRIMARY_TARGET_MIN_ATR or td>PRIMARY_TARGET_MAX_ATR:continue
        entry_idx=ridx+1
        if entry_idx>=n:continue
        entry=float(O[entry_idx]); stop=float(lv-direction*RESOLUTION_ATR*a)
        # The mapped destination and invalidation must both still lie on the correct side at executable next open.
        if direction*(target-entry)<=0 or direction*(entry-stop)<=0:continue
        exit_idx,exit_px,exit_reason,amb=first_barrier_exit(O,H,L,entry_idx,direction,target,stop)
        if exit_idx is None:continue
        gross=float(direction*(exit_px-entry)/entry)
        risk=float(direction*(entry-stop)/entry); reward=float(direction*(target-entry)/entry)
        t={
            'ticker':ticker,'snapshot_time':r.snapshot_time,'touch_time':r.touch_time,'resolution_time':ts.iloc[ridx].isoformat(),
            'entry_time':ts.iloc[entry_idx].isoformat(),'exit_time':ts.iloc[exit_idx].isoformat(),'period':r.period,
            'trust_family':r.trust_family,'resolution':resolution,'direction':direction,'interacted_Q':int(r.Q_simple),
            'target_Q':int(target_node['Q']),'qr':qr,'qr_q67':q67,'qr_high':qr_high,'promoted_state':promoted,
            'target_dist_atr_from_confirmation':td,'entry_price':entry,'target_price':target,'stop_price':stop,'exit_price':exit_px,
            'gross_return':gross,'risk_frac':risk,'reward_frac':reward,'reward_risk':reward/risk if risk>0 else np.nan,
            'bars_held':int(exit_idx-entry_idx+1),'exit_reason':exit_reason,'ambiguous_bar':amb,
            'resolution_idx':ridx,'entry_idx':entry_idx,'exit_idx':exit_idx,
        }
        # Keep all promoted-state candidates; variants are formed later from QR/Q without changing executions.
        if promoted:
            tid=f"{ticker}|{entry_idx}|{direction}|{target:.8g}|{si}"
            t['trade_id']=tid; trades.append(t)
            for tm,rr in trade_path(ts,O,C,entry_idx,exit_idx,entry,exit_px,direction):pathrows.append((tid,tm,rr))
    return trades,pathrows


def process_batch(args):
    batch,raw_file,bmap,events,qr_cuts=args
    use=['time']
    for rr in bmap.itertuples(index=False):use += [f'S{int(rr.slot):02d}_{x}' for x in ['O','H','L','C','V']]
    frame=pd.read_csv(raw_file,usecols=use)
    trades=[]; paths=[]
    for rr in bmap.itertuples(index=False):
        ev=events[events.ticker.astype(str)==str(rr.ticker)]
        if ev.empty:continue
        tr,pa=process_stock((str(rr.ticker),int(rr.slot),frame,ev,qr_cuts));trades.extend(tr);paths.extend(pa)
    return batch,trades,paths


def dedupe_and_nonoverlap(trades,variant):
    x=trades.copy()
    if variant=='PROMOTED_ALL': pass
    elif variant=='HIGH_QR': x=x[x.qr_high]
    elif variant=='HIGH_QR_INTERACTED_Q2': x=x[x.qr_high & x.interacted_Q.eq(2)]
    elif variant=='HIGH_QR_TARGET_Q2': x=x[x.qr_high & x.target_Q.eq(2)]
    elif variant=='HIGH_QR_BOTH_Q2': x=x[x.qr_high & x.interacted_Q.eq(2) & x.target_Q.eq(2)]
    else: raise ValueError(variant)
    if x.empty:return x
    for c in ['snapshot_time','entry_time','exit_time','resolution_time']:x[c]=pd.to_datetime(x[c],utc=True)
    # Same execution can be generated by overlapping prior EOD snapshots. Keep the latest causal snapshot.
    x=x.sort_values(['ticker','entry_time','snapshot_time']).drop_duplicates(['ticker','entry_time','direction'],keep='last')
    # One open position per ticker. Signals arriving while an existing trade is active are ignored.
    keep=[]
    for tic,g in x.sort_values(['ticker','entry_time']).groupby('ticker',sort=False):
        last_exit=pd.Timestamp.min.tz_localize('UTC')
        for i,r in g.iterrows():
            if r.entry_time>last_exit:
                keep.append(i); last_exit=r.exit_time
    return x.loc[keep].sort_values('entry_time').reset_index(drop=True)


def pf_and_stats(x,cost_bps):
    if x.empty:return {}
    net=x.gross_return.to_numpy()-2*cost_bps*1e-4
    wins=net[net>0]; losses=net[net<0]
    pf=wins.sum()/(-losses.sum()) if len(losses) and losses.sum()<0 else np.inf
    return {
        'n':len(x),'win_rate':float((net>0).mean()),'mean_trade':float(net.mean()),'median_trade':float(np.median(net)),
        'profit_factor':float(pf),'avg_win':float(wins.mean()) if len(wins) else np.nan,'avg_loss':float(losses.mean()) if len(losses) else np.nan,
        'payoff_ratio':float(wins.mean()/(-losses.mean())) if len(wins) and len(losses) else np.nan,
        'mean_bars_held':float(x.bars_held.mean()),'median_bars_held':float(x.bars_held.median()),
        'mean_rr_at_entry':float(x.reward_risk.mean()),'target_exit_share':float(x.exit_reason.str.startswith('target').mean()),
        'stop_exit_share':float((x.exit_reason.str.startswith('stop')|x.exit_reason.eq('ambiguous_stop_first')).mean()),
        'ambiguous_share':float(x.ambiguous_bar.mean()),
    }


def active_book_metrics(x,paths,cost_bps,calendar):
    if x.empty:return {}
    ids=set(x.trade_id); p=paths[paths.trade_id.isin(ids)].copy()
    if p.empty:return {}
    p['time']=pd.to_datetime(p.time,utc=True)
    timing=x[['trade_id','entry_time','exit_time']].copy()
    p=p.merge(timing,on='trade_id',how='left',validate='many_to_one')
    p['cost']=0.0
    cb=cost_bps*1e-4
    p.loc[p['time'].eq(p['entry_time']),'cost'] += cb
    p.loc[p['time'].eq(p['exit_time']),'cost'] += cb
    p['net_ret']=p.raw_return-p.cost
    # Standardized diagnostic book: equal notional across active trades at each 78m bar, 100% gross when active.
    br=p.groupby('time').agg(ret=('net_ret','mean'),active=('trade_id','nunique')).sort_index()
    idx=calendar[(calendar>=br.index.min())&(calendar<=br.index.max())]
    br=br.reindex(idx,fill_value=0.0)
    eq=(1+br.ret).cumprod(); years=(br.index[-1]-br.index[0]).total_seconds()/(365.25*24*3600)
    cagr=float(eq.iloc[-1]**(1/years)-1) if years>0 and eq.iloc[-1]>0 else np.nan
    ann=252*5
    vol=float(br.ret.std(ddof=1)*np.sqrt(ann)); sharpe=float(br.ret.mean()/br.ret.std(ddof=1)*np.sqrt(ann)) if br.ret.std(ddof=1)>0 else np.nan
    dn=br.ret[br.ret<0]; sortino=float(br.ret.mean()/dn.std(ddof=1)*np.sqrt(ann)) if len(dn)>1 and dn.std(ddof=1)>0 else np.nan
    dd=eq/eq.cummax()-1; mdd=float(dd.min()); calmar=float(cagr/abs(mdd)) if mdd<0 else np.nan
    return {'book_cagr':cagr,'book_vol':vol,'book_sharpe':sharpe,'book_sortino':sortino,'book_max_dd':mdd,'book_calmar':calmar,
            'book_end_multiple':float(eq.iloc[-1]),'mean_active_positions_when_active':float(br.active.mean()),'max_active_positions':int(br.active.max()),
            'bars_active':len(br),'start':str(br.index[0]),'end':str(br.index[-1])}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-dir',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--inference',required=True)
    ap.add_argument('--phase3-data',required=True);ap.add_argument('--phase4-package',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=min(6,cpu_count()))
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    raw=Path(a.raw_dir);mf=pd.read_csv(a.manifest);inf=pd.read_csv(a.inference)
    with zipfile.ZipFile(a.phase4_package) as z:
        member=[n for n in z.namelist() if n.endswith('qr_train_state_cuts_full503.csv')][0]
        cuts=pd.read_csv(z.open(member))
    qr_cuts={(r.trust_family,r.resolution):float(r.q67) for r in cuts.itertuples(index=False)}
    # Only batches whose exact raw source is actually persisted locally. Never silently substitute missing source bars.
    fmap={}
    for r in inf.itertuples(index=False):
        if bool(r.is_duplicate_payload):continue
        f=raw/str(r.file)
        if f.exists():fmap[int(r.batch)]=f
    batches=sorted(set(mf.batch.astype(int)) & set(fmap))
    with zipfile.ZipFile(a.phase3_data) as z:
        jobs=[]
        for b in batches:
            ev=pd.read_csv(z.open(f'events_by_batch/batch_{b:02d}_lifecycle.csv.gz'),compression='gzip')
            ev=ev[ev.near_train_q1 & ev.clean_resolution.eq(1) & ev.trust_family.isin([x[0] for x in PROMOTED])].copy()
            bm=mf[mf.batch.eq(b)].copy();jobs.append((b,str(fmap[b]),bm,ev,qr_cuts))
    print('available raw batches',len(batches),batches,'stocks',len(mf[mf.batch.isin(batches)]),'workers',a.workers,flush=True)
    all_tr=[]; all_paths=[]
    with Pool(a.workers) as pool:
        for k,(b,tr,pa) in enumerate(pool.imap_unordered(process_batch,jobs),1):
            all_tr.extend(tr);all_paths.extend(pa);print('done',k,'/',len(jobs),'batch',b,'trades',len(tr),flush=True)
    t=pd.DataFrame(all_tr);p=pd.DataFrame(all_paths,columns=['trade_id','time','raw_return'])
    if t.empty:raise RuntimeError('No candidate trades reconstructed')
    t.to_csv(out/'phase5_promoted_state_candidate_trades.csv.gz',index=False,compression='gzip')
    p.to_csv(out/'phase5_candidate_trade_paths.csv.gz',index=False,compression='gzip')
    ref=next(raw.glob('*.csv'))
    calendar=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ref,usecols=['time']).time,utc=True).dropna().drop_duplicates().sort_values())
    variants=['PROMOTED_ALL','HIGH_QR','HIGH_QR_INTERACTED_Q2','HIGH_QR_TARGET_Q2','HIGH_QR_BOTH_Q2']
    costs=[0,1,2,5,10]; rows=[]; subrows=[]; kept={}
    for v in variants:
        x=dedupe_and_nonoverlap(t,v);kept[v]=x;x.to_csv(out/f'trades_{v.lower()}.csv',index=False)
        for cost in costs:
            s=pf_and_stats(x,cost);s.update(active_book_metrics(x,p,cost,calendar));s.update({'variant':v,'one_way_cost_bps':cost,'sample':'ALL'});rows.append(s)
            for per in PERIODS:
                g=x[x.period.eq(per)];ss=pf_and_stats(g,cost);ss.update({'variant':v,'one_way_cost_bps':cost,'sample':per});subrows.append(ss)
    pd.DataFrame(rows).to_csv(out/'portfolio_metrics_all.csv',index=False);pd.DataFrame(subrows).to_csv(out/'trade_metrics_by_period.csv',index=False)
    # Structural composition / long-short / family-resolution diagnostics at 2 bps one-way.
    diag=[]
    x=kept['HIGH_QR']
    for keys,g in x.groupby(['period','trust_family','resolution','direction']):
        s=pf_and_stats(g,2);s.update(dict(zip(['period','trust_family','resolution','direction'],keys)));diag.append(s)
    pd.DataFrame(diag).to_csv(out/'high_qr_state_direction_diagnostics_2bps.csv',index=False)
    methodology={
        'phase':'Phase 5 standalone structural portfolio pilot','status':'PILOT because exact raw bars currently materialized for only persisted raw batches',
        'available_batches':batches,'available_stocks':int(len(mf[mf.batch.isin(batches)])),'full_universe':503,
        'entry':'next 78m bar open after observed 0.5 ATR hold/accept resolution','target':'nearest frozen-map node still ahead of resolution-confirmation close',
        'target_distance_atr':[.5,1.5],'invalidation':'interacted node +/- 0.5 snapshot ATR opposite realized resolution direction',
        'time_exit':None,'same_bar_target_stop':'stop-first conservative','entry_gap_rule':'skip if target already passed or structural invalidation already breached before executable next open',
        'geometry':'TRAIN-fixed nearest first quartile inherited from Phase4','lifecycle':'diagnostic only, no gate','qr':'TRAIN-only trust_family x resolution q67 cuts inherited from Phase4',
        'promoted_qr_states':sorted([list(x) for x in PROMOTED]),'q_variants':'Q evaluated separately; not redefined as hold probability',
        'duplicate_control':'latest snapshot for duplicate same ticker/entry/direction; max one open trade per ticker','portfolio':'standardized equal-weight active-book diagnostic, 100% gross when active',
        'costs_one_way_bps':costs,'pnl_used_to_choose_structural_parameters':False,
    }
    (out/'PHASE5_METHODOLOGY.json').write_text(json.dumps(methodology,indent=2))
    print('\nALL METRICS')
    m=pd.DataFrame(rows);print(m[m.one_way_cost_bps.eq(2)][['variant','n','win_rate','mean_trade','profit_factor','mean_bars_held','mean_rr_at_entry','book_cagr','book_vol','book_sharpe','book_sortino','book_max_dd','mean_active_positions_when_active']].to_string(index=False))
    print('\nPERIOD 2BPS')
    sm=pd.DataFrame(subrows);print(sm[sm.one_way_cost_bps.eq(2)][['variant','sample','n','win_rate','mean_trade','profit_factor']].to_string(index=False))

if __name__=='__main__':main()
