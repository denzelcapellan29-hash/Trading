#!/usr/bin/env python3
from __future__ import annotations
import argparse, math, re, zipfile, gzip, io
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

MAX_AGE=100
CLUSTER_BAND_ATR=.15
MAP_RADIUS_ATR=3.0
ACTIONABLE_ATR=1.5
SIDE_SHARE=.65
SIDE_DECAY=.75


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
        for cf,iv,lv,kd in zip(conf,invalid,levels,np.full(len(levels),kind)):
            out.append((cf,iv,lv,kd))
    if not out: return tuple(np.array([]) for _ in range(4))
    x=np.array(out,float); order=np.argsort(x[:,0])
    return x[order,0].astype(int), x[order,1].astype(int), x[order,2], x[order,3].astype(int)


def active_levels(piv,t,max_age=MAX_AGE):
    conf,invalid,level,kind=piv
    if len(conf)==0: return np.array([])
    mask=(conf<=t)&(invalid>t)&((t-conf)<=max_age)
    return level[mask]


def classify_family(nodes):
    ups=[x for x in nodes if x[3]>0]; dns=[x for x in nodes if x[3]<0]
    up_near=min((x[3] for x in ups),default=np.inf)
    dn_near=min((-x[3] for x in dns),default=np.inf)
    um=sum(math.exp(-abs(x[3])/SIDE_DECAY) for x in ups)
    dm=sum(math.exp(-abs(x[3])/SIDE_DECAY) for x in dns)
    total=um+dm
    if total<=0: return 'no_map',0,np.nan,np.nan,um,dm
    share=max(um,dm)/total; bias=1 if um>=dm else -1
    if up_near<=ACTIONABLE_ATR and dn_near<=ACTIONABLE_ATR and share<SIDE_SHARE:
        return 'corridor',bias,share,min(up_near,dn_near),um,dm
    if share>=SIDE_SHARE:
        side=ups if bias==1 else dns
        if side:
            side=sorted(side,key=lambda x:abs(x[3])); nearest=side[0]; near_dist=abs(nearest[3])
            denser_behind=any(x[1]>=2 for x in side[1:])
            if near_dist<=ACTIONABLE_ATR and nearest[1]==1 and denser_behind:
                return 'fragile_gateway_proxy',bias,share,near_dist,um,dm
            if near_dist<=ACTIONABLE_ATR:
                return 'usable_side_mixed',bias,share,near_dist,um,dm
    nd=min(up_near,dn_near)
    return 'other_map',bias,share,nd if np.isfinite(nd) else np.nan,um,dm


def snapshot_features(frame,slot,ticker,signal_dates):
    names=[f'S{slot:02d}_{x}' for x in ['O','H','L','C','V']]
    vals=frame[names].apply(pd.to_numeric,errors='coerce')
    mask=vals.iloc[:,:4].notna().all(axis=1).to_numpy()
    vals=vals.loc[mask].reset_index(drop=True)
    ts=pd.to_datetime(frame.loc[mask,'time'],errors='coerce',utc=True).dt.tz_convert('America/New_York').reset_index(drop=True)
    O,H,L,C,V=[vals.iloc[:,i].to_numpy(float) for i in range(5)]
    n=len(C)
    if n<200: return pd.DataFrame()
    prev=np.r_[np.nan,C[:-1]]
    tr=np.maximum(H-L,np.maximum(np.abs(H-prev),np.abs(L-prev)))
    atr=pd.Series(tr).rolling(100,min_periods=50).mean().shift(1).to_numpy()
    p2=make_pivots(H,L,C,2); p5=make_pivots(H,L,C,5); p10=make_pivots(H,L,C,10)
    dates=np.array([d.isoformat() for d in ts.dt.date])
    eod=np.r_[np.flatnonzero(dates[1:]!=dates[:-1]),n-1]
    eod_by_date={dates[t]:int(t) for t in eod}
    rows=[]
    for sd in signal_dates:
        t=eod_by_date.get(str(sd))
        base={'ticker':ticker,'signal_date':str(sd),'liq_map_available':0,'liq_trust_family':'no_map',
              'liq_bias_side':0.0,'liq_side_share':np.nan,'liq_signed_imbalance':0.0,
              'liq_nearest_dist_atr':np.nan,'liq_nearest_Q':np.nan,'liq_nearest_members':np.nan,
              'liq_nearest_source_diversity':np.nan,'liq_node_count':0,'liq_up_node_count':0,'liq_down_node_count':0,
              'liq_up_mass':0.0,'liq_down_mass':0.0}
        if t is None or t<110:
            rows.append(base); continue
        a=atr[t]
        if not np.isfinite(a) or a<=0:
            rows.append(base); continue
        px=C[t]; fast=active_levels(p2,t)
        if len(fast)==0:
            rows.append(base); continue
        vals2=np.sort(fast); band=CLUSTER_BAND_ATR*a
        gaps=np.flatnonzero(np.diff(vals2)>band)+1; ends=np.r_[gaps,len(vals2)]
        clusters=[]; st=0
        for en in ends:
            vv=vals2[st:en]
            clusters.append((float(vv.mean()),len(vv),float((vv.max()-vv.min())/a if len(vv)>1 else 0.0)))
            st=en
        med=active_levels(p5,t); slow=active_levels(p10,t)
        nodes=[]
        for lv,members,span in clusters:
            dist=(lv-px)/a
            if abs(dist)>MAP_RADIUS_ATR or abs(dist)<1e-12: continue
            div=1+(int(np.any(np.abs(med-lv)<=band)) if len(med) else 0)+(int(np.any(np.abs(slow-lv)<=band)) if len(slow) else 0)
            nodes.append((lv,members,int(div),dist,span))
        if not nodes:
            rows.append(base); continue
        fam,bias,share,nearest_dist,um,dm=classify_family(nodes)
        nearest=min(nodes,key=lambda x:abs(x[3]))
        q=int(nearest[1]>=3)+int(nearest[2]>=2)
        base.update({
            'liq_map_available':1,'liq_trust_family':fam,'liq_bias_side':float(bias),'liq_side_share':float(share),
            'liq_signed_imbalance':float(bias*(2*share-1)),'liq_nearest_dist_atr':float(nearest_dist),
            'liq_nearest_Q':float(q),'liq_nearest_members':float(nearest[1]),'liq_nearest_source_diversity':float(nearest[2]),
            'liq_node_count':int(len(nodes)),'liq_up_node_count':int(sum(x[3]>0 for x in nodes)),
            'liq_down_node_count':int(sum(x[3]<0 for x in nodes)),'liq_up_mass':float(um),'liq_down_mass':float(dm)
        })
        rows.append(base)
    return pd.DataFrame(rows)


def process_batch(args):
    batch, raw_file, bmap, needs = args
    usecols=['time']
    for _,r in bmap.iterrows():
        if str(r.ticker) in needs:
            ss=int(r.slot); usecols += [f'S{ss:02d}_{x}' for x in ['O','H','L','C','V']]
    frame=pd.read_csv(raw_file,usecols=usecols)
    outs=[]
    for _,r in bmap.iterrows():
        tk=str(r.ticker)
        if tk not in needs: continue
        x=snapshot_features(frame,int(r.slot),tk,sorted(needs[tk]))
        if not x.empty:
            x['batch']=batch; x['slot']=int(r.slot); outs.append(x)
    return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--barbell-zip',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--raw-dir',required=True)
    ap.add_argument('--phase5-data-zip',required=False)
    ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.barbell_zip) as z:
        b=gzip.decompress(z.read('walkforward_trade_panel_with_fundamentals.csv.gz'))
        bar=pd.read_csv(io.BytesIO(b),parse_dates=['signal_date','entry_date','exit_date'])
        bb=gzip.decompress(z.read('branch_aware_trade_scores.csv.gz'))
        branch=pd.read_csv(io.BytesIO(bb),parse_dates=['signal_date'])[['signal_date','ticker','state_turnaround','state_leader']]
    bar=bar.merge(branch,on=['signal_date','ticker'],how='left',validate='one_to_one')
    manifest=pd.read_csv(a.manifest)
    needs={}
    for tk,g in bar.groupby('ticker'):
        needs[str(tk)]=set(g.signal_date.dt.strftime('%Y-%m-%d'))
    args=[]
    for b in sorted(manifest.batch.unique()):
        bmap=manifest[manifest.batch==b].copy()
        bn={tk:needs[tk] for tk in bmap.ticker.astype(str) if tk in needs}
        if not bn: continue
        raw=Path(a.raw_dir)/f'SP500_78M_B{int(b):02d}.csv'
        args.append((int(b),raw,bmap,bn))
    with Pool(min(a.workers,max(1,cpu_count()-1))) as pool:
        pieces=list(pool.imap_unordered(process_batch,args))
    feat=pd.concat([x for x in pieces if not x.empty],ignore_index=True)
    feat['signal_date']=pd.to_datetime(feat.signal_date)
    feat=feat.sort_values(['signal_date','ticker'])
    feat['liq_ca_active']=0
    feat['liq_ca_recent_5d']=0
    feat['liq_ca_recent_20d']=0
    if a.phase5_data_zip:
        with zipfile.ZipFile(a.phase5_data_zip) as z:
            raw=gzip.decompress(z.read('causal_target_trades_all_states.csv.gz'))
            ca=pd.read_csv(io.BytesIO(raw),parse_dates=['entry_time','exit_time'])
        ca=ca[(ca.trust_family=='corridor')&(ca.resolution=='accept')].copy()
        for tk,idx in feat.groupby('ticker').groups.items():
            ev=ca[ca.ticker==tk]
            if ev.empty: continue
            et=pd.to_datetime(ev.entry_time,utc=True); xt=pd.to_datetime(ev.exit_time,utc=True)
            for i in idx:
                st=pd.Timestamp(feat.at[i,'signal_date']).tz_localize('America/New_York')+pd.Timedelta(hours=16)
                st=st.tz_convert('UTC')
                feat.at[i,'liq_ca_active']=int(((et<=st)&(xt>=st)).any())
                feat.at[i,'liq_ca_recent_5d']=int(((et<st)&(et>=st-pd.Timedelta(days=7))).sum())
                feat.at[i,'liq_ca_recent_20d']=int(((et<st)&(et>=st-pd.Timedelta(days=28))).sum())
    panel=bar.merge(feat,on=['signal_date','ticker'],how='left',validate='one_to_one')
    panel.to_csv(out/'barbell_liquidity_causal_feature_panel.csv.gz',index=False,compression='gzip')
    feat.to_csv(out/'barbell_liquidity_causal_features.csv.gz',index=False,compression='gzip')
    print({'barbell_trades':len(bar),'feature_rows':len(feat),'matched':int(panel.liq_trust_family.notna().sum()),
           'map_available_2015plus':float(panel.loc[panel.signal_date>='2015-01-01','liq_map_available'].mean()),
           'ca_active_2015plus':int(panel.loc[panel.signal_date>='2015-01-01','liq_ca_active'].sum())})

if __name__=='__main__': main()
