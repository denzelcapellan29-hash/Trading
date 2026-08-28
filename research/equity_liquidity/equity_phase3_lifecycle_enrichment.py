from __future__ import annotations
import pandas as pd, numpy as np, zipfile, warnings
from pathlib import Path
from numba import njit
from multiprocessing import Pool
warnings.filterwarnings('ignore')

OLD_ZIP=Path('/mnt/data/38632e72-5f3c-4034-b3ae-781089b8261c.zip')
NEW_ZIP=Path('/mnt/data/a909fd01-a682-4c66-93ed-1250029375b6.zip')
INFERENCE=Path('/mnt/data/SP500_78m_complete_final_2026-08-27/source_file_batch_inference.csv')
MANIFEST=Path('/mnt/data/SP500_78m_complete_ingest_2026-08-27/full_503_batch_manifest_complete.csv')
PHASE2=Path('/mnt/data/phase2_q_interaction_events_full503.csv.gz')
MAX_AGE=100; BAND=.15; TRAIN_NEAR_Q=.357323996666185

@njit(cache=True)
def make_pivots(H,L,C,k,max_age):
    n=len(C); maxout=2*n
    conf=np.empty(maxout,np.int32); invalid=np.empty(maxout,np.int32); level=np.empty(maxout,np.float64); count=0
    for p in range(k,n-k):
        hv=H[p]; lv=L[p]; ishi=True; islo=True
        for j in range(p-k,p+k+1):
            if H[j] > hv + 1e-12: ishi=False
            if L[j] < lv - 1e-12: islo=False
        if ishi:
            cf=p+k; inv=n+1; through=0; end=min(n,cf+max_age+1)
            for j in range(cf+1,end):
                if C[j]>hv:
                    through+=1
                    if through>=2: inv=j; break
            conf[count]=cf; invalid[count]=inv; level[count]=hv; count+=1
        if islo:
            cf=p+k; inv=n+1; through=0; end=min(n,cf+max_age+1)
            for j in range(cf+1,end):
                if C[j]<lv:
                    through+=1
                    if through>=2: inv=j; break
            conf[count]=cf; invalid[count]=inv; level[count]=lv; count+=1
    conf=conf[:count]; invalid=invalid[:count]; level=level[:count]
    order=np.argsort(conf)
    return conf[order],invalid[order],level[order]

@njit(cache=True)
def lower_bound(a,x):
    lo=0; hi=len(a)
    while lo<hi:
        mid=(lo+hi)//2
        if a[mid] < x: lo=mid+1
        else: hi=mid
    return lo

@njit(cache=True)
def enrich_events(H,L,C,ATR,day_id,confs,invalids,levels,event_ix,event_dist,event_upper,event_members,max_age,band_atr):
    n=len(event_ix)
    known_ep=np.zeros(n,np.int16); known_bars=np.zeros(n,np.int16); known_sessions=np.zeros(n,np.int16)
    zone_sessions=np.zeros(n,np.int16); age=np.full(n,-1,np.int16); since=np.full(n,np.nan)
    parity=np.zeros(n,np.int8); found=np.zeros(n,np.int8)
    for e in range(n):
        t=event_ix[e]; a=ATR[t]
        if not np.isfinite(a) or a<=0: continue
        target=C[t] + (event_dist[e]*a if event_upper[e] else -event_dist[e]*a)
        loix=lower_bound(confs,t-max_age); hiix=lower_bound(confs,t+1)
        m=0; tmp_lv=np.empty(128,np.float64); tmp_cf=np.empty(128,np.int32)
        for q in range(loix,hiix):
            if invalids[q] > t and t-confs[q] <= max_age:
                if m<128:
                    tmp_lv[m]=levels[q]; tmp_cf[m]=confs[q]; m+=1
        if m==0: continue
        for i in range(1,m):
            x=tmp_lv[i]; y=tmp_cf[i]; j=i-1
            while j>=0 and tmp_lv[j]>x:
                tmp_lv[j+1]=tmp_lv[j]; tmp_cf[j+1]=tmp_cf[j]; j-=1
            tmp_lv[j+1]=x; tmp_cf[j+1]=y
        bestdiff=1e99; bests=-1; beste=-1; s=0
        for i in range(1,m+1):
            endcluster=(i==m) or (tmp_lv[i]-tmp_lv[i-1] > band_atr*a)
            if endcluster:
                sm=0.0
                for j in range(s,i): sm+=tmp_lv[j]
                mean=sm/(i-s); diff=abs(mean-target)
                if diff<bestdiff:
                    bestdiff=diff; bests=s; beste=i
                s=i
        if bests<0 or bestdiff > 0.03*a: continue
        found[e]=1; parity[e]=1 if (beste-bests)==event_members[e] else 0
        zlo=tmp_lv[bests]; zhi=tmp_lv[beste-1]
        birth=tmp_cf[bests]
        for j in range(bests+1,beste):
            if tmp_cf[j]<birth: birth=tmp_cf[j]
        age[e]=t-birth
        st=birth+1
        prevhit=False; ep=0; nb=0; ns=0; last=-1; last_day=-999999
        if st<=t:
            for j in range(st,t+1):
                hit=(H[j]>=zlo and L[j]<=zhi)
                if hit:
                    nb+=1; last=j
                    if not prevhit: ep+=1
                    if day_id[j]!=last_day:
                        ns+=1; last_day=day_id[j]
                prevhit=hit
        known_ep[e]=ep; known_bars[e]=nb; known_sessions[e]=ns
        if last>=0: since[e]=t-last
        stz=max(0,t-max_age+1); zsess=0; lastz=-999999
        for j in range(stz,t+1):
            hit=(H[j]>=zlo and L[j]<=zhi)
            if hit and day_id[j]!=lastz:
                zsess+=1; lastz=day_id[j]
        zone_sessions[e]=zsess
    return known_ep,known_bars,known_sessions,zone_sessions,age,since,parity,found

def source_map():
    inf=pd.read_csv(INFERENCE); out={}
    for _,r in inf.iterrows():
        if bool(r.is_duplicate_payload): continue
        b=int(r.batch); prefix,fname=str(r.file).split('/',1)
        if b in out and prefix=='old': continue
        out[b]=(NEW_ZIP if prefix=='new' else OLD_ZIP,fname)
    return out

def process_batch(args):
    b,bmap,ev_batch,src=args
    zpath,fname=src[b]
    use=['time']
    for s in bmap.slot.astype(int): use += [f'S{s:02d}_{x}' for x in ['H','L','C']]
    with zipfile.ZipFile(zpath) as z: frame=pd.read_csv(z.open(fname),usecols=use)
    ft=pd.to_datetime(frame.time,errors='coerce',utc=True)
    pieces=[]; parity_rows=[]
    for _,r in bmap.iterrows():
        tkr=str(r.ticker); s=int(r.slot); ev=ev_batch[ev_batch.ticker==tkr].copy()
        if ev.empty: continue
        vals=frame[[f'S{s:02d}_H',f'S{s:02d}_L',f'S{s:02d}_C']].apply(pd.to_numeric,errors='coerce')
        mask=vals.notna().all(axis=1).to_numpy(); vals=vals.loc[mask].reset_index(drop=True); ts=ft.loc[mask].reset_index(drop=True)
        H,L,C=[vals.iloc[:,i].to_numpy(float) for i in range(3)]
        prev=np.r_[np.nan,C[:-1]]; tr=np.maximum(H-L,np.maximum(np.abs(H-prev),np.abs(L-prev)))
        ATR=pd.Series(tr).rolling(100,min_periods=50).mean().shift(1).to_numpy()
        day_id=pd.factorize(ts.dt.floor('D'))[0].astype(np.int32)
        conf,inv,lev=make_pivots(H,L,C,2,MAX_AGE)
        idx=pd.Series(np.arange(len(ts)),index=ts).groupby(level=0).last()
        edt=pd.to_datetime(ev.snapshot_time,utc=True); ix=idx.reindex(edt).to_numpy(); ok=np.isfinite(ix)
        ev=ev.loc[ok].copy(); ix=ix[ok].astype(np.int64)
        ep,tb,tsess,zsess,age,last,par,found=enrich_events(H,L,C,ATR,day_id,conf,inv,lev,ix,
            ev.dist_atr.to_numpy(float),ev.interacted_side.to_numpy()=='upper',ev.members.to_numpy(np.int64),MAX_AGE,BAND)
        ev['known_touch_episodes']=ep; ev['known_touch_bars']=tb; ev['known_touch_sessions']=tsess
        ev['zone_touch_sessions_100']=zsess; ev['node_age_bars']=age; ev['bars_since_last_known_touch']=last
        ev['cluster_member_parity']=par; ev['cluster_found']=found
        ev=ev[ev.cluster_found==1].copy()
        parity_rows.append({'ticker':tkr,'n':len(ev),'member_parity':ev.cluster_member_parity.mean() if len(ev) else np.nan})
        pieces.append(ev)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(), pd.DataFrame(parity_rows)

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--batches',required=True,help='comma-separated batch numbers')
    ap.add_argument('--outdir',default='/mnt/data/phase3_batches')
    args=ap.parse_args(); wanted={int(x) for x in args.batches.split(',') if x.strip()}
    outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    H=np.array([1.,2.,1.,2.,1.,2.,1.]); L=H-.5; C=H-.2; A=np.ones(7); D=np.arange(7,dtype=np.int32)
    cf,iv,lv=make_pivots(H,L,C,1,5)
    if len(cf): enrich_events(H,L,C,A,D,cf,iv,lv,np.array([5],np.int64),np.array([.5]),np.array([True]),np.array([1],np.int64),5,.15)
    mf=pd.read_csv(MANIFEST); ev=pd.read_csv(PHASE2); src=source_map(); tasks=[]
    for b,bmap in mf.groupby('batch'):
        if int(b) not in wanted: continue
        tick=set(bmap.ticker.astype(str)); tasks.append((int(b),bmap.copy(),ev[ev.ticker.isin(tick)].copy(),src))
    with Pool(min(len(tasks),8)) as pool: res=pool.map(process_batch,tasks)
    for (b,_,_,_), (x,p) in zip(tasks,res):
        if x.empty: continue
        x['snapshot_dt']=pd.to_datetime(x.snapshot_time,utc=True); y=x.snapshot_dt.dt.year
        x['period']=np.select([y<=2016,y<=2021],['TRAIN_2013_2016','VALID_2017_2021'],default='HOLDOUT_2022PLUS')
        x['week']=x.snapshot_dt.dt.to_period('W-FRI').astype(str); x['near_train_q1']=x.dist_atr<=TRAIN_NEAR_Q
        x['known_episode_lifecycle']=np.where(x.known_touch_episodes<=1,'fresh_0_1',np.where(x.known_touch_episodes<=4,'intermediate_2_4','depleted_5plus'))
        x['known_session_lifecycle']=np.where(x.known_touch_sessions<=1,'fresh_0_1',np.where(x.known_touch_sessions<=4,'intermediate_2_4','depleted_5plus'))
        x['zone_session_lifecycle']=np.where(x.zone_touch_sessions_100<=1,'fresh_0_1',np.where(x.zone_touch_sessions_100<=4,'intermediate_2_4','depleted_5plus'))
        x.drop(columns=['snapshot_dt'],inplace=True)
        x.to_csv(outdir/f'batch_{b:02d}_lifecycle.csv.gz',index=False,compression='gzip')
        p.to_csv(outdir/f'batch_{b:02d}_parity.csv',index=False)
        print('batch',b,'events',len(x),'stocks',x.ticker.nunique(),'parity',x.cluster_member_parity.mean(),flush=True)

if __name__=='__main__': main()
