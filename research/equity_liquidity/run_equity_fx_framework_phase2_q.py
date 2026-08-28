from __future__ import annotations
import pandas as pd, numpy as np, zipfile, io, os, json, math, hashlib, warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

OLD_ZIP=Path('/mnt/data/38632e72-5f3c-4034-b3ae-781089b8261c.zip')
NEW_ZIP=Path('/mnt/data/a909fd01-a682-4c66-93ed-1250029375b6.zip')
MANIFEST=Path('/mnt/data/SP500_78m_complete_ingest_2026-08-27/full_503_batch_manifest_complete.csv')
INFERENCE=Path('/mnt/data/SP500_78m_complete_final_2026-08-27/source_file_batch_inference.csv')
OUT=Path('/mnt/data/equity_fx_framework_phase2_q')
OUT.mkdir(parents=True,exist_ok=True)

MAX_AGE=100
CLUSTER_BAND_ATR=.15
MAP_RADIUS_ATR=3.0
ACTIONABLE_ATR=1.5
SIDE_SHARE=.65
SIDE_DECAY=.75
RESOLUTION_ATR=.50
RESOLUTION_BARS=5
FIRST_TOUCH_BARS=10
PATH_BARS=10


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
    ups=[x for x in nodes if x[3]>0]
    dns=[x for x in nodes if x[3]<0]
    up_near=min((x[3] for x in ups),default=np.inf)
    dn_near=min((-x[3] for x in dns),default=np.inf)
    um=sum(math.exp(-abs(x[3])/SIDE_DECAY) for x in ups)
    dm=sum(math.exp(-abs(x[3])/SIDE_DECAY) for x in dns)
    total=um+dm
    if total<=0: return 'other_map',0,0,np.nan
    share=max(um,dm)/total; bias=1 if um>=dm else -1
    if up_near<=ACTIONABLE_ATR and dn_near<=ACTIONABLE_ATR and share<SIDE_SHARE:
        return 'corridor',bias,share,min(up_near,dn_near)
    if share>=SIDE_SHARE:
        side=ups if bias==1 else dns
        if side:
            side=sorted(side,key=lambda x:abs(x[3]))
            nearest=side[0]
            near_dist=abs(nearest[3])
            denser_behind=any(x[1]>=2 for x in side[1:])
            if near_dist<=ACTIONABLE_ATR and nearest[1]==1 and denser_behind:
                return 'fragile_gateway_proxy',bias,share,near_dist
            if near_dist<=ACTIONABLE_ATR:
                return 'usable_side_mixed',bias,share,near_dist
    return 'other_map',bias,share,min(up_near,dn_near)


def process_stock(frame,slot,ticker):
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
    dates=ts.dt.date.to_numpy(); eod=np.r_[np.flatnonzero(dates[1:]!=dates[:-1]),n-1]
    rows=[]
    for t in eod:
        if t<110 or t+FIRST_TOUCH_BARS+RESOLUTION_BARS+PATH_BARS>=n: continue
        a=atr[t]
        if not np.isfinite(a) or a<=0: continue
        px=C[t]; fast=active_levels(p2,t)
        if len(fast)==0: continue
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
        if not nodes: continue
        fam,bias,share,nearest_dist=classify_family(nodes)
        fh=H[t+1:t+1+FIRST_TOUCH_BARS]; fl=L[t+1:t+1+FIRST_TOUCH_BARS]
        best=None
        for ni,nod in enumerate(nodes):
            lv=nod[0]
            hit=np.flatnonzero((fl<=lv)&(fh>=lv))
            if len(hit):
                key=(int(hit[0]),abs(nod[3]))
                if best is None or key<best[0]: best=(key,ni,t+1+int(hit[0]))
        if best is None: continue
        _,ni,touch=best; lv,members,div,dist,span=nodes[ni]; upper=lv>px
        hi=H[touch:min(n,touch+RESOLUTION_BARS+1)]; lo=L[touch:min(n,touch+RESOLUTION_BARS+1)]
        if upper:
            hold_hits=np.flatnonzero(lo<=lv-RESOLUTION_ATR*a)
            acc_hits=np.flatnonzero(hi>=lv+RESOLUTION_ATR*a)
        else:
            hold_hits=np.flatnonzero(hi>=lv+RESOLUTION_ATR*a)
            acc_hits=np.flatnonzero(lo<=lv-RESOLUTION_ATR*a)
        hd=int(hold_hits[0]) if len(hold_hits) else 99
        ad=int(acc_hits[0]) if len(acc_hits) else 99
        clean=0; resolution='unresolved'; ridx=-1
        if hd<ad:
            clean=1; resolution='hold'; ridx=touch+hd
        elif ad<hd:
            clean=1; resolution='accept'; ridx=touch+ad
        target_exists=0; path_hit=0; target_dist_atr=np.nan
        if clean:
            direction=(-1 if upper else 1) if resolution=='hold' else (1 if upper else -1)
            cand=[]
            for k,n2 in enumerate(nodes):
                if k==ni: continue
                lv2=n2[0]
                if direction>0 and lv2>lv: cand.append((lv2-lv,k))
                elif direction<0 and lv2<lv: cand.append((lv-lv2,k))
            if cand:
                delta,ki=min(cand); target=nodes[ki][0]; target_exists=1; target_dist_atr=delta/a
                hi2=H[ridx+1:min(n,ridx+1+PATH_BARS)]; lo2=L[ridx+1:min(n,ridx+1+PATH_BARS)]
                path_hit=int(np.any((lo2<=target)&(hi2>=target)))
        q=(int(members>=3)+int(div>=2))
        rows.append({
            'ticker':ticker,'snapshot_time':ts.iloc[t].isoformat(),'touch_time':ts.iloc[touch].isoformat(),
            'trust_family':fam,'bias_side':bias,'side_share':share,'nearest_dist_atr':nearest_dist,
            'interacted_side':'upper' if upper else 'lower','dist_atr':abs(dist),
            'members':members,'source_diversity':div,'cluster_span_atr':span,'Q_simple':q,
            'clean_resolution':clean,'resolution':resolution,'hold':int(resolution=='hold'),'accept':int(resolution=='accept'),
            'target_exists':target_exists,'target_dist_atr':target_dist_atr,'path_hit':path_hit,
        })
    return pd.DataFrame(rows)


def batch_source_map():
    inf=pd.read_csv(INFERENCE)
    out={}
    for _,r in inf.iterrows():
        b=int(r.batch)
        if bool(r.is_duplicate_payload): continue
        if b in out and str(r.file).startswith('old/'): continue
        prefix,fname=str(r.file).split('/',1)
        out[b]=(NEW_ZIP if prefix=='new' else OLD_ZIP,fname)
    return out


def process_batch(args):
    batch,zpath,fname,bmap=args
    usecols=['time']
    for _,rr in bmap.iterrows():
        ss=int(rr.slot)
        usecols += [f'S{ss:02d}_{x}' for x in ['O','H','L','C','V']]
    with zipfile.ZipFile(zpath) as z:
        frame=pd.read_csv(z.open(fname), usecols=usecols)
    outs=[]
    for _,r in bmap.iterrows():
        x=process_stock(frame,int(r.slot),str(r.ticker))
        if not x.empty:
            x['batch']=batch; x['slot']=int(r.slot); outs.append(x)
    return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()


def period(s):
    y=pd.to_datetime(s,utc=True).dt.year
    return np.select([y<=2016,y<=2021],['TRAIN_2013_2016','VALID_2017_2021'],default='HOLDOUT_2022PLUS')


def main():
    manifest=pd.read_csv(MANIFEST)
    src=batch_source_map()
    assert set(src)==set(range(1,43)),sorted(set(range(1,43))-set(src))
    args=[]
    for b in range(1,43):
        zpath,fname=src[b]; args.append((b,zpath,fname,manifest[manifest.batch==b].copy()))
    workers=min(10,max(2,cpu_count()-1))
    print('processing',len(args),'batches with',workers,'workers')
    with Pool(workers) as pool:
        pieces=list(pool.imap_unordered(process_batch,args))
    events=pd.concat([x for x in pieces if not x.empty],ignore_index=True)
    events.to_csv(OUT/'phase2_q_interaction_events.csv.gz',index=False,compression='gzip')
    print('events',len(events),'tickers',events.ticker.nunique())

if __name__=='__main__': main()
