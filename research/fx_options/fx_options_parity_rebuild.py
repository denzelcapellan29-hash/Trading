#!/usr/bin/env python3
"""Rebuild modern CME FX option surfaces from canonical monthly option DBNs.

Consumes collision-safe canonical files created by fx_options_dbn_ingest.py.
For each trade date/expiry it recovers the Black-76 forward and discount factor
from call-put parity, then computes implied volatilities, 10d/25d RR/BF and
30/60/90-day constant-maturity features.
"""
from __future__ import annotations
import argparse, json, math, struct, subprocess
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.special import ndtr
SCALE=1e9; NS_DAY=86_400_000_000_000; UNDEF_I64=np.iinfo(np.int64).max; UNDEF_U64=np.iinfo(np.uint64).max
ROOTS={"AUD":"ADU","CAD":"CAU","CHF":"CHU","EUR":"EUU","GBP":"GBU","JPY":"JPU","NZD":"6N"}
STAT_DTYPE=np.dtype({'names':['length','rtype','publisher_id','instrument_id','ts_event','ts_recv','ts_ref','price','quantity','sequence','ts_in_delta','stat_type','channel_id','update_action','stat_flags'],'formats':['u1','u1','<u2','<u4','<u8','<u8','<u8','<i8','<i8','<u4','<i4','<u2','<u2','u1','u1'],'offsets':[0,1,2,4,8,16,24,32,40,48,52,56,58,60,61],'itemsize':80})
DEF_DTYPE=np.dtype({'names':['length','rtype','publisher_id','instrument_id','ts_event','ts_recv','expiration','strike_price','underlying_id','leg_count','raw_symbol','asset','underlying','instrument_class','security_update_action'],'formats':['u1','u1','<u2','<u4','<u8','<u8','<u8','<i8','<u4','<u2','S71','S11','S21','S1','S1'],'offsets':[0,1,2,4,8,16,40,104,140,220,238,335,391,487,493],'itemsize':520})
def _skip(s,n):
    while n:
        b=s.read(min(n,8<<20))
        if not b: raise EOFError('truncated DBN')
        n-=len(b)
def stream_arr(path,dtype,recsize,chunk_records=700_000):
    p=subprocess.Popen(['zstdcat',str(path)],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL); h=p.stdout.read(8)
    if len(h)!=8 or h[:3]!=b'DBN': p.kill(); raise ValueError(f'bad DBN header: {path}')
    ml=struct.unpack('<I',h[4:8])[0]; _skip(p.stdout,ml); first=p.stdout.read(1)
    if not first: p.kill(); return
    actual=first[0]*4
    if actual!=recsize: p.kill(); raise ValueError((str(path),actual,recsize))
    pending=first+p.stdout.read(recsize-1)
    try:
        while pending:
            data=pending+p.stdout.read(recsize*chunk_records-len(pending)); n=len(data)//recsize; rem=data[n*recsize:]
            if n: yield np.frombuffer(data[:n*recsize],dtype=dtype,count=n).copy()
            more=p.stdout.read(recsize*chunk_records-len(rem)); pending=rem+more
            if not more and not rem: break
    finally: p.kill()
def _decode(a): return [bytes(v).split(b'\0',1)[0].decode('ascii','ignore') for v in a]
def parse_defs(path,root):
    frames=[]
    for a in stream_arr(path,DEF_DTYPE,520,120_000): frames.append(pd.DataFrame({'instrument_id':a['instrument_id'],'ts_recv':a['ts_recv'],'expiration':a['expiration'],'strike_price':a['strike_price'],'underlying_id':a['underlying_id'],'leg_count':a['leg_count'],'raw_symbol':_decode(a['raw_symbol']),'asset':_decode(a['asset']),'instrument_class':_decode(a['instrument_class'])}))
    d=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if d.empty: return d
    d=d[d.asset.eq(root)&d.instrument_class.isin(['C','P'])&d.leg_count.eq(0)&d.raw_symbol.ne('')].copy(); d['expiration_dt']=pd.to_datetime(d.expiration,unit='ns',utc=True,errors='coerce'); d['strike']=d.strike_price.astype(float)/SCALE; d.loc[d.strike_price.eq(UNDEF_I64),'strike']=np.nan; d=d.sort_values(['instrument_id','ts_recv']); cols=['raw_symbol','asset','instrument_class','strike','expiration','underlying_id']; prev=d.groupby('instrument_id',sort=False)[cols].shift(); return d[prev.ne(d[cols]).any(axis=1)].copy()
def _reduce_stats(a):
    if not len(a): return a
    day=(a['ts_ref']//NS_DAY).astype(np.uint32); order=np.lexsort((a['ts_recv'],a['stat_type'],day,a['instrument_id'])); a=a[order]; day=day[order]; same=(a['instrument_id'][:-1]==a['instrument_id'][1:])&(day[:-1]==day[1:])&(a['stat_type'][:-1]==a['stat_type'][1:]); keep=np.ones(len(a),bool); keep[:-1]=~same; return a[keep]
def parse_stats(path,types:Sequence[int]=(3,6,9)):
    parts=[]
    for a in stream_arr(path,STAT_DTYPE,80,800_000):
        m=np.isin(a['stat_type'],types)&(a['ts_ref']!=UNDEF_U64)
        if m.any(): parts.append(_reduce_stats(a[m]))
    if not parts: return pd.DataFrame()
    a=_reduce_stats(np.concatenate(parts)); a=a[a['update_action']!=2]; out=pd.DataFrame({'instrument_id':a['instrument_id'],'ts_recv':a['ts_recv'],'ts_ref':a['ts_ref'],'price':a['price'],'quantity':a['quantity'],'stat_type':a['stat_type'],'stat_flags':a['stat_flags']}); out['trade_date']=pd.to_datetime(out.ts_ref,unit='ns',utc=True,errors='coerce').dt.date; return out
def pit_attach(stats,defs):
    l=stats.copy(); l['join_ts']=l.ts_recv.astype(np.int64); r=defs[['instrument_id','ts_recv','raw_symbol','asset','instrument_class','strike','expiration','underlying_id','expiration_dt']].copy(); r['def_ts']=r.ts_recv.astype(np.int64); r=r.drop(columns='ts_recv'); return pd.merge_asof(l.sort_values(['join_ts','instrument_id']),r.sort_values(['def_ts','instrument_id']),left_on='join_ts',right_on='def_ts',by='instrument_id',direction='backward',allow_exact_matches=True)
def stat_panel(att):
    keys=['trade_date','instrument_id']; ident=att[keys+['raw_symbol','asset','instrument_class','strike','expiration','underlying_id','expiration_dt']].dropna(subset=['raw_symbol']).drop_duplicates(keys,keep='last'); s=att[att.stat_type.eq(3)][keys+['price','stat_flags','ts_recv']].rename(columns={'ts_recv':'settlement_ts'}).copy(); s['settlement_price']=s.price.where(s.price.ne(UNDEF_I64))/SCALE; s['settlement_final']=(s.stat_flags.astype(int)&1)!=0; s['settlement_actual']=(s.stat_flags.astype(int)&2)!=0; v=att[att.stat_type.eq(6)][keys+['quantity']].rename(columns={'quantity':'cleared_volume'}); o=att[att.stat_type.eq(9)][keys+['quantity']].rename(columns={'quantity':'open_interest'}); out=ident.merge(s.drop(columns='price'),on=keys,how='left').merge(v,on=keys,how='left').merge(o,on=keys,how='left');
    for c in ['cleared_volume','open_interest']: out.loc[out[c].eq(UNDEF_I64),c]=np.nan
    return out
def infer_parity(g):
    T=float(g.t_years.median()); p=g.pivot_table(index='strike',columns='option_type',values='settlement_price',aggfunc='last')
    if T<=0 or not {'C','P'}.issubset(p.columns): return np.nan,0.0,0,np.nan
    p=p.dropna(subset=['C','P']); K=p.index.to_numpy(float); y=(p['C']-p['P']).to_numpy(float); m=np.isfinite(K)&np.isfinite(y); K,y=K[m],y[m]
    if len(K)<3: return np.nan,0.0,len(K),np.nan
    keep=np.ones(len(K),bool); beta=np.array([np.nan,np.nan])
    for _ in range(4):
        X=np.c_[np.ones(keep.sum()),K[keep]]; beta=np.linalg.lstsq(X,y[keep],rcond=None)[0]; resid=y-(beta[0]+beta[1]*K); center=np.median(resid[keep]); mad=np.median(np.abs(resid[keep]-center))
        if not np.isfinite(mad) or mad<=1e-12: break
        new=np.abs(resid-center)<=max(4*1.4826*mad,1e-8)
        if new.sum()<3 or np.array_equal(new,keep): break
        keep=new
    D=-float(beta[1])
    if not np.isfinite(D) or not (0.85<D<1.15): return np.nan,0.0,int(keep.sum()),np.nan
    F=float(beta[0])/D
    if not np.isfinite(F) or F<=0: return np.nan,0.0,int(keep.sum()),np.nan
    r=-math.log(D)/T
    if not np.isfinite(r) or not (-0.10<r<0.35): r=0.0
    resid=y[keep]-(beta[0]+beta[1]*K[keep]); mad=float(np.median(np.abs(resid-np.median(resid)))) if len(resid) else np.nan; return F,float(r),int(keep.sum()),mad
def b76_price(F,K,T,sig,r,c):
    st=np.sqrt(T); d1=(np.log(F/K)+.5*sig*sig*T)/(sig*st); d2=d1-sig*st; D=np.exp(-r*T); return np.where(c,D*(F*ndtr(d1)-K*ndtr(d2)),D*(K*ndtr(-d2)-F*ndtr(-d1)))
def implied_vol(p,F,K,T,r,c,iters=46):
    p=np.asarray(p,float); F=np.asarray(F,float); K=np.asarray(K,float); T=np.asarray(T,float); r=np.asarray(r,float); c=np.asarray(c,bool); out=np.full(len(p),np.nan); ok=np.isfinite(p)&np.isfinite(F)&np.isfinite(K)&np.isfinite(T)&np.isfinite(r)&(p>0)&(F>0)&(K>0)&(T>0); idx=np.flatnonzero(ok)
    if not len(idx): return out
    pv,Fv,Kv,Tv,rv,cv=p[idx],F[idx],K[idx],T[idx],r[idx],c[idx]; D=np.exp(-rv*Tv); intrinsic=D*np.where(cv,np.maximum(Fv-Kv,0),np.maximum(Kv-Fv,0)); upper=D*np.where(cv,Fv,Kv); good=(pv>=intrinsic-1e-8)&(pv<=upper+1e-8); idx=idx[good]
    if not len(idx): return out
    pv,Fv,Kv,Tv,rv,cv=p[idx],F[idx],K[idx],T[idx],r[idx],c[idx]; lo=np.full(len(idx),1e-4); hi=np.full(len(idx),3.)
    for _ in range(iters): mid=(lo+hi)/2; model=b76_price(Fv,Kv,Tv,mid,rv,cv); lower=model<pv; lo=np.where(lower,mid,lo); hi=np.where(lower,hi,mid)
    out[idx]=(lo+hi)/2; return out
def b76_delta(F,K,T,sig,r,c):
    d1=(np.log(F/K)+.5*sig*sig*T)/(sig*np.sqrt(T)); D=np.exp(-r*T); return np.where(c,D*ndtr(d1),-D*ndtr(-d1))
def interp(x,y,target):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<2: return np.nan,False
    o=np.argsort(x); x,y=x[o],y[o]; ux=np.unique(x)
    if len(ux)<2: return np.nan,False
    if len(ux)!=len(x): y=np.array([np.median(y[x==v]) for v in ux]); x=ux
    if target<x[0] or target>x[-1]: return np.nan,False
    return float(np.interp(target,x,y)),True
def build_points(defp,statp,root):
    defs=parse_defs(defp,root); stats=parse_stats(statp); p=stat_panel(pit_attach(stats,defs)); p=p[p.instrument_class.isin(['C','P'])&p.settlement_price.notna()].dropna(subset=['strike','expiration_dt']).copy(); p['option_type']=p.instrument_class; p['trade_ts']=pd.to_datetime(p.trade_date.astype(str),utc=True); p['dte_days']=(p.expiration_dt-p.trade_ts).dt.total_seconds()/86400.; p['t_years']=p.dte_days/365.25; p=p[(p.dte_days>=2)&(p.dte_days<=200)&(p.settlement_price>0)].copy(); rows=[]
    for key,g in p.groupby(['trade_date','expiration_dt','underlying_id'],sort=False): F,r,n,mad=infer_parity(g); rows.append((*key,F,r,n,mad))
    parity=pd.DataFrame(rows,columns=['trade_date','expiration_dt','underlying_id','underlying_settlement','rate','rate_pairs','rate_mad']); p=p.merge(parity,on=['trade_date','expiration_dt','underlying_id'],how='left').dropna(subset=['underlying_settlement']); p['iv']=implied_vol(p.settlement_price,p.underlying_settlement,p.strike,p.t_years,p.rate,p.option_type.eq('C')); p=p[np.isfinite(p.iv)&(p.iv>=.005)&(p.iv<=2.)].copy(); p['delta']=b76_delta(p.underlying_settlement.to_numpy(float),p.strike.to_numpy(float),p.t_years.to_numpy(float),p.iv.to_numpy(float),p.rate.to_numpy(float),p.option_type.eq('C').to_numpy(bool)); return p
def build_surfaces(pts):
    rows=[]
    for key,g in pts.groupby(['trade_date','expiration_dt','underlying_id'],sort=True):
        F=float(g.underlying_settlement.median()); K=g.strike.to_numpy(float); iv=g.iv.to_numpy(float); ad=np.abs(g.delta.to_numpy(float)); typ=g.option_type.to_numpy(); put=(typ=='P')&(K<=F); call=(typ=='C')&(K>=F); otm=put|call; atm,atm_ok=interp(np.log(K[otm]/F),iv[otm],0.); row={'trade_date':key[0],'expiration':key[1],'underlying_id':key[2],'dte_days':float(g.dte_days.median()),'forward':F,'rate':float(g.rate.median()),'rate_pairs':int(g.rate_pairs.median()),'rate_mad':float(g.rate_mad.median()),'atm_iv':atm,'atm_ok':bool(atm_ok),'n_valid':len(g),'settlement_final_share':float(g.settlement_final.eq(True).mean()),'settlement_actual_share':float(g.settlement_actual.eq(True).mean())}
        for target,tag in [(.25,'25'),(.10,'10')]: ci,cok=interp(ad[call],iv[call],target); pi,pok=interp(ad[put],iv[put],target); row[f'c{tag}']=ci; row[f'p{tag}']=pi; row[f'rr{tag}']=ci-pi if np.isfinite(ci) and np.isfinite(pi) else np.nan; row[f'bf{tag}']=.5*(ci+pi)-atm if np.isfinite(ci) and np.isfinite(pi) and np.isfinite(atm) else np.nan; row[f'c{tag}_ok']=bool(cok); row[f'p{tag}_ok']=bool(pok)
        core=otm&(ad>=.10)&(ad<=.90); mid=otm&(ad>=.20)&(ad<=.80); row['n_core']=int(core.sum()); row['n_mid']=int(mid.sum()); row['research_grade']=bool(atm_ok and row['n_core']>=6 and row['n_mid']>=4 and row['settlement_final_share']>=.5 and row['settlement_actual_share']>=.5); rows.append(row)
    return pd.DataFrame(rows)
def cm_value(g,col,target,max_span=75):
    z=g[['dte_days',col]].dropna().sort_values('dte_days')
    if z.empty: return np.nan
    lo=z[z.dte_days<=target]; hi=z[z.dte_days>=target]
    if lo.empty or hi.empty: return np.nan
    a,b=lo.iloc[-1],hi.iloc[0]; d0,d1=float(a.dte_days),float(b.dte_days); v0,v1=float(a[col]),float(b[col])
    if d0==d1: return v0
    if d1-d0>max_span: return np.nan
    if col=='atm_iv': t0,t1,tt=d0/365.25,d1/365.25,target/365.25; w0,w1=v0*v0*t0,v1*v1*t1; return math.sqrt(max((w0+(w1-w0)*(tt-t0)/(t1-t0))/tt,0))
    return v0+(v1-v0)*(target-d0)/(d1-d0)
def build_cm(surf,currency,root):
    rg=surf[surf.research_grade].copy(); rows=[]
    for dt,g in rg.groupby('trade_date',sort=True):
        r={'trade_date':dt,'currency':currency,'source_root':root,'forward_method':'PARITY_IMPLIED'}
        for T in (30,60,90):
            for col in ('atm_iv','rr25','bf25','rr10','bf10'): r[f'{T}d_{col}']=cm_value(g,col,T)
        rows.append(r)
    out=pd.DataFrame(rows)
    if out.empty: return out
    out['atm_30m60']=out['30d_atm_iv']-out['60d_atm_iv']; out['atm_60m90']=out['60d_atm_iv']-out['90d_atm_iv']; out['atm_30m90']=out['30d_atm_iv']-out['90d_atm_iv']; out['rr25_30m90']=out['30d_rr25']-out['90d_rr25']; out['rr10_30m90']=out['30d_rr10']-out['90d_rr10']
    for T in (30,60,90): out[f'{T}d_core_complete']=out[[f'{T}d_atm_iv',f'{T}d_rr25',f'{T}d_rr10']].notna().all(axis=1)
    return out.sort_values('trade_date')
def find_year_file(raw_root,currency,root,year,schema):
    p=raw_root/currency/schema/(f'{currency}_{root}_20260101_20260818_{schema}.dbn.zst' if year==2026 else f'{currency}_{root}_{year}0101_{year}1231_{schema}.dbn.zst')
    if not p.exists(): raise FileNotFoundError(p)
    return p
def process_year(raw_root,out_root,currency,year):
    root=ROOTS[currency]; pts=build_points(find_year_file(raw_root,currency,root,year,'definition'),find_year_file(raw_root,currency,root,year,'statistics'),root); surf=build_surfaces(pts); cm=build_cm(surf,currency,root); out=out_root/currency; out.mkdir(parents=True,exist_ok=True); surf.to_csv(out/f'{currency}_{year}_surfaces.csv.gz',index=False,compression='gzip'); cm.to_csv(out/f'{currency}_{year}_cm.csv',index=False); qa={'currency':currency,'year':year,'forward_method':'PARITY_IMPLIED','iv_points':len(pts),'expiry_surfaces':len(surf),'research_grade_surfaces':int(surf.research_grade.sum()) if len(surf) else 0,'cm_days':len(cm),'start':str(cm.trade_date.min()) if len(cm) else None,'end':str(cm.trade_date.max()) if len(cm) else None}
    for T in (30,60,90):
        for col in ('atm_iv','rr25','bf25','rr10','bf10'): qa[f'coverage_{T}d_{col}']=float(cm[f'{T}d_{col}'].notna().mean()) if len(cm) else np.nan
    (out/f'{currency}_{year}_QA.json').write_text(json.dumps(qa,indent=2,default=str)); print(json.dumps(qa,default=str),flush=True)
def rebuild_aggregate(out_root,currency):
    out=out_root/currency; cms=[pd.read_csv(p) for p in sorted(out.glob(f'{currency}_20??_cm.csv'))]
    if cms: pd.concat(cms,ignore_index=True).sort_values('trade_date').drop_duplicates(['trade_date'],keep='last').to_csv(out/f'{currency}_2018_2026_cm.csv',index=False)
    qas=[json.loads(p.read_text()) for p in sorted(out.glob(f'{currency}_20??_QA.json'))]
    if qas: pd.DataFrame(qas).sort_values('year').to_csv(out/f'{currency}_2018_2026_QA.csv',index=False)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw-root',type=Path,required=True); ap.add_argument('--out-root',type=Path,required=True); ap.add_argument('--currency',choices=sorted(ROOTS),required=True); ap.add_argument('--years',required=True); args=ap.parse_args();
    for y in [int(x) for x in args.years.split(',')]: process_year(args.raw_root,args.out_root,args.currency,y)
    rebuild_aggregate(args.out_root,args.currency)
if __name__=='__main__': main()
