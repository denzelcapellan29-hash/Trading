#!/usr/bin/env python3
"""General FX 'tribes' research pass using frozen COT-PCA state lineage and frozen FX portfolios.

This is research telemetry, not a strategy retune. It reconstructs:
- participant-level theme direction for Leveraged Funds, Asset Managers and Dealers;
- cross-tribe theme geometry;
- forward G7/USD spot outcomes;
- following-week FAST and frozen 65FAST/35ALT behavior;
- ALT sleeve trade behavior.

Important source limitation: the COT-PCA state panel is recovered from the frozen FAST overlay handoff,
so it contains 642 unique weekly participant states rather than every calendar CFTC week. The study is
therefore exploratory until rebuilt from full raw CFTC TFF history including Other Reportables.
"""
from __future__ import annotations
import argparse, io, json, math, re, zipfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

CCY=['AUD','CAD','CHF','EUR','GBP','JPY','NZD']
PAIR_MAP={'AUD':'AUDUSD','CAD':'USDCAD','CHF':'USDCHF','EUR':'EURUSD','GBP':'GBPUSD','JPY':'USDJPY','NZD':'NZDUSD'}
INVERT={'AUD':False,'CAD':True,'CHF':True,'EUR':False,'GBP':False,'JPY':True,'NZD':False}
PARTS=['leveraged','asset_mgr','dealer']
SEED=20260912


def load_pair_2h(zpath,pair):
    with zipfile.ZipFile(zpath) as z:
        name=next(n for n in z.namelist() if re.search(fr'OANDA_{pair},',n))
        d=pd.read_csv(z.open(name))
    parts=[]
    for s in range(1,5):
        pr=f'EXPORT S{s} '
        q=d[[pr+'Start Timestamp',pr+'Close',pr+'Complete Flag']].copy()
        q.columns=['start','close','complete']
        q=q[pd.to_numeric(q.complete,errors='coerce').fillna(0).eq(1)].copy()
        q['time']=pd.to_datetime(pd.to_numeric(q.start,errors='coerce'),unit='ms',utc=True)
        q['close']=pd.to_numeric(q.close,errors='coerce')
        parts.append(q[['time','close']])
    return pd.concat(parts,ignore_index=True).dropna().sort_values('time').drop_duplicates('time')


def weekly_spot(zpath):
    out=[]
    for c in CCY:
        x=load_pair_2h(zpath,PAIR_MAP[c]);f=x[x.time.dt.dayofweek.eq(4)].copy();f['friday']=f.time.dt.normalize()
        s=f.groupby('friday').tail(1).set_index('friday').close.sort_index()
        if INVERT[c]:s=1/s
        out.append(s.rename(c))
    return pd.concat(out,axis=1).sort_index().dropna(how='any')


def load_states(overlay_zip):
    with zipfile.ZipFile(overlay_zip) as z:
        n=next(n for n in z.namelist() if n.endswith('fast_trades_with_cot_pca_context.csv'));raw=pd.read_csv(z.open(n))
    frames=[]
    for p in PARTS:
        cols=[f'{p}_friday',f'{p}_report_date',f'{p}_cot_score',f'{p}_cot_share',f'{p}_cot_activity',f'{p}_build4',f'{p}_activitychg4',f'{p}_match_cos',f'{p}_crowded_80_70',f'{p}_crowded_80_80']+[f'{p}_load_{c}' for c in CCY]
        q=raw[cols].dropna(subset=[f'{p}_friday']).drop_duplicates().copy();q['friday']=pd.to_datetime(q[f'{p}_friday'],utc=True).dt.normalize();q=q.drop(columns=f'{p}_friday')
        if q.friday.duplicated().any(): raise RuntimeError(f'duplicate {p} state Friday')
        frames.append(q.set_index('friday'))
    return pd.concat(frames,axis=1,join='inner').reset_index().sort_values('friday')


def load_ports(fast_zip):
    with zipfile.ZipFile(fast_zip) as z:
        f=pd.read_csv(z.open('FX_FAST_DATA_PACKAGE_2026-08-28/data/fx_fast_weekly_returns.csv'))
        c=pd.read_csv(z.open('FX_FAST_DATA_PACKAGE_2026-08-28/data/fx_validated_65fast_35alt_weekly_returns.csv'))
    f['date']=pd.to_datetime(f.date,utc=True);c['date']=pd.to_datetime(c.date,utc=True)
    return f[['date','net_portfolio_return']].rename(columns={'net_portfolio_return':'FAST'}).merge(c[['date','ALT_RM','FX_65FAST_35ALT']],on='date',how='outer').sort_values('date')


def unit(v):
    v=np.asarray(v,float);n=np.linalg.norm(v);return v/n if np.isfinite(n) and n>0 else np.full_like(v,np.nan)

def l1(v):
    v=np.asarray(v,float);n=np.abs(v).sum();return v/n if np.isfinite(n) and n>0 else np.full_like(v,np.nan)

def cos(a,b):
    a=unit(a);b=unit(b);return float(a@b) if np.isfinite(a).all() and np.isfinite(b).all() else np.nan

def vec(r,p):
    return np.sign(float(r[f'{p}_cot_score']))*np.array([r[f'{p}_load_{c}'] for c in CCY],float)


def enrich(states,spot,ports):
    d=states.merge(spot.reset_index(),on='friday',how='inner',validate='one_to_one')
    for h in [1,4,8]:
        q=spot.shift(-h)/spot-1;q.columns=[f'fwd{h}_{c}' for c in CCY];d=d.merge(q.reset_index(),on='friday',how='left',validate='one_to_one')
    for p in PARTS:
        for h in [1,4,8]:
            vals=[]
            for _,r in d.iterrows():
                w=l1(vec(r,p));rr=np.array([r[f'fwd{h}_{c}'] for c in CCY],float);vals.append(float(w@rr) if np.isfinite(w).all() and np.isfinite(rr).all() else np.nan)
            d[f'{p}_same_dir_fwd{h}']=vals
    geo={'cos_lf_am':[],'cos_lf_dealer':[],'cos_am_dealer':[],'cos_dealer_client':[]}
    for h in [1,4,8]:geo[f'client_consensus_fwd{h}']=[];geo[f'ccy_dispersion_fwd{h}']=[];geo[f'ccy_absavg_fwd{h}']=[]
    for _,r in d.iterrows():
        lf,am,de=vec(r,'leveraged'),vec(r,'asset_mgr'),vec(r,'dealer');client=unit(lf)+unit(am)
        geo['cos_lf_am'].append(cos(lf,am));geo['cos_lf_dealer'].append(cos(lf,de));geo['cos_am_dealer'].append(cos(am,de));geo['cos_dealer_client'].append(cos(de,client));cw=l1(client)
        for h in [1,4,8]:
            rr=np.array([r[f'fwd{h}_{c}'] for c in CCY],float)
            geo[f'client_consensus_fwd{h}'].append(float(cw@rr) if np.isfinite(cw).all() and np.isfinite(rr).all() else np.nan)
            geo[f'ccy_dispersion_fwd{h}'].append(float(np.std(rr,ddof=1)) if np.isfinite(rr).all() else np.nan)
            geo[f'ccy_absavg_fwd{h}'].append(float(np.mean(np.abs(rr))) if np.isfinite(rr).all() else np.nan)
    for k,v in geo.items():d[k]=v
    d['lf_crowded']=d.leveraged_crowded_80_70.eq(True);d['am_highmatch']=d.asset_mgr_match_cos.ge(.85);d['dealer_highmatch']=d.dealer_match_cos.ge(.85)
    d['lf_am_aligned']=d.cos_lf_am.ge(.5);d['lf_am_opposed']=d.cos_lf_am.le(-.5);d['buy_side_conflict']=d.lf_am_opposed
    d['dealer_opposes_clients']=d.cos_dealer_client.le(-.5);d['client_consensus_dealer_absorption']=d.lf_am_aligned&d.dealer_opposes_clients;d['lf_crowded_am_highmatch']=d.lf_crowded&d.am_highmatch
    p=ports.copy();p['friday']=p.date.dt.normalize()-pd.Timedelta(days=7);d=d.merge(p.drop(columns='date'),on='friday',how='left')
    return d


def state_summary(d):
    rows=[]
    for p in PARTS:
        defs={'ALL':pd.Series(True,index=d.index),'CROWDED_80_70':d[f'{p}_crowded_80_70'].eq(True),'HIGH_PRICE_MATCH_85':d[f'{p}_match_cos'].ge(.85),'BUILD4_POS':d[f'{p}_build4'].gt(0),'BUILD4_NEG':d[f'{p}_build4'].lt(0)}
        for nm,m in defs.items():
            z=d[m];r={'participant':p,'state':nm,'n_weeks':len(z),'mean_abs_score':z[f'{p}_cot_score'].abs().mean(),'mean_share':z[f'{p}_cot_share'].mean(),'mean_match_cos':z[f'{p}_match_cos'].mean()}
            for h in [1,4,8]:
                x=z[f'{p}_same_dir_fwd{h}'];r[f'same_dir_ann_h{h}']=x.mean()*52/h;r[f'same_dir_hit_h{h}']=(x>0).mean()
            rows.append(r)
    return pd.DataFrame(rows)


def geometry_outcomes(d):
    defs={'LF_AM_ALIGNED':d.lf_am_aligned,'LF_AM_OPPOSED':d.lf_am_opposed,'DEALER_OPPOSES_CLIENTS':d.dealer_opposes_clients,'CLIENT_CONSENSUS_DEALER_ABSORPTION':d.client_consensus_dealer_absorption,'BUY_SIDE_CONFLICT':d.buy_side_conflict,'LF_CROWDED':d.lf_crowded,'AM_HIGHMATCH':d.am_highmatch,'LF_CROWDED_AND_AM_HIGHMATCH':d.lf_crowded_am_highmatch}
    rows=[]
    for nm,m in defs.items():
        for g,ms in [('STATE',m),('OTHER',~m)]:
            z=d[ms];r={'state':nm,'group':g,'n_weeks':len(z),'mean_cos_lf_am':z.cos_lf_am.mean(),'mean_cos_dealer_client':z.cos_dealer_client.mean()}
            for h in [1,4,8]:
                r[f'client_consensus_ann_h{h}']=z[f'client_consensus_fwd{h}'].mean()*52/h;r[f'ccy_dispersion_ann_h{h}']=z[f'ccy_dispersion_fwd{h}'].mean()*math.sqrt(52/h);r[f'ccy_absavg_h{h}']=z[f'ccy_absavg_fwd{h}'].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def tails(d):
    defs={'LF_CROWDED':d.lf_crowded,'AM_HIGHMATCH':d.am_highmatch,'LF_AM_ALIGNED':d.lf_am_aligned,'BUY_SIDE_CONFLICT':d.buy_side_conflict,'DEALER_OPPOSES_CLIENTS':d.dealer_opposes_clients,'CLIENT_CONSENSUS_DEALER_ABSORPTION':d.client_consensus_dealer_absorption,'LF_CROWDED_AND_AM_HIGHMATCH':d.lf_crowded_am_highmatch}
    rows=[]
    for nm,m in defs.items():
        for port in ['FAST','ALT_RM','FX_65FAST_35ALT']:
            for g,ms in [('STATE',m),('OTHER',~m)]:
                x=d.loc[ms,port].dropna();q=x.quantile(.05) if len(x) else np.nan;es=x[x<=q].mean() if len(x) else np.nan
                rows.append({'state':nm,'portfolio':port,'group':g,'n':len(x),'ann':x.mean()*52 if len(x) else np.nan,'loss':(x<0).mean() if len(x) else np.nan,'q05':q,'es05':es,'vol':x.std(ddof=1)*np.sqrt(52) if len(x)>1 else np.nan})
    return pd.DataFrame(rows)


def chronology(d):
    defs={'LF_CROWDED':d.lf_crowded,'AM_HIGHMATCH':d.am_highmatch,'BUY_SIDE_CONFLICT':d.buy_side_conflict,'DEALER_OPPOSES_CLIENTS':d.dealer_opposes_clients}
    periods=[('2011-2016',2011,2016),('2017-2021',2017,2021),('2022+',2022,9999),('2011+',2011,9999)];rows=[]
    for pn,a,b in periods:
        pm=d.friday.dt.year.between(a,b)
        for sn,sm in defs.items():
            for g,gm in [('STATE',sm),('OTHER',~sm)]:
                z=d[pm&gm];rows.append({'period':pn,'state':sn,'group':g,'n_weeks':len(z),'lf_theme_ann_1w':z.leveraged_same_dir_fwd1.mean()*52,'am_theme_ann_1w':z.asset_mgr_same_dir_fwd1.mean()*52,'dealer_theme_ann_1w':z.dealer_same_dir_fwd1.mean()*52,'client_consensus_ann_1w':z.client_consensus_fwd1.mean()*52,'FAST_ann_nextweek':z.FAST.mean()*52,'ALT_RM_ann_nextweek':z.ALT_RM.mean()*52,'FX65_35_ann_nextweek':z.FX_65FAST_35ALT.mean()*52})
    return pd.DataFrame(rows)


def block_indices(n,b,reps,rng):
    k=int(np.ceil(n/b));s=rng.integers(0,n,size=(reps,k));o=np.arange(b);return ((s[:,:,None]+o)%n).reshape(reps,-1)[:,:n]

def bootstrap(d,state,outcome,b,reps):
    q=d[[state,outcome]].dropna().reset_index(drop=True);s=q[state].to_numpy(bool);y=q[outcome].to_numpy(float);n=len(q);rng=np.random.default_rng(SEED+sum(map(ord,state+outcome))+b);ix=block_indices(n,b,reps,rng);ss=s[ix];yy=y[ix];ns=ss.sum(1);no=n-ns;vld=(ns>0)&(no>0);delta=np.full(reps,np.nan);delta[vld]=np.where(ss,yy,0).sum(1)[vld]/ns[vld]-np.where(~ss,yy,0).sum(1)[vld]/no[vld];v=delta[np.isfinite(delta)];obs=y[s].mean()-y[~s].mean();h=4 if 'fwd4' in outcome else 8 if 'fwd8' in outcome else 1;scale=52/h
    return {'state':state,'outcome':outcome,'block_weeks':b,'reps':len(v),'observed_delta':obs,'observed_delta_ann':obs*scale,'p_gt0':(v>0).mean(),'p_lt0':(v<0).mean(),'ci90_low_ann':np.quantile(v,.05)*scale,'ci90_high_ann':np.quantile(v,.95)*scale}


def alt_trade_context(alt_zip,d):
    with zipfile.ZipFile(alt_zip) as z:b=z.read('reference/FX_ALT_PORTFOLIOS_NODE_CHAIN_AND_ROUTER_FINALIZATION_V1_2026-08-17.zip')
    with zipfile.ZipFile(io.BytesIO(b)) as q:
        n=next(n for n in q.namelist() if n.endswith('data/router_selected_trades.csv'));t=pd.read_csv(q.open(n))
    t['entry_time']=pd.to_datetime(t.entry_time,utc=True);t['net_ret_5pip']=t.trade_dir*(t['exit']/t['entry']-1)-5*t.pip_size/t['entry']
    s=d[['friday','lf_crowded','am_highmatch','buy_side_conflict','dealer_opposes_clients','client_consensus_dealer_absorption','lf_crowded_am_highmatch','lf_am_aligned']].copy();s['avail']=s.friday+pd.Timedelta(hours=21)
    t=pd.merge_asof(t.sort_values('entry_time'),s.sort_values('avail'),left_on='entry_time',right_on='avail',direction='backward');t['state_age_days']=(t.entry_time-t.avail).dt.total_seconds()/86400;t=t[t.state_age_days<=14].copy()
    defs=['lf_crowded','am_highmatch','buy_side_conflict','dealer_opposes_clients','client_consensus_dealer_absorption','lf_crowded_am_highmatch','lf_am_aligned'];rows=[]
    for sl in ['corridor','rotation','compression','ALL']:
        z=t if sl=='ALL' else t[t.sleeve==sl]
        for st in defs:
            m=z[st].eq(True)
            for g,ms in [('STATE',m),('OTHER',~m)]:
                x=z.loc[ms,'net_ret_5pip'];rows.append({'sleeve':sl,'state':st,'group':g,'n':len(x),'mean_net_bps':x.mean()*1e4 if len(x) else np.nan,'hit_rate':(x>0).mean() if len(x) else np.nan,'median_bps':x.median()*1e4 if len(x) else np.nan})
    return pd.DataFrame(rows),len(t)


def persistence(d):
    defs={'LF_CROWDED':d.lf_crowded,'AM_HIGHMATCH':d.am_highmatch,'LF_AM_ALIGNED':d.lf_am_aligned,'BUY_SIDE_CONFLICT':d.buy_side_conflict,'DEALER_OPPOSES_CLIENTS':d.dealer_opposes_clients,'CLIENT_CONSENSUS_DEALER_ABSORPTION':d.client_consensus_dealer_absorption,'LF_CROWDED_AND_AM_HIGHMATCH':d.lf_crowded_am_highmatch};rows=[]
    for sn,m in defs.items():
        a=m.to_numpy(bool);starts=np.where(a & np.r_[True,~a[:-1]])[0];lens=[]
        for st in starts:
            j=st
            while j+1<len(a) and a[j+1] and (d.friday.iloc[j+1]-d.friday.iloc[j]).days<=8:j+=1
            lens.append(j-st+1)
        rows.append({'state':sn,'episodes':len(lens),'state_weeks':int(a.sum()),'mean_episode_weeks':np.mean(lens) if lens else np.nan,'median_episode_weeks':np.median(lens) if lens else np.nan,'max_episode_weeks':max(lens) if lens else np.nan})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--overlay-zip',required=True);ap.add_argument('--spot-zip',required=True);ap.add_argument('--fast-data-zip',required=True);ap.add_argument('--alt-production-zip',required=True);ap.add_argument('--out',required=True);ap.add_argument('--bootstrap-reps',type=int,default=5000);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=enrich(load_states(Path(a.overlay_zip)),weekly_spot(Path(a.spot_zip)),load_ports(Path(a.fast_data_zip)));d.to_csv(out/'fx_tribes_weekly_state_panel.csv',index=False)
    state_summary(d).to_csv(out/'fx_tribes_participant_state_summary.csv',index=False);geometry_outcomes(d).to_csv(out/'fx_tribes_geometry_forward_outcomes.csv',index=False);tails(d).to_csv(out/'fx_tribes_portfolio_tail_context.csv',index=False);chronology(d).to_csv(out/'fx_tribes_chronology.csv',index=False);persistence(d).to_csv(out/'fx_tribes_state_persistence.csv',index=False)
    alt,nalt=alt_trade_context(Path(a.alt_production_zip),d);alt.to_csv(out/'fx_tribes_alt_sleeve_trade_context.csv',index=False)
    specs=[('lf_crowded','leveraged_same_dir_fwd1'),('lf_crowded','leveraged_same_dir_fwd4'),('am_highmatch','asset_mgr_same_dir_fwd4'),('am_highmatch','asset_mgr_same_dir_fwd8'),('buy_side_conflict','FAST'),('buy_side_conflict','ALT_RM'),('buy_side_conflict','FX_65FAST_35ALT'),('dealer_opposes_clients','FAST'),('dealer_opposes_clients','FX_65FAST_35ALT'),('am_highmatch','FAST'),('am_highmatch','FX_65FAST_35ALT'),('lf_crowded_am_highmatch','FAST'),('lf_crowded_am_highmatch','FX_65FAST_35ALT')]
    br=[]
    for s,o in specs:
        for b in [4,13,26]:br.append(bootstrap(d,s,o,b,a.bootstrap_reps))
    pd.DataFrame(br).to_csv(out/'fx_tribes_block_bootstrap.csv',index=False)
    inc=[]
    for y,g in d.groupby(d.friday.dt.year):inc.append({'year':int(y),'observed_weeks':len(g),'lf_crowded':int(g.lf_crowded.sum()),'am_highmatch':int(g.am_highmatch.sum()),'lf_am_aligned':int(g.lf_am_aligned.sum()),'buy_side_conflict':int(g.buy_side_conflict.sum()),'dealer_opposes_clients':int(g.dealer_opposes_clients.sum()),'client_consensus_dealer_absorption':int(g.client_consensus_dealer_absorption.sum())})
    pd.DataFrame(inc).to_csv(out/'fx_tribes_state_incidence_by_year.csv',index=False)
    # Summary values for report generation / audit
    meta={'analysis_date':'2026-09-12','aligned_weekly_rows':len(d),'first_state':str(d.friday.min()),'last_state':str(d.friday.max()),'matched_alt_trades':nalt,'currency_universe':CCY,'participants':PARTS,'timing':'Tuesday TFF state available Friday; market outcomes from Friday close; strategy outcomes following week','fixed_thresholds':{'LF crowded':'pre-existing rolling PC share q80 + abs score q70','AM price-PC highmatch':0.85,'geometry aligned':0.5,'geometry opposed':-0.5},'limitations':['TFF state rows recovered from FAST overlay lineage; not every calendar CFTC week','Other Reportables absent','TFF classifies traders by predominant business type, not each trade motive','spot/theme results exclude forwards/carry']}
    (out/'fx_tribes_run_metadata.json').write_text(json.dumps(meta,indent=2))
    # manifest
    rows=[]
    for p in sorted(out.iterdir()):
        if p.is_file():rows.append({'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    pd.DataFrame(rows).to_csv(out/'SHA256_MANIFEST.csv',index=False)
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
