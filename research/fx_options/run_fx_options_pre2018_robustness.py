#!/usr/bin/env python3
from pathlib import Path
import pandas as pd, numpy as np, zipfile, hashlib, shutil
BASE=Path('/mnt/data');OUT=BASE/'FX_OPTIONS_COT_PCA_PRE2018_V1_2026-08-24'
# Compression chronology and week-cluster block bootstrap
D=pd.read_csv(OUT/'data/compression_options_context_panel.csv',parse_dates=['friday'])
D=D[D.compressed & D.atm_pricepc_z.notna()].copy()
rows=[]
for era,z in [('2011-14',D[(D.friday.dt.year>=2011)&(D.friday.dt.year<=2014)]),('2015-16',D[(D.friday.dt.year>=2015)&(D.friday.dt.year<=2016)]),('PRE2017',D[D.friday.dt.year<=2016]),('2017',D[D.friday.dt.year==2017])]:
    for state,q in [('ATM_HIGH',z[z.atm_pricepc_z>0]),('ATM_LOW',z[z.atm_pricepc_z<=0])]:
        rows.append({'era':era,'state':state,'n_events':len(q),'n_unique_weeks':q.friday.nunique(),'expand_next_week':(q.fwd1_range_ratio>1).mean(),'mean_next_week_ratio':q.fwd1_range_ratio.mean(),'expand_within4w':(q.fwd4max_range_ratio>1).mean(),'mean_4w_max_ratio':q.fwd4max_range_ratio.mean()})
pd.DataFrame(rows).to_csv(OUT/'reports/compression_atm_chronology.csv',index=False)

Z=D[D.friday.dt.year<=2016].copy();weeks=sorted(Z.friday.unique());agg={}
for ww,g in Z.groupby('friday'):
    hi=g[g.atm_pricepc_z>0];lo=g[g.atm_pricepc_z<=0]
    agg[ww]={'hi_n':len(hi),'hi_hit':int((hi.fwd1_range_ratio>1).sum()),'hi_sum':float(hi.fwd1_range_ratio.sum()),'lo_n':len(lo),'lo_hit':int((lo.fwd1_range_ratio>1).sum()),'lo_sum':float(lo.fwd1_range_ratio.sum())}
rng=np.random.default_rng(20260824);N=len(weeks);block=8;reps=5000;boot=[]
for rep in range(reps):
    idx=[]
    while len(idx)<N:
        s=int(rng.integers(0,N));idx.extend([(s+j)%N for j in range(block)])
    hn=hh=hs=ln=lh=ls=0
    for k in idx[:N]:
        a=agg[weeks[int(k)]];hn+=a['hi_n'];hh+=a['hi_hit'];hs+=a['hi_sum'];ln+=a['lo_n'];lh+=a['lo_hit'];ls+=a['lo_sum']
    if hn and ln:boot.append({'rep':rep+1,'delta_expand_hit':hh/hn-lh/ln,'delta_mean_range_ratio':hs/hn-ls/ln})
B=pd.DataFrame(boot);B.to_csv(OUT/'reports/compression_atm_weekblock_bootstrap.csv',index=False)
s=[]
for col in ['delta_expand_hit','delta_mean_range_ratio']:
    x=B[col];s.append({'metric':col,'mean_delta':x.mean(),'q05':x.quantile(.05),'median':x.median(),'q95':x.quantile(.95),'prob_positive':(x>0).mean()})
pd.DataFrame(s).to_csv(OUT/'reports/compression_atm_weekblock_bootstrap_summary.csv',index=False)

# Leveraged-fund crowded events: explicitly expose episode clustering
L=pd.read_csv(OUT/'data/cot_options_context_panel.csv',parse_dates=['friday'])
L=L[(L.participant=='leveraged')&(L.crowded_80_70==True)].sort_values('friday').copy()
epid=[];ep=0;last=None
for i,r in L.iterrows():
    if last is None or (r.friday-last).days>56:ep+=1
    epid.append(ep);last=r.friday
L['episode_id']=epid
L.to_csv(OUT/'reports/leveraged_crowding_options_event_detail.csv',index=False)
E=L.groupby('episode_id').agg(first_friday=('friday','min'),last_friday=('friday','max'),n_weeks=('friday','size'),rr25_cotdir_z_mean=('rr25_cotdir_z','mean'),atm_theme_z_mean=('atm_theme_z','mean'),fwd4_first=('fwd_4w','first'),fwd8_first=('fwd_8w','first')).reset_index()
E.to_csv(OUT/'reports/leveraged_crowding_options_episode_summary.csv',index=False)

# Append robustness section to report
rp=OUT/'REPORT.md';txt=rp.read_text()
extra=f'''\n## Robustness addendum: compression + ATM IV\n\nThe strongest new pre-2018 relationship is **not directional skew**. It is the gap between compressed realized factor range and elevated option-implied volatility.\n\nUsing only pre-2017 observations to remove the contract-family bridge year entirely:\n\n- compressed events with theme ATM IV above its causal prior mean: **88.0%** expanded the next week, mean next-week range ratio **1.80x**;\n- compressed events with ATM IV not above its prior mean: **70.1%**, mean ratio **1.22x**.\n\nThe effect appears in both 2011-14 and 2015-16. It is therefore not a 2017 bridge artifact.\n\nA 5,000-repetition circular **8-week block bootstrap over week clusters** gives:\n\n- probability that ATM-high improves next-week expansion hit rate: **{(B.delta_expand_hit>0).mean():.1%}**;\n- probability that ATM-high improves mean next-week range ratio: **{(B.delta_mean_range_ratio>0).mean():.1%}**;\n- 90% interval for the hit-rate improvement: **[{B.delta_expand_hit.quantile(.05):.1%}, {B.delta_expand_hit.quantile(.95):.1%}]**;\n- 90% interval for the mean range-ratio improvement: **[{B.delta_mean_range_ratio.quantile(.05):.2f}x, {B.delta_mean_range_ratio.quantile(.95):.2f}x]**.\n\nThis makes **compressed realized range + elevated ATM IV** the highest-priority hypothesis to freeze for the 2018-2026 holdout. It is economically interpretable as the options market continuing to price movement while realized factor range is temporarily suppressed.\n\nBy contrast, 25D RR selects compressed-release direction only about **53%** of the time in this development half. Do not treat skew as a solved direction layer.\n\nLeveraged-fund options conditioning remains too episode-concentrated for inference: the 25 crowded weekly observations cluster into a small number of episodes, including a single February-March 2016 high-ATM episode. The apparent stronger unwind under high ATM must therefore be treated as a hypothesis for the later holdout, not independent evidence.\n'''
if '## Robustness addendum' not in txt:rp.write_text(txt+extra)
# copy script and refresh manifest/zip
shutil.copy2(__file__,OUT/'scripts'/Path(__file__).name)
manifest=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='SHA256_MANIFEST.csv':manifest.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(manifest).to_csv(OUT/'SHA256_MANIFEST.csv',index=False)
zipout=BASE/'FX_OPTIONS_COT_PCA_PRE2018_V1_2026-08-24.zip'
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,arcname=f'{OUT.name}/{p.relative_to(OUT)}')
print(zipout)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(s).to_string(index=False))
print(E.to_string(index=False))
