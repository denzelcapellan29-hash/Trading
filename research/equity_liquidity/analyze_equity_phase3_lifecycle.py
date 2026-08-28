from pathlib import Path
import glob, pandas as pd, numpy as np
OUT=Path('/mnt/data/equity_fx_framework_phase3_final'); OUT.mkdir(exist_ok=True)
FILES=sorted(glob.glob('/mnt/data/phase3_batches/batch_*_lifecycle.csv.gz'))
USE=['ticker','snapshot_time','period','week','trust_family','Q_simple','near_train_q1','dist_atr','node_age_bars',
     'known_touch_episodes','known_touch_sessions','zone_touch_sessions_100','clean_resolution','hold','target_exists','path_hit']
D=pd.concat([pd.read_csv(f,usecols=USE) for f in FILES],ignore_index=True)
D['snapshot_time']=pd.to_datetime(D.snapshot_time,utc=True)
D['week_dt']=D.snapshot_time.dt.to_period('W-FRI').dt.start_time

x=D[D.near_train_q1].copy(); x['episode_bucket']=pd.cut(x.known_touch_episodes,[-1,0,1,2,3,4,999],labels=['0','1','2','3','4','5+'])
rows=[]
for fam,fg in list(x.groupby('trust_family'))+[('ALL',x)]:
 for per,g in list(fg.groupby('period'))+[('ALL',fg)]:
  for ep,z in g.groupby('episode_bucket',observed=True):
   c=z[z.clean_resolution==1]; den=c[c.target_exists==1]
   rows.append({'trust_family':fam,'period':per,'episode_bucket':str(ep),'n':len(z),'clean_resolution':z.clean_resolution.mean(),
                'hold_share_clean':c.hold.mean() if len(c) else np.nan,'path_hit_conditional':den.path_hit.mean() if len(den) else np.nan,
                'mean_node_age':z.node_age_bars.mean(),'mean_dist':z.dist_atr.mean()})
pd.DataFrame(rows).to_csv(OUT/'primary_known_node_episode_buckets.csv',index=False)

defs=[('known_node_episodes','known_touch_episodes'),('known_node_sessions','known_touch_sessions'),('zone_history_sessions','zone_touch_sessions_100')]
rows=[]
for defname,fld in defs:
 for fam in ['usable_side_mixed','fragile_gateway_proxy','corridor','ALL']:
  fg=x if fam=='ALL' else x[x.trust_family==fam]
  for per in ['TRAIN_2013_2016','VALID_2017_2021','HOLDOUT_2022PLUS','ALL']:
   g=fg if per=='ALL' else fg[fg.period==per]
   groups=[('fresh_0_1',g[fld]<=1),('intermediate_2_4',(g[fld]>=2)&(g[fld]<=4)),('depleted_5plus',g[fld]>=5)]
   for lab,mask in groups:
    z=g[mask]; c=z[z.clean_resolution==1]; den=c[c.target_exists==1]
    rows.append({'definition':defname,'trust_family':fam,'period':per,'lifecycle':lab,'n':len(z),
                 'clean_resolution':z.clean_resolution.mean() if len(z) else np.nan,'hold_share_clean':c.hold.mean() if len(c) else np.nan,
                 'path_hit_conditional':den.path_hit.mean() if len(den) else np.nan})
sens=pd.DataFrame(rows); sens.to_csv(OUT/'lifecycle_definition_sensitivity.csv',index=False)

rows=[]
for fmax in [0,1,2]:
 for dmin in [4,5,6]:
  for qname,qg in [('ALL',x),('Q0',x[x.Q_simple==0]),('Q1',x[x.Q_simple==1]),('Q2',x[x.Q_simple==2])]:
   for per,g in list(qg.groupby('period'))+[('ALL',qg)]:
    for lab,z in [('FRESH',g[g.known_touch_episodes<=fmax]),('DEPLETED',g[g.known_touch_episodes>=dmin])]:
     c=z[z.clean_resolution==1]
     rows.append({'fresh_max':fmax,'depleted_min':dmin,'q_scope':qname,'period':per,'group':lab,'n':len(z),
                  'clean_resolution':z.clean_resolution.mean() if len(z) else np.nan,'hold_share_clean':c.hold.mean() if len(c) else np.nan})
pd.DataFrame(rows).to_csv(OUT/'primary_threshold_neighborhood.csv',index=False)

u=D[D.near_train_q1 & (D.trust_family=='usable_side_mixed') & (D.clean_resolution==1)].copy()
u['grp']=np.where(u.known_touch_episodes==0,'virgin',np.where(u.known_touch_episodes.isin([1,2]),'early','other')); u=u[u.grp!='other']
u['age_bin']=pd.cut(u.node_age_bars,[-1,5,10,15,25,50,100],labels=False); u['dist_bin']=pd.cut(u.dist_atr,[-1,.08,.16,.24,.357324],labels=False)
rr=[]
for keys,g in u.groupby(['period','Q_simple','age_bin','dist_bin'],dropna=True):
 a=g.groupby('grp').hold.agg(['mean','count'])
 if {'virgin','early'}.issubset(a.index) and a.loc['virgin','count']>=30 and a.loc['early','count']>=30:
  nv=float(a.loc['virgin','count']); ne=float(a.loc['early','count']); w=2*nv*ne/(nv+ne)
  rr.append({'period':keys[0],'Q_simple':keys[1],'age_bin':keys[2],'dist_bin':keys[3],'n_virgin':int(nv),'n_early':int(ne),
             'hold_virgin':a.loc['virgin','mean'],'hold_early':a.loc['early','mean'],'diff':a.loc['virgin','mean']-a.loc['early','mean'],'weight':w})
strata=pd.DataFrame(rr); strata.to_csv(OUT/'virgin_early_matched_strata.csv',index=False)
ss=[]
for per,g in list(strata.groupby('period'))+[('ALL',strata)]:
 ss.append({'period':per,'weighted_diff_virgin_minus_early':np.average(g['diff'],weights=g.weight),'n_strata':len(g),'n_virgin':int(g.n_virgin.sum()),'n_early':int(g.n_early.sum())})
pd.DataFrame(ss).to_csv(OUT/'virgin_early_standardized_summary.csv',index=False)

rr=[]
for t,g in u.groupby('ticker'):
 v=g[g.grp=='virgin']; r=g[g.grp=='early']
 if len(v)>=30 and len(r)>=30:
  rr.append({'ticker':t,'n_virgin':len(v),'n_early':len(r),'hold_virgin':v.hold.mean(),'hold_early':r.hold.mean(),'diff_virgin_minus_early':v.hold.mean()-r.hold.mean()})
pd.DataFrame(rr).to_csv(OUT/'virgin_early_stock_breadth.csv',index=False)

rng=np.random.default_rng(20260828); reps=3000; block=8; rows=[]
def boot(g,ma,mb):
 g=g[g.clean_resolution==1].copy(); weeks=np.array(sorted(g.week_dt.dropna().unique())); n=len(weeks)
 def arr(mask):
  a=g[mask].groupby('week_dt').hold.agg(['sum','count']).reindex(weeks,fill_value=0)
  return a['sum'].to_numpy(float),a['count'].to_numpy(float)
 sa,na=arr(ma.loc[g.index]); sb,nb=arr(mb.loc[g.index]); point=sa.sum()/na.sum()-sb.sum()/nb.sum()
 nblocks=(n+block-1)//block; starts=rng.integers(0,n,size=(reps,nblocks)); offs=np.arange(block)[None,None,:]
 ix=((starts[:,:,None]+offs)%n).reshape(reps,-1)[:,:n]
 vals=sa[ix].sum(axis=1)/np.maximum(na[ix].sum(axis=1),1)-sb[ix].sum(axis=1)/np.maximum(nb[ix].sum(axis=1),1)
 return point,np.quantile(vals,.025),np.quantile(vals,.975),(vals>0).mean(),int(na.sum()),int(nb.sum()),n
for fam in ['usable_side_mixed','fragile_gateway_proxy','corridor']:
 fg=D[D.near_train_q1 & (D.trust_family==fam)].copy()
 for per in ['TRAIN_2013_2016','VALID_2017_2021','HOLDOUT_2022PLUS','ALL']:
  g=fg if per=='ALL' else fg[fg.period==per]
  for label,ma,mb in [
   ('fresh0_1_minus_depleted5plus',g.known_touch_episodes<=1,g.known_touch_episodes>=5),
   ('virgin0_minus_early1_2',g.known_touch_episodes==0,g.known_touch_episodes.isin([1,2])),
   ('zone_fresh0_1_minus_depleted5plus',g.zone_touch_sessions_100<=1,g.zone_touch_sessions_100>=5)]:
   point,lo,hi,p,nA,nB,nw=boot(g,ma,mb)
   rows.append({'trust_family':fam,'period':per,'contrast':label,'point':point,'ci025':lo,'ci975':hi,'p_positive':p,'n_a_clean':nA,'n_b_clean':nB,'weeks':nw,'reps':reps})
pd.DataFrame(rows).to_csv(OUT/'lifecycle_weekly_block_bootstrap.csv',index=False)

main=sens[(sens.definition.isin(['known_node_episodes','known_node_sessions','zone_history_sessions'])) & (sens.trust_family=='usable_side_mixed')]
main.to_csv(OUT/'usable_side_lifecycle_main_table.csv',index=False)
