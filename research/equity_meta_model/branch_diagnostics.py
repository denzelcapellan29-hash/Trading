import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
OUT=Path('/mnt/data/fund_meta/results_v2')
sc=pd.read_csv(OUT/'walkforward_trade_panel_with_fundamentals.csv.gz',parse_dates=['signal_date'])
base=pd.read_csv('/mnt/data/Equity_Barbell_MetaModel_v1_2026-08-25/data/barbell_trade_panel.csv.gz',parse_dates=['signal_date'])[['signal_date','ticker','gross_trade_ret_5d','state_turnaround','state_leader']]
d=sc.merge(base,on=['signal_date','ticker','gross_trade_ret_5d'],how='left')
r20=pd.read_csv(OUT/'trade_20d_fundamental_diagnostic.csv.gz',parse_dates=['signal_date'])[['signal_date','ticker','ret20_fwd']]
d=d.merge(r20,on=['signal_date','ticker'],how='left')
rows=[]
for branch,col in [('turnaround','state_turnaround'),('leader','state_leader')]:
    z=d[(d.signal_date>='2015-01-01')&(d.signal_date<='2026-08-07')&(d[col]==1)]
    for model in ['base11','fund12']:
      for score in ['pwin','q50','ev']:
        s=f'{model}_{score}'; b=f'{model}_{score}_bucket'; zz=z[z[s].notna()]
        rows.append({'branch':branch,'model':model,'score':score,'n':len(zz),'spearman':spearmanr(zz[s],zz.gross_trade_ret_5d).statistic,
                     'q1_ret':zz[zz[b]==1].gross_trade_ret_5d.mean(),'q5_ret':zz[zz[b]==5].gross_trade_ret_5d.mean(),
                     'q5_minus_q1':zz[zz[b]==5].gross_trade_ret_5d.mean()-zz[zz[b]==1].gross_trade_ret_5d.mean()})
pd.DataFrame(rows).to_csv(OUT/'branch_model_diagnostics.csv',index=False)
d['wf_extension_half']=''
for yr in range(2015,2027):
    tr=d[(d.signal_date.dt.year<yr)&d.fund_extension.notna()]; te=d[(d.signal_date.dt.year==yr)&d.fund_extension.notna()]
    if len(tr)<100: continue
    med=tr.fund_extension.median(); d.loc[te.index,'wf_extension_half']=np.where(te.fund_extension<=med,'low','high')
fr=[]
for branch,col in [('turnaround','state_turnaround'),('leader','state_leader')]:
    z=d[(d.signal_date>='2015-01-01')&(d.signal_date<='2026-08-07')&(d[col]==1)&(d.wf_extension_half!='')]
    for half,g in z.groupby('wf_extension_half'):
        fr.append({'branch':branch,'extension_half':half,'n':len(g),'mean_5d':g.gross_trade_ret_5d.mean(),'win5':(g.gross_trade_ret_5d>0).mean(),
                   'mean_20d':g.ret20_fwd.mean(),'win20':(g.ret20_fwd>0).mean(),'MAE5':g.MAE_5d.mean(),'MFE5':g.MFE_5d.mean()})
pd.DataFrame(fr).to_csv(OUT/'branch_fundamental_extension_diagnostics.csv',index=False)
print('Model by branch')
print(pd.read_csv(OUT/'branch_model_diagnostics.csv').to_string(index=False))
print('\nFundamentals by branch')
print(pd.read_csv(OUT/'branch_fundamental_extension_diagnostics.csv').to_string(index=False))
