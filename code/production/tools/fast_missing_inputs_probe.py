#!/usr/bin/env python3
"""Target only unqualified current-source FAST pairs; prints no licensed raw prices."""
from __future__ import annotations
import json,os,sys,time,argparse,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fast_live_signal_diagnostic import fetch_chart, _select_observed_chart_bar

TARGETS=('CADJPY','CHFJPY','EURJPY','EURNZD','GBPNZD','NZDCAD','NZDCHF','NZDJPY','NZDUSD','USDJPY')

def summarize(c,charts,obs):
 out={}
 for pair in TARGETS:
  setting=c['pairs'][pair];symbol=setting['fxcm_model_symbol'];out[pair]={}
  for tf,window,limit in [('D',126,4*86400),('W',52,8*86400)]:
   model=charts[(symbol,tf)];span=86400 if tf=='D' else 5*86400
   ends=[x+span for x in model[:,0] if x+span<=obs][-window:]
   missing=[]
   for src in sorted({s for f in setting['factors'] for s in f['symbols'].values()}):
    source=charts[(src,tf)];bad=0;worst=0.;examples=[]
    for t in ends:
     idx=int(np.searchsorted(source[:,0],t,side='right')-1)
     age=t-source[idx,0] if idx>=0 else float('inf')
     if not math.isfinite(age) or age>limit:
      bad+=1
      if math.isfinite(age):worst=max(worst,age/86400)
    if bad:missing.append({'symbol':src,'missing_in_last_window':bad,'max_finite_stale_age_days_rounded':round(worst,1),'rows_returned':len(source)})
   out[pair][tf]={'window_bars':len(ends),'sources_with_gaps':missing}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('frozen_fast_pair_inputs.json'));a=p.parse_args()
 sid=os.getenv('SESSIONID') or os.getenv('TV_SESSIONID');sig=os.getenv('SESSIONID_SIGN') or os.getenv('TV_SESSIONID_SIGN')
 if not sid or not sig:raise SystemExit('BLOCK: AUTH_TRADINGVIEW_REQUIRED')
 from tradingviewApiPython import Client
 c=json.loads(a.manifest.read_text());symbols=sorted({s for pair in TARGETS for f in c['pairs'][pair]['factors'] for s in f['symbols'].values()}|{c['pairs'][pair]['fxcm_model_symbol'] for pair in TARGETS})
 client=Client(token=sid,signature=sig);charts={}
 try:
  for tf,depth in [('D',270),('W',260)]:
   for s in symbols:
    charts[(s,tf)]=fetch_chart(client,s,tf,depth)
 finally:client.end()
 obs=time.time();result={'targeted_pairs':len(TARGETS),'source_symbols':len(symbols),'source_surfaces':len(charts),'coverage':summarize(c,charts,obs),'licensed_raw_prices_logged':False}
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
