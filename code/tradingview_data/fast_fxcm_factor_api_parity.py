#!/usr/bin/env python3
"""Diagnostic: compare API-provided frozen FXCM weekly model inputs against archived Pine weekly x/spot surfaces.

Does NOT assert production parity: weekly chart snapshots may be revised/back-adjusted; Pine
request.security uses completed-week boundaries. Outputs differences and coverage for review.
"""
import argparse,ast,io,json,math,re,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ALLOWED=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Name,ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.USub,ast.UAdd,ast.Call,ast.Load)
def calc_expr(expr, values):
    tree=ast.parse(expr,mode='eval')
    if not all(isinstance(n,ALLOWED) for n in ast.walk(tree)):
        raise ValueError('disallowed factor expression')
    names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
    if names-set(values)-{'f_log'}:raise ValueError('unmapped factor: '+str(names-set(values)-{'f_log'}))
    def f_log(v):return math.log(v) if np.isfinite(v) and v>0 else np.nan
    return eval(compile(tree,'<frozen-factor>','eval'),{'__builtins__':{},'f_log':f_log},values)

def load_api(path):
    with zipfile.ZipFile(path) as z:
        names={re.sub(r'_W\.csv$','',Path(n).name).rstrip('_'):n for n in z.namelist() if n.endswith('_W.csv')}
        out={}
        for key,member in names.items():
            df=pd.read_csv(io.BytesIO(z.read(member)))
            df['time']=pd.to_numeric(df.time,errors='coerce')
            df['close']=pd.to_numeric(df.close,errors='coerce')
            df=df.dropna(subset=['time','close']).sort_values('time').drop_duplicates('time')
            out[key]=df[['time','close']]
        return out

def get_at(series,t):
    """Select the trading week containing the archived Friday model timestamp, not next week."""
    if series is None or series.empty:return (np.nan,np.nan)
    ts=series.time.to_numpy(dtype=float)
    idx=int(np.searchsorted(ts,t,side='right')-1)
    if idx<0:return (np.nan,np.nan)
    # If no bar started in the previous 7 days, mark missing instead of carrying stale weeks.
    age=t-ts[idx]
    if age<0 or age>7*86400:return (np.nan,np.nan)
    return float(series.close.iloc[idx]),float(age/86400)

def main(hand,fx_api,factor_api,out):
    out.mkdir(parents=True,exist_ok=True)
    z=zipfile.ZipFile(hand);prefix=next(n for n in z.namelist() if n.endswith('/START_HERE.md')).removesuffix('START_HERE.md')
    config=json.loads(z.read(prefix+'config/PAIR_MODEL_CONFIG.json'))
    ledger=pd.read_csv(io.BytesIO(z.read(prefix+'reference/weekly_signal_ledger.csv')))
    fx=load_api(fx_api);fac=load_api(factor_api);all_sources={**fac,**fx}
    rows=[];raw=[];spot=[];missing=[]
    for pair,setting in config.items():
        pine=z.read(prefix+'pine/'+pair+'.pine').decode()
        aliases={k.upper():v for k,v in re.findall(r'string\s+sym([A-Za-z0-9]+)\s*=\s*input\.symbol\("([^"]+)',pine)}
        data=ledger.loc[ledger.pair==pair].sort_values('selected_ts')
        for row in data.itertuples(index=False):
            selected_ts=int(row.selected_ts);t=selected_ts/1000
            val,age=get_at(fx.get('FX_'+pair),t)
            if np.isfinite(val):
                spot.append({'pair':pair,'selected_ts':selected_ts,'reference_spot':row.spot,
                    'api_spot':val,'api_week_age_days':age,'delta':val-row.spot,
                    'rel_delta':(val/row.spot-1) if row.spot else np.nan})
            for i,factor in enumerate(setting['factors'],start=1):
                xref=getattr(row,'x'+str(i),np.nan)
                if pd.isna(xref):continue
                vars_={};ages=[];ins=[]
                for alias in factor['symbols']:
                    symbol=aliases.get(alias.upper()); api=all_sources.get(symbol.replace(':','_').replace('!','_').strip('_')) if symbol else None
                    # CSV adapter uses re.sub('[^A-Za-z0-9]+', '_', symbol) with suffix _W.
                    if symbol:
                        key=re.sub(r'[^A-Za-z0-9]+','_',symbol).strip('_')
                        api=all_sources.get(key)
                    v,a=get_at(api,t)
                    vars_[alias.lower()]=v;ages.append(a);ins.append(symbol)
                if any(not np.isfinite(v) for v in vars_.values()):
                    missing.append({'pair':pair,'selected_ts':selected_ts,'factor_index':i,'symbols': '|'.join(str(x) for x in ins)})
                    continue
                value=calc_expr(factor['expression'],vars_)
                raw.append({'pair':pair,'selected_ts':selected_ts,'factor_index':i,'factor_name':factor['name'],
                    'manual_x':xref,'api_x':value,'delta':value-xref,'max_source_week_age_days':max(ages),
                    'symbols':'|'.join(ins)})
    fxr=pd.DataFrame(spot);xr=pd.DataFrame(raw)
    fxr.to_csv(out/'fxcm_weekly_spot_diffs.csv',index=False)
    xr.to_csv(out/'weekly_factor_diffs.csv',index=False)
    pd.DataFrame(missing).to_csv(out/'missing_factor_alignment.csv',index=False)
    def stat(df):
        if df.empty:return {'n':0}
        z=df.delta.abs()
        return {'n':len(df),'exact_1e9':int((z<1e-9).sum()),'max_abs':float(z.max()),
            'median_abs':float(z.median()),'p95_abs':float(z.quantile(.95))}
    s={'spot':stat(fxr),'factors':stat(xr),'factor_missing_rows':len(missing),
       'factor_by_pair':xr.groupby('pair').apply(lambda d:stat(d),include_groups=False).to_dict()}
    (out/'summary.json').write_text(json.dumps(s,indent=2))
    print(json.dumps({k:v for k,v in s.items() if k!='factor_by_pair'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('handoff',type=Path);p.add_argument('fxcm_weekly',type=Path);p.add_argument('factors_weekly',type=Path)
    p.add_argument('--out',type=Path,default=Path('/mnt/data/fast_production_gate'))
    a=p.parse_args();main(a.handoff,a.fxcm_weekly,a.factors_weekly,a.out)
