#!/usr/bin/env python3
"""Independent frozen rolling daily EG computation; retrospective timing DIAGNOSTIC.

Respects per-pair factor count, ridge, ADF lag1 and frozen per-pair critical.
API chart daily bar START does not provide actual historical publication time.
Therefore this comparison explicitly cannot certify production event-time parity.
Actual live EG must consume verified source-version event ledger, not this join.
"""
import argparse, ast, io, json, math, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

OK=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Name,ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.USub,ast.UAdd,ast.Call,ast.Load)

def calc_expr(expr, v):
    tree=ast.parse(expr,mode='eval')
    if not all(isinstance(n,OK) for n in ast.walk(tree)):raise ValueError('unexpected frozen expression')
    if {n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}-set(v)-{'f_log'}:return np.nan
    def f_log(x):return math.log(x) if math.isfinite(x) and x>0 else np.nan
    return eval(compile(tree,'<frozen-expression>','eval'),{'__builtins__':{},'f_log':f_log},v)

def load_api(path):
    data={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('_D.csv'):continue
            key=Path(name).stem.removesuffix('_D').rstrip('_')
            df=pd.read_csv(io.BytesIO(z.read(name)))[['time','close']]
            df=df.dropna().sort_values('time').drop_duplicates('time')
            data[key]=df
    return data

def fit_eg(matrix,critical):
    # Pine f_olsN: ridge 1e-10, intercept included, nObs == length.
    matrix=np.asarray(matrix,float)
    if matrix.ndim!=2 or not np.isfinite(matrix).all():return (False,False,np.nan)
    y=matrix[:,0];X=np.column_stack((np.ones(len(y)),matrix[:,1:]))
    try:
        beta=np.linalg.inv(X.T@X+1e-10*np.eye(X.shape[1]))@(X.T@y)
        r=y-X@beta
        # Pine f_adf_c_lag1: regress Delta residual_t on 1,r[t-1],Delta residual[t-1].
        dep=r[2:]-r[1:-1]
        Z=np.column_stack((np.ones(len(dep)),r[1:-1],r[1:-1]-r[:-2]))
        inv=np.linalg.inv(Z.T@Z+1e-12*np.eye(3))
        coeff=inv@(Z.T@dep)
        err=dep-Z@coeff
        sse=float(err@err);s2=sse/(len(dep)-3)
        v=s2*inv[1,1]
        if v<=0 or not np.isfinite(v):return (False,False,np.nan)
        t=float(coeff[1]/np.sqrt(v))
        return (True,t<=critical,t)
    except np.linalg.LinAlgError:return (False,False,np.nan)

def main(hand,fx_zip,fac_zip,out):
    out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(hand) as z:
        prefix=next(n for n in z.namelist() if n.endswith('/START_HERE.md')).removesuffix('START_HERE.md')
        config=json.loads(z.read(prefix+'config/PAIR_MODEL_CONFIG.json'))
        archived=pd.read_csv(io.BytesIO(z.read(prefix+'reference/weekly_signal_ledger.csv')))
        pine={pair:z.read(prefix+'pine/'+pair+'.pine').decode() for pair in config}
    src={**load_api(fac_zip),**load_api(fx_zip)}
    rows=[];test=[]
    for pair,setting in config.items():
        model=src.get('FX_'+pair)
        if model is None:continue
        p=pine[pair]
        aliases={x.upper():y for x,y in re.findall(r'string\s+sym([A-Za-z0-9]+)\s*=\s*input\.symbol\("([^"]+)',p)}
        match=re.search(r'float dailyEgCritical = input.float\(([-+0-9.eE]+)',p)
        longmatch=re.search(r'float longEgCritical = input.float\(([-+0-9.eE]+)',p)
        if not match or not longmatch:raise ValueError('missing frozen critical for '+pair)
        critical=float(match.group(1));long_critical=float(longmatch.group(1))
        frame=model.rename(columns={'time':'model_start','close':'spot'}).copy()
        for symbol in sorted({x for f in setting['factors'] for x in f['symbols']}):
            actual=aliases.get(symbol.upper())
            if actual is None:raise ValueError('missing frozen symbol '+pair+' '+symbol)
            key=re.sub('[^A-Za-z0-9]+','_',actual).strip('_')
            d=src.get(key)
            if d is None:
                frame[symbol]=np.nan;continue
            # Historical retrospective start-to-start alignment ONLY, not true as-published proof.
            rhs=d.rename(columns={'time':'src_start','close':symbol}).sort_values('src_start')
            frame=pd.merge_asof(frame.sort_values('model_start'),rhs,left_on='model_start',right_on='src_start',direction='backward')
            frame=frame.drop(columns=['src_start'])
        x=[]
        for v in frame.itertuples(index=False):
            values=v._asdict();row=[]
            for f in setting['factors']:
                vals={alias.lower():values.get(alias,np.nan) for alias in f['symbols']}
                try:row.append(calc_expr(f['expression'],vals))
                except Exception:row.append(np.nan)
            x.append(row)
        matrix=np.column_stack((frame.spot.to_numpy(float),np.asarray(x,float)))
        for i in range(125,len(frame)):
            sample63=matrix[i-62:i+1];sample126=matrix[i-125:i+1]
            a,s,t=fit_eg(sample63,critical);b,q,u=fit_eg(sample126,long_critical)
            model_start=float(frame.iloc[i].model_start)
            # Reference Pine weekly signal context takes last daily observed state of prior FXCM week.
            rows.append({'pair':pair,'model_daily_start':model_start,'approx_model_daily_end':model_start+86400,
                         'eg63_available':int(a),'eg63_stable':int(s),'eg63_t':t,
                         'eg126_available':int(b),'eg126_stable':int(q),'eg126_t':u})
    result=pd.DataFrame(rows)
    result.to_csv(out/'daily_eg_recomputed.csv',index=False)
    # If archived weekly selected_ts falls after last daily model bar, nearest prior daily version
    # could correspond to the frozen week, but no historical publication proof is asserted.
    if not result.empty:
        for pair,grp in archived.groupby('pair'):
            x=result[result.pair==pair].sort_values('approx_model_daily_end')
            if x.empty:continue
            grp=grp.assign(selected_epoch=grp.selected_ts/1000.0)
            merged=pd.merge_asof(grp.sort_values('selected_epoch'),x,left_on='selected_epoch',right_on='approx_model_daily_end',direction='backward',suffixes=('_ref','_api'))
            for v in merged.itertuples(index=False):
                t=getattr(v,'eg63_t_api')
                if not np.isfinite(t):continue
                test.append({'pair':pair,'selected_ts':int(v.selected_ts),'matched_daily_start':v.model_daily_start,
                            'api_t63':t,'archived_t63':getattr(v,'eg63_t_ref'),
                            'abs_t63_diff':abs(t-getattr(v,'eg63_t_ref')),
                            'api_eg63_stable':getattr(v,'eg63_stable_api'),
                            'archived_eg63_stable':getattr(v,'eg63_stable_ref')})
    comp=pd.DataFrame(test);comp.to_csv(out/'archived_eg_retrospective_diagnostic.csv',index=False)
    summ={'daily_eg_rows':len(result),'computed_pairs':result.pair.nunique() if len(result) else 0,
          'archived_compared_rows':len(comp),'eg63_stable_matches':int((comp.api_eg63_stable==comp.archived_eg63_stable).sum()) if len(comp) else 0,
          'median_abs_eg63_t_diff':float(comp.abs_t63_diff.median()) if len(comp) else None,
          'status':'RETROSPECTIVE_ALIGNMENT_ONLY; weekly Pine daily security session timing unverified; no live order authorization'}
    (out/'summary.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('handoff',type=Path);p.add_argument('fxcm_daily',type=Path);p.add_argument('factor_daily',type=Path);p.add_argument('--out',type=Path,default=Path('artifacts/daily_eg_shadow'))
 a=p.parse_args();main(a.handoff,a.fxcm_daily,a.factor_daily,a.out)
