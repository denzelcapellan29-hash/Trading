from __future__ import annotations
import csv, io, json, math, re, zipfile, os, hashlib
from pathlib import Path
from collections import OrderedDict, defaultdict
import numpy as np

ZIP=Path('/mnt/data/Parity_Sample_FAST_8.zip')
OUT=Path('/mnt/data/FAST31_PYTHON_REBUILD_CURRENT_SOURCE')
OUT.mkdir(parents=True, exist_ok=True)
RIDGE=1e-10
TOL=1e-7

def fnum(s):
    if s is None: return math.nan
    s=s.strip()
    if s=='': return math.nan
    try: return float(s)
    except: return math.nan

def fint(s):
    v=fnum(s)
    return int(round(v)) if math.isfinite(v) else 0

def ols(y,x):
    y=np.asarray(y,dtype=float); x=np.asarray(x,dtype=float)
    if y.ndim!=1 or x.ndim!=2 or len(y)!=len(x): return None
    z=np.column_stack([y,x])
    if not np.isfinite(z).all(): return None
    X=np.column_stack([np.ones(len(y)),x])
    try:
        b=np.linalg.solve(X.T@X+RIDGE*np.eye(X.shape[1]),X.T@y)
    except np.linalg.LinAlgError:
        return None
    fit=X@b; r=y-fit
    return b,r,fit

def get_idx(h,name):
    try:return h.index(name)
    except ValueError:return None

def parse_chart(zf, member, pair):
    with zf.open(member) as raw:
        text=io.TextIOWrapper(raw,encoding='utf-8-sig',newline='')
        rd=csv.reader(text)
        h=next(rd)
        idx={name:get_idx(h,name) for name in [
            'time','EXPORT Selected Completed Period Timestamp','EXPORT Selected FXCM Model Spot Close',
            'EXPORT Primary Fair Value','EXPORT Primary Residual','EXPORT Primary Sigma','EXPORT Primary Z',
            'EXPORT Secondary Fair Value','EXPORT Secondary Residual','EXPORT Secondary Sigma','EXPORT Secondary Z',
            'EXPORT EG63 Available','EXPORT EG63 Stable','EXPORT EG63 T','EXPORT EG126 Available','EXPORT EG126 Stable','EXPORT EG126 T',
            'EXPORT EG63 Regime Age Weeks','EXPORT EG126 Regime Age Weeks',
            'EXPORT Confidence Coefficient Drift13 Z','EXPORT Confidence Split Disagreement Z','EXPORT Recent Residual Vol Ratio',
            'EXPORT Residual Acceleration Raw Z','EXPORT Confidence Score','EXPORT Confidence High',
            'EXPORT FXCM Signal Vol13 Annualized','EXPORT FXCM Signal Vol13 Percentile','EXPORT FXCM Signal Vol Regime 1Low 2Middle 3High',
            'EXPORT Frozen Signal Direction','EXPORT Branch 1Primary 2Secondary','EXPORT Primary Marginal 1p50 to 1p625',
            'EXPORT Weak Mature Primary','EXPORT Targeted Middle Vol Reduction','EXPORT Residual Acceleration Favorable',
            'EXPORT Final Categorical Multiplier','EXPORT Entry Event','EXPORT Active Entry Timestamp','EXPORT Active Entry Price',
            'EXPORT Active Quantity Units','EXPORT Active Selected Risk Percent'
        ]}
        xcols=[(i,c) for i,c in enumerate(h) if c.startswith('EXPORT X')]
        rawcols=[(i,c) for i,c in enumerate(h) if c.startswith('EXPORT Raw ') and not c.endswith('Source Timestamp') and c!='EXPORT Raw Symbol Count']
        srccols=[(i,c) for i,c in enumerate(h) if c.startswith('EXPORT Raw ') and c.endswith('Source Timestamp')]
        mode_idx=get_idx(h,'EXPORT Surface Mode 1Weekly 2Daily')
        weekly=OrderedDict(); entries=[]
        for row in rd:
            if len(row)<len(h): row += ['']*(len(h)-len(row))
            if mode_idx is not None and fint(row[mode_idx])!=1: continue
            ts=fint(row[idx['EXPORT Selected Completed Period Timestamp']]) if idx['EXPORT Selected Completed Period Timestamp'] is not None else 0
            if ts>0 and ts not in weekly:
                rec={'pair':pair,'selected_ts':ts,'chart_time':row[idx['time']] if idx['time'] is not None else ''}
                rec['spot']=fnum(row[idx['EXPORT Selected FXCM Model Spot Close']])
                rec['x']=[fnum(row[i]) for i,_ in xcols]
                rec['x_names']=[n for _,n in xcols]
                for name,i in idx.items():
                    if i is not None and name not in ('time','EXPORT Selected Completed Period Timestamp','EXPORT Selected FXCM Model Spot Close'):
                        rec[name]=fnum(row[i])
                for i,n in rawcols: rec[n]=fnum(row[i])
                for i,n in srccols: rec[n]=fnum(row[i])
                weekly[ts]=rec
            ie=idx['EXPORT Entry Event']
            if ie is not None and fint(row[ie])==1:
                entries.append({
                    'pair':pair,'chart_time':row[idx['time']],
                    'selected_ts':ts,
                    'entry_ts':fint(row[idx['EXPORT Active Entry Timestamp']]),
                    'entry_price':fnum(row[idx['EXPORT Active Entry Price']]),
                    'qty':fnum(row[idx['EXPORT Active Quantity Units']]),
                    'risk_pct':fnum(row[idx['EXPORT Active Selected Risk Percent']]),
                    'direction':fint(row[idx['EXPORT Frozen Signal Direction']]),
                    'branch':fint(row[idx['EXPORT Branch 1Primary 2Secondary']]),
                })
        return list(weekly.values()),entries,[n for _,n in xcols],h

def parse_trades(zf,member,pair):
    rows=[]
    with zf.open(member) as raw:
        text=io.TextIOWrapper(raw,encoding='utf-8-sig',newline='')
        rd=csv.DictReader(text)
        for r in rd:
            typ=r.get('Type','')
            if not typ.startswith('Entry '): continue
            direction=1 if 'long' in typ.lower() else -1
            rows.append({
                'pair':pair,'trade_number':r.get('Trade number',''),'entry_time':r.get('Date and time',''),
                'direction':direction,'price':fnum(r.get('Price '+pair[3:],'') or next((v for k,v in r.items() if k and k.startswith('Price ')),'')),
                'qty':fnum(r.get('Size (qty)','')),'net_pnl_usd':fnum(r.get('Net PnL USD','')),
                'return_pct':fnum(r.get('Return %','')),'duration_bars':fnum(r.get('Duration (bars)','')),
            })
    return rows

def rebuild_pair(rows):
    rows=sorted(rows,key=lambda r:r['selected_ts'])
    n=len(rows); k=len(rows[0]['x']) if rows else 0
    y=np.array([r['spot'] for r in rows],float)
    x=np.array([r['x'] for r in rows],float) if rows else np.empty((0,0))
    detail=[]
    for i,rw in enumerate(rows):
        calc={k:math.nan for k in ['pfv','pres','psig','pz','sfv','sres','ssig','sz']}
        if i>=51:
            z=ols(y[i-51:i+1],x[i-51:i+1])
            if z is not None:
                b,res,fit=z; sd=float(np.std(res,ddof=1))
                if sd>0:
                    calc.update(pfv=float(fit[-1]),pres=float(res[-1]),psig=sd,pz=float(res[-1]/sd))
        if i>=52:
            dy=np.diff(y[i-52:i+1]); dx=np.diff(x[i-52:i+1],axis=0)
            z=ols(dy,dx)
            if z is not None:
                b,res,fit=z; sd=float(np.std(res,ddof=1))
                if sd>0:
                    calc.update(sfv=float(y[i-1]+fit[-1]),sres=float(res[-1]),ssig=sd,sz=float(res[-1]/sd))
        pine={
            'pfv':rw.get('EXPORT Primary Fair Value',math.nan),'pres':rw.get('EXPORT Primary Residual',math.nan),
            'psig':rw.get('EXPORT Primary Sigma',math.nan),'pz':rw.get('EXPORT Primary Z',math.nan),
            'sfv':rw.get('EXPORT Secondary Fair Value',math.nan),'sres':rw.get('EXPORT Secondary Residual',math.nan),
            'ssig':rw.get('EXPORT Secondary Sigma',math.nan),'sz':rw.get('EXPORT Secondary Z',math.nan),
        }
        eg_av=bool(round(rw.get('EXPORT EG63 Available',0)))
        eg_st=bool(round(rw.get('EXPORT EG63 Stable',0)))
        pz=calc['pz']; sz=calc['sz']; absz=abs(pz) if math.isfinite(pz) else math.nan
        direction=0; branch=0
        if eg_av and math.isfinite(pz):
            if eg_st:
                if pz < -1.5: direction=1
                elif pz > 1.5: direction=-1
                if direction: branch=1
            elif absz>1.625 and math.isfinite(sz):
                if sz <= -2.0: direction=1
                elif sz >= 2.0: direction=-1
                if direction: branch=2
        marginal=branch==1 and absz>1.5 and absz<=1.625 if math.isfinite(absz) else False
        eg63age=rw.get('EXPORT EG63 Regime Age Weeks',math.nan)
        eg126av=bool(round(rw.get('EXPORT EG126 Available',0)))
        eg126st=bool(round(rw.get('EXPORT EG126 Stable',0)))
        eg126age=rw.get('EXPORT EG126 Regime Age Weeks',math.nan)
        longconfirm=eg126av and eg126st and math.isfinite(eg126age) and eg126age>=26
        weak=branch==1 and math.isfinite(eg63age) and eg63age>=13 and not longconfirm
        accz=rw.get('EXPORT Residual Acceleration Raw Z',math.nan)
        accel=direction!=0 and math.isfinite(accz) and direction*accz>=0
        confhigh=bool(round(rw.get('EXPORT Confidence High',0)))
        base=math.nan if direction==0 else (1.25 if confhigh and accel else 1.0 if confhigh or accel else 0.75)
        volreg=fint(str(rw.get('EXPORT FXCM Signal Vol Regime 1Low 2Middle 3High',0)))
        mid=branch==1 and not weak and base==0.75 and volreg==2
        mult=math.nan if direction==0 else (1.0 if branch==2 else 0.5 if marginal or weak or mid else base)
        out={'pair':rw['pair'],'selected_ts':rw['selected_ts'],'chart_time':rw['chart_time'],'spot':rw['spot']}
        for j,nm in enumerate(rw['x_names']): out[f'x{j+1}']=rw['x'][j]; out[f'x{j+1}_name']=nm
        for key in calc:
            out['python_'+key]=calc[key]; out['pine_'+key]=pine[key]
            out['absdiff_'+key]=abs(calc[key]-pine[key]) if math.isfinite(calc[key]) and math.isfinite(pine[key]) else math.nan
        out.update({
            'python_direction':direction,'pine_direction':fint(str(rw.get('EXPORT Frozen Signal Direction',0))),
            'python_branch':branch,'pine_branch':fint(str(rw.get('EXPORT Branch 1Primary 2Secondary',0))),
            'python_marginal':int(marginal),'pine_marginal':fint(str(rw.get('EXPORT Primary Marginal 1p50 to 1p625',0))),
            'python_weak_mature':int(weak),'pine_weak_mature':fint(str(rw.get('EXPORT Weak Mature Primary',0))),
            'python_targeted_middle':int(mid),'pine_targeted_middle':fint(str(rw.get('EXPORT Targeted Middle Vol Reduction',0))),
            'python_accel_favorable':int(accel),'pine_accel_favorable':fint(str(rw.get('EXPORT Residual Acceleration Favorable',0))),
            'python_multiplier':mult,'pine_multiplier':rw.get('EXPORT Final Categorical Multiplier',math.nan),
            'eg63_available':int(eg_av),'eg63_stable':int(eg_st),'eg63_t':rw.get('EXPORT EG63 T',math.nan),
            'eg126_available':int(eg126av),'eg126_stable':int(eg126st),'eg126_t':rw.get('EXPORT EG126 T',math.nan),
        })
        detail.append(out)
    return detail

def write_csv(path,rows,fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('',encoding='utf-8'); return
    if fields is None:
        fields=[]; seen=set()
        for r in rows:
            for k in r:
                if k not in seen:seen.add(k);fields.append(k)
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

with zipfile.ZipFile(ZIP) as zf:
    members=zf.namelist()
    chart_members={re.search(r'OANDA_([A-Z]{6}),',m).group(1):m for m in members if re.search(r'OANDA_([A-Z]{6}),',m)}
    trade_members={re.match(r'([A-Z]{6})_F615_',Path(m).name).group(1):m for m in members if re.match(r'([A-Z]{6})_F615_',Path(m).name)}
    all_weekly=[];all_entries=[];all_trades=[];pair_summ=[];indicator_meta=[];factor_long=[]
    for pair in sorted(chart_members):
        weekly,entries,xnames,header=parse_chart(zf,chart_members[pair],pair)
        detail=rebuild_pair(weekly)
        for wr in weekly:
            for key,val in wr.items():
                if key.startswith('EXPORT Raw ') and not key.endswith('Source Timestamp') and key!='EXPORT Raw Symbol Count':
                    label=key[len('EXPORT Raw '):]
                    skey=key+' Source Timestamp'
                    sts=wr.get(skey,math.nan)
                    age=(wr['selected_ts']-sts)/86400000.0 if math.isfinite(sts) else math.nan
                    factor_long.append({'pair':pair,'selected_ts':wr['selected_ts'],'factor':label,'value':val,'source_ts':sts,'source_age_days':age,
                                        'missing_value':int(not math.isfinite(val)),'missing_source_ts':int(not math.isfinite(sts)),
                                        'future_source':int(math.isfinite(sts) and sts>wr['selected_ts'])})
        all_weekly.extend(detail);all_entries.extend(entries)
        trades=parse_trades(zf,trade_members[pair],pair) if pair in trade_members else []
        all_trades.extend(trades)
        valid=[r for r in detail if math.isfinite(r['python_pz']) and math.isfinite(r['pine_pz'])]
        def maxd(k):
            vals=[r['absdiff_'+k] for r in valid if math.isfinite(r['absdiff_'+k])]
            return max(vals) if vals else math.nan
        core_fail=sum(any(math.isfinite(r['absdiff_'+k]) and r['absdiff_'+k]>TOL for k in ['pfv','pres','psig','pz','sfv','sres','ssig','sz']) for r in valid)
        pair_summ.append({
            'pair':pair,'weekly_rows':len(detail),'valid_recomputed_rows':len(valid),'x_count':len(xnames),
            'max_abs_primary_fair_value':maxd('pfv'),'max_abs_primary_z':maxd('pz'),'max_abs_secondary_z':maxd('sz'),
            'core_rows_over_1e-7':core_fail,
            'python_signal_weeks':sum(r['python_direction']!=0 for r in valid),
            'pine_signal_weeks':sum(r['pine_direction']!=0 for r in valid),
            'signal_direction_mismatches':sum(r['python_direction']!=r['pine_direction'] for r in valid),
            'branch_mismatches':sum(r['python_branch']!=r['pine_branch'] for r in valid),
            'marginal_flag_mismatches':sum(r['python_marginal']!=r['pine_marginal'] for r in valid),
            'weak_mature_mismatches':sum(r['python_weak_mature']!=r['pine_weak_mature'] for r in valid),
            'targeted_middle_mismatches':sum(r['python_targeted_middle']!=r['pine_targeted_middle'] for r in valid),
            'accel_flag_mismatches':sum(r['python_accel_favorable']!=r['pine_accel_favorable'] for r in valid),
            'multiplier_mismatches':sum((math.isfinite(r['python_multiplier']) or math.isfinite(r['pine_multiplier'])) and (not(math.isfinite(r['python_multiplier']) and math.isfinite(r['pine_multiplier'])) or abs(r['python_multiplier']-r['pine_multiplier'])>1e-9) for r in valid),
            'chart_entry_events':len(entries),'strategy_tester_entries':len(trades),
        })
        indicator_meta.append({'pair':pair,'chart_member':chart_members[pair],'trade_member':trade_members.get(pair,''),'x_fields':' | '.join(xnames)})

write_csv(OUT/'weekly_surface_detail.csv',all_weekly)
write_csv(OUT/'pair_level_rebuild_parity.csv',pair_summ)
write_csv(OUT/'chart_entry_events.csv',all_entries)
write_csv(OUT/'strategy_tester_entry_ledger.csv',all_trades)
write_csv(OUT/'input_inventory.csv',indicator_meta)
write_csv(OUT/'factor_surface_long.csv',factor_long)
fac_groups=defaultdict(list)
for r in factor_long: fac_groups[(r['pair'],r['factor'])].append(r)
fac_summary=[]
for (pair,factor),rr in sorted(fac_groups.items()):
    ages=[x['source_age_days'] for x in rr if math.isfinite(x['source_age_days'])]
    fac_summary.append({'pair':pair,'factor':factor,'observations':len(rr),'missing_values':sum(x['missing_value'] for x in rr),
                        'missing_source_timestamps':sum(x['missing_source_ts'] for x in rr),'future_source_timestamps':sum(x['future_source'] for x in rr),
                        'median_source_age_days':float(np.median(ages)) if ages else math.nan,
                        'max_source_age_days':max(ages) if ages else math.nan,
                        'source_age_over_10d':sum(a>10 for a in ages)})
write_csv(OUT/'factor_source_age_summary.csv',fac_summary)
sig=[r for r in all_weekly if math.isfinite(r['python_pz']) and (r['python_direction']!=0 or r['pine_direction']!=0)]
write_csv(OUT/'weekly_signal_ledger.csv',sig)
entry_by={(r['pair'],r['selected_ts']):r for r in all_entries if r.get('selected_ts',0)>0}
signal_entry=[]
for r in sig:
    e=entry_by.get((r['pair'],r['selected_ts']))
    signal_entry.append({
        'pair':r['pair'],'selected_ts':r['selected_ts'],'python_direction':r['python_direction'],
        'python_branch':r['python_branch'],'python_multiplier':r['python_multiplier'],
        'chart_entry_found':int(e is not None),'chart_entry_time':e['chart_time'] if e else '',
        'chart_direction':e['direction'] if e else '', 'chart_branch':e['branch'] if e else '',
        'direction_match':int(e is not None and e['direction']==r['python_direction']),
        'branch_match':int(e is not None and e['branch']==r['python_branch']),
    })
write_csv(OUT/'signal_to_chart_entry_parity.csv',signal_entry)

summary={
 'pairs':len(pair_summ),
 'weekly_rows':len(all_weekly),
 'valid_recomputed_rows':sum(r['valid_recomputed_rows'] for r in pair_summ),
 'python_signal_weeks':sum(r['python_signal_weeks'] for r in pair_summ),
 'pine_signal_weeks':sum(r['pine_signal_weeks'] for r in pair_summ),
 'signal_direction_mismatches':sum(r['signal_direction_mismatches'] for r in pair_summ),
 'branch_mismatches':sum(r['branch_mismatches'] for r in pair_summ),
 'multiplier_mismatches':sum(r['multiplier_mismatches'] for r in pair_summ),
 'chart_entry_events':len(all_entries),
 'strategy_tester_entries':len(all_trades),
 'core_rows_over_1e-7':sum(r['core_rows_over_1e-7'] for r in pair_summ),
 'recomputed_signals_with_chart_entry':sum(r['chart_entry_found'] for r in signal_entry),
 'recomputed_signal_direction_entry_mismatches':sum(1-r['direction_match'] for r in signal_entry),
 'recomputed_signal_branch_entry_mismatches':sum(1-r['branch_match'] for r in signal_entry),
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
for r in pair_summ:
    print(r['pair'],r['valid_recomputed_rows'],r['max_abs_primary_z'],r['signal_direction_mismatches'],r['multiplier_mismatches'])
