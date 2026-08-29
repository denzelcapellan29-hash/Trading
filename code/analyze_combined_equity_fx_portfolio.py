#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

CORRIDOR_WEIGHTS = [0.0, 0.10, 0.15, 0.20]
FX_CAPITAL_WEIGHTS = [0.25, 1/3, 0.40, 0.50]
FX_LEVERAGES = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_equity(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith('14_weekly_protected_plus_corridor_accept.csv'))
        x = pd.read_csv(z.open(name), parse_dates=['period_end']).set_index('period_end').sort_index()
    # Phase-6 equity rows are labeled by the signal/rebalance Friday even though
    # the contained return realizes over the following week. Convert to the
    # actual realization-week Friday before joining to FAST, whose weekly rows
    # are already labeled by the actual Monday-entry trade week's Friday.
    x.index = x.index + pd.Timedelta(days=7)
    x.index.name = 'period_end'
    return x


def load_fx(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(path) as z:
        f1 = next(n for n in z.namelist() if n.endswith('/data/fx_fast_weekly_returns.csv'))
        f2 = next(n for n in z.namelist() if n.endswith('/data/fx_validated_65fast_35alt_weekly_returns.csv'))
        fast = pd.read_csv(z.open(f1), parse_dates=['date']).set_index('date').sort_index()
        alt = pd.read_csv(z.open(f2))
    alt['date'] = pd.to_datetime(alt['date'], utc=True).dt.tz_convert(None)
    alt = alt.set_index('date').sort_index()
    return fast, alt


def perf(r: pd.Series) -> dict:
    r = pd.Series(r).dropna().astype(float)
    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    ann_arith = r.mean() * 52
    vol = r.std(ddof=1) * np.sqrt(52)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(52) if r.std(ddof=1) > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(52)
    sortino = r.mean() * 52 / downside if downside > 0 else np.nan
    dd = eq / eq.cummax() - 1
    ulcer = np.sqrt(np.mean((dd * 100) ** 2))
    q = r.quantile(0.05)
    cvar = r[r <= q].mean()
    return {
        'weeks': len(r), 'CAGR': cagr, 'ann_arithmetic_return': ann_arith,
        'vol': vol, 'Sharpe': sharpe, 'Sortino_downside_RMS': sortino,
        'maxDD': dd.min(), 'Ulcer': ulcer, 'CVaR5_weekly': cvar,
        'end_multiple': eq.iloc[-1],
    }


def perf_arr(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float)
    eq = np.cumprod(1 + r)
    cagr = eq[-1] ** (52 / len(r)) - 1
    vol = np.std(r, ddof=1) * np.sqrt(52)
    sharpe = np.mean(r) / np.std(r, ddof=1) * np.sqrt(52)
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(52)
    sortino = np.mean(r) * 52 / downside if downside > 0 else np.nan
    dd = eq / np.maximum.accumulate(eq) - 1
    ulcer = np.sqrt(np.mean((dd * 100) ** 2))
    q = np.quantile(r, 0.05)
    cvar = np.mean(r[r <= q])
    return np.asarray([cagr, vol, sharpe, sortino, dd.min(), ulcer, cvar])


def paired_block_bootstrap(base: pd.Series, alt: pd.Series, B: int = 3000, block: int = 26, seed: int = 123) -> pd.DataFrame:
    x = pd.concat([base.rename('base'), alt.rename('alt')], axis=1).dropna().to_numpy(float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    d = np.empty((B, 7))
    for b in range(B):
        starts = rng.integers(0, n, size=nb)
        ix = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        d[b] = perf_arr(x[ix, 1]) - perf_arr(x[ix, 0])
    names = ['CAGR', 'vol', 'Sharpe', 'Sortino', 'maxDD', 'Ulcer', 'CVaR5_weekly']
    rows = []
    for i, name in enumerate(names):
        lower_better = name in ['vol', 'Ulcer']
        improve = d[:, i] < 0 if lower_better else d[:, i] > 0
        rows.append({
            'metric': name,
            'median_diff': np.median(d[:, i]),
            'ci025': np.quantile(d[:, i], 0.025),
            'ci975': np.quantile(d[:, i], 0.975),
            'prob_improve': improve.mean(),
        })
    return pd.DataFrame(rows)


def drawdown_episodes(ret: pd.Series, min_dd: float = -0.04) -> list[dict]:
    eq = (1 + ret.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    out = []
    active = False
    for d, v in dd.items():
        if not active and v < 0:
            active = True
            start = trough = d
            trough_dd = v
        elif active:
            if v < trough_dd:
                trough = d
                trough_dd = v
            if v >= -1e-12:
                if trough_dd <= min_dd:
                    out.append({'start': start, 'trough': trough, 'recovery': d, 'maxDD': trough_dd})
                active = False
    if active and trough_dd <= min_dd:
        out.append({'start': start, 'trough': trough, 'recovery': dd.index[-1], 'maxDD': trough_dd})
    return sorted(out, key=lambda z: z['maxDD'])


def worst_block(r: pd.Series, block: int = 26):
    v = r.to_numpy(float)
    best = None
    for i in range(len(v) - block + 1):
        x = v[i:i+block]
        cr = np.prod(1 + x) - 1
        if best is None or cr < best[0]:
            best = (cr, i, x, r.index[i], r.index[i+block-1])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--equity-phase6-data', required=True)
    ap.add_argument('--fx-data-package', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bootstrap', type=int, default=3000)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    eq = load_equity(Path(args.equity_phase6_data))
    fast, alt = load_fx(Path(args.fx_data_package))

    panel = eq.join(fast[['net_portfolio_return']].rename(columns={'net_portfolio_return':'FAST'}), how='inner')
    panel = panel.join(alt[['FX_65FAST_35ALT']], how='left')
    panel = panel.dropna(subset=['FAST', 'FX_65FAST_35ALT']).copy()

    for c in CORRIDOR_WEIGHTS:
        tag = f'EQ_C{int(round(c*100)):02d}'
        panel[tag] = (1-c)*panel['preferred_gross'] + c*panel['corridor_accept'] + 0.20*panel['put_overlay_additive']
        panel[f'ALPHA_C{int(round(c*100)):02d}'] = (1-c)*panel['preferred_gross'] + c*panel['corridor_accept']

    panel.to_csv(out/'01_aligned_weekly_panel.csv')

    comp = ['preferred_gross','put_overlay_additive','corridor_accept','EQ_C20','FAST','FX_65FAST_35ALT']
    panel[comp].corr().to_csv(out/'02_component_correlations.csv')

    rows=[]
    masks={
        'FULL': np.ones(len(panel),dtype=bool),
        '2015_2019': panel.index<'2020-01-01',
        '2020_PLUS': panel.index>='2020-01-01',
        '2022_PLUS': panel.index>='2022-01-01',
    }
    for p,mask in masks.items():
        z=panel.loc[mask]
        for fxname in ['FAST','FX_65FAST_35ALT']:
            rows.append({'period':p,'fx_stream':fxname,'weeks':len(z),'corr_equity_C20':z['EQ_C20'].corr(z[fxname]),
                         'corr_preferred_gross':z['preferred_gross'].corr(z[fxname]),
                         'corr_corridor_accept':z['corridor_accept'].corr(z[fxname]),
                         'corr_put_overlay':z['put_overlay_additive'].corr(z[fxname])})
    pd.DataFrame(rows).to_csv(out/'03_correlation_chronology.csv',index=False)

    rows=[]
    for c in CORRIDOR_WEIGHTS:
        tag=f'EQ_C{int(round(c*100)):02d}'
        rows.append({'corridor_weight':c, **perf(panel[tag])})
    pd.DataFrame(rows).to_csv(out/'04_equity_corridor_frontier.csv',index=False)

    for fxname in ['FAST','FX_65FAST_35ALT']:
        rows=[]
        for c in CORRIDOR_WEIGHTS:
            e=panel[f'EQ_C{int(round(c*100)):02d}']
            for w in FX_CAPITAL_WEIGHTS:
                for L in FX_LEVERAGES:
                    r=(1-w)*e + w*L*panel[fxname]
                    rows.append({'fx_stream':fxname,'corridor_weight':c,'fx_capital_weight':w,'equity_capital_weight':1-w,
                                 'fx_sleeve_leverage':L,'gross_account_exposure_ex_put':(1-w)+w*L,
                                 'put_overlay_account_scale':(1-w)*0.20, **perf(r)})
        df=pd.DataFrame(rows)
        name='05_combined_fixed_grid_FAST.csv' if fxname=='FAST' else '06_combined_fixed_grid_ALT_reference.csv'
        df.to_csv(out/name,index=False)

    candidates={
        'EQUITY_C20': panel['EQ_C20'],
        'PRODUCTION_BASE_FAST_50EQ_50FX_1X': 0.5*panel['EQ_C20'] + 0.5*panel['FAST'],
        'BALANCED_FAST_60EQ_40FX_1P5X': 0.6*panel['EQ_C20'] + 0.4*1.5*panel['FAST'],
        'GROWTH_FAST_2TO1_EQFX_2X': (2/3)*panel['EQ_C20'] + (1/3)*2.0*panel['FAST'],
        'ALT_REFERENCE_50EQ_50FX_1X': 0.5*panel['EQ_C20'] + 0.5*panel['FX_65FAST_35ALT'],
        'ALT_REFERENCE_50EQ_50FX_2X': 0.5*panel['EQ_C20'] + 0.5*2.0*panel['FX_65FAST_35ALT'],
    }
    pd.DataFrame(candidates,index=panel.index).to_csv(out/'07_candidate_weekly_returns.csv')
    rows=[]
    for name,r in candidates.items():
        rows.append({'candidate':name,'sample':'FULL',**perf(r)})
        for p,mask in masks.items():
            if p=='FULL': continue
            rows.append({'candidate':name,'sample':p,**perf(r.loc[mask])})
    pd.DataFrame(rows).to_csv(out/'08_candidate_metrics_chronology.csv',index=False)

    rows=[]
    E=panel['EQ_C20']
    for fxname in ['FAST','FX_65FAST_35ALT']:
        for q in [0.20,0.10,0.05]:
            th=E.quantile(q); g=panel.loc[E<=th,fxname]
            rows.append({'fx_stream':fxname,'worst_equity_quantile':q,'n':len(g),'equity_mean':E[E<=th].mean(),
                         'fx_mean':g.mean(),'fx_positive_share':(g>0).mean()})
    pd.DataFrame(rows).to_csv(out/'09_bad_equity_week_behavior.csv',index=False)

    comparisons={
        'FAST50_C20_vs_EQUITY_C20': (panel['EQ_C20'], candidates['PRODUCTION_BASE_FAST_50EQ_50FX_1X']),
        'FAST50_C20_vs_FAST50_C00': (0.5*panel['EQ_C00']+0.5*panel['FAST'], candidates['PRODUCTION_BASE_FAST_50EQ_50FX_1X']),
        'ALT50_C20_vs_FAST50_C20': (candidates['PRODUCTION_BASE_FAST_50EQ_50FX_1X'], candidates['ALT_REFERENCE_50EQ_50FX_1X']),
    }
    outs=[]
    for name,(a,b) in comparisons.items():
        x=paired_block_bootstrap(a,b,B=args.bootstrap)
        x.insert(0,'comparison',name)
        outs.append(x)
    pd.concat(outs,ignore_index=True).to_csv(out/'10_paired_block_bootstrap.csv',index=False)

    rows=[]
    E=panel['EQ_C20']; eb=worst_block(E)
    for fxname in ['FAST','FX_65FAST_35ALT']:
        fb=worst_block(panel[fxname])
        es=np.sort(eb[2]); fs=np.sort(fb[2])
        for label,w,L in [('PRODUCTION_BASE',0.5,1.0),('BALANCED',0.4,1.5),('GROWTH',1/3,2.0),('FX50_2X',0.5,2.0),('FX50_2P5X',0.5,2.5),('FX50_3X',0.5,3.0)]:
            rr=(1-w)*es+w*L*fs
            m=perf_arr(rr)
            rows.append({'fx_stream':fxname,'candidate':label,'equity_worst_block_return':eb[0],
                         'equity_worst_block_start':eb[3],'equity_worst_block_end':eb[4],
                         'fx_worst_block_return':fb[0],'fx_worst_block_start':fb[3],'fx_worst_block_end':fb[4],
                         'aligned_26week_compound_return':np.prod(1+rr)-1,'aligned_26week_maxDD':m[4]})
    pd.DataFrame(rows).to_csv(out/'11_adversarial_26week_stress.csv',index=False)

    rows=[]
    alpha=panel['ALPHA_C20']; put=panel['put_overlay_additive']
    for fxname in ['FAST','FX_65FAST_35ALT']:
        for label,w,L in [('PRODUCTION_BASE',0.5,1.0),('BALANCED',0.4,1.5),('GROWTH',1/3,2.0),('FX50_2X',0.5,2.0)]:
            scaled=(1-w)*(alpha+0.20*put)+w*L*panel[fxname]
            fixed=(1-w)*alpha+0.20*put+w*L*panel[fxname]
            for convention,r in [('sleeve_scaled_20pct',scaled),('account_fixed_20pct',fixed)]:
                rows.append({'fx_stream':fxname,'candidate':label,'hedge_convention':convention,**perf(r)})
    pd.DataFrame(rows).to_csv(out/'12_hedge_scaling_sensitivity.csv',index=False)

    rows=[]
    for name,r in candidates.items():
        ar=(1+r).groupby(r.index.year).prod()-1
        for y,v in ar.items(): rows.append({'candidate':name,'year':int(y),'return':v})
    pd.DataFrame(rows).to_csv(out/'13_annual_returns.csv',index=False)

    rows=[]
    for name,r in candidates.items():
        for ep in drawdown_episodes(r,-0.04): rows.append({'candidate':name,**ep})
    pd.DataFrame(rows).to_csv(out/'14_drawdown_windows.csv',index=False)

    with zipfile.ZipFile(Path(args.fx_data_package)) as z:
        cname = next(n for n in z.namelist() if n.endswith('/data/fx_fast_currency_exposure.csv'))
        cur = pd.read_csv(z.open(cname), parse_dates=['date'])
    cur = cur[(cur['date']>=panel.index.min()) & (cur['date']<=panel.index.max())].copy()
    weekly_max_currency = cur.groupby('date')['net_currency_exposure_equity'].apply(lambda x: x.abs().max()).reindex(panel.index).fillna(0.0)
    fast_gross = fast['portfolio_gross_leverage_entry'].reindex(panel.index).fillna(0.0)
    fast_ntrades = fast['number_active_trades'].reindex(panel.index).fillna(0.0)
    rows=[]
    fixed = [('PRODUCTION_BASE',0.5,1.0),('BALANCED',0.4,1.5),('GROWTH',1/3,2.0),('FX50_2X',0.5,2.0)]
    for label,w,L in fixed:
        scale=w*L
        fxgross=scale*fast_gross
        totalgross=(1-w)+fxgross
        curr=scale*weekly_max_currency
        rows.append({'candidate':label,'fx_return_scale_w_times_L':scale,
            'fx_gross_notional_median':fxgross.median(),'fx_gross_notional_p95':fxgross.quantile(.95),'fx_gross_notional_max':fxgross.max(),
            'total_gross_ex_put_median':totalgross.median(),'total_gross_ex_put_p95':totalgross.quantile(.95),'total_gross_ex_put_max':totalgross.max(),
            'max_abs_single_currency_median':curr.median(),'max_abs_single_currency_p95':curr.quantile(.95),'max_abs_single_currency_max':curr.max(),
            'share_weeks_currency_gt_0p75':(curr>0.75).mean(),'share_weeks_currency_gt_1p00':(curr>1.00).mean(),
            'active_trades_median':fast_ntrades.median(),'active_trades_p95':fast_ntrades.quantile(.95),'active_trades_max':fast_ntrades.max()})
    pd.DataFrame(rows).to_csv(out/'15_fast_notional_currency_concentration.csv',index=False)

    rows=[]
    for label,w,L in fixed[:3]:
        scale=w*L
        raw_pre=scale*weekly_max_currency
        uncapped=(1-w)*panel['EQ_C20'] + scale*panel['FAST']
        rows.append({'candidate':label,'currency_cap_account_nav':np.nan,'share_weeks_scaled':0.0,'mean_fx_scale':1.0,**perf(uncapped)})
        for cap in [0.75,1.00,1.25,1.50]:
            scaler=pd.Series(np.where(raw_pre>0,np.minimum(1.0,cap/raw_pre),1.0),index=panel.index)
            r=(1-w)*panel['EQ_C20'] + scale*scaler*panel['FAST']
            rows.append({'candidate':label,'currency_cap_account_nav':cap,'share_weeks_scaled':(scaler<0.999999).mean(),'mean_fx_scale':scaler.mean(),**perf(r)})
    pd.DataFrame(rows).to_csv(out/'16_currency_cap_sensitivity.csv',index=False)

    methodology={
        'sample_start': str(panel.index.min().date()), 'sample_end': str(panel.index.max().date()), 'weeks': len(panel),
        'alignment': 'Phase-6 equity signal-week labels are shifted forward 7 days into actual realization-week Friday before exact joining to FAST; FAST is not shifted.',
        'equity_definition': 'Corridor allocation c replaces c of preferred gross alpha; 20% put-overlay additive remains within the equity sleeve.',
        'primary_combination_convention': 'R=(1-w_FX)*R_equity + w_FX*L_FX*R_FX. Only FX sleeve is levered. Put overlay scales with equity sleeve in primary grid.',
        'hedge_sensitivity': 'Separate diagnostic also holds 20% put overlay fixed at account NAV rather than scaling with equity sleeve.',
        'FAST_cost': 'Canonical FAST net research stream: 3 pips round trip + 1 pip stop slippage only if stopped.',
        'Corridor_cost': '2 bp one-way primary research cost.',
        'equity_cost_caveat': 'Preferred Barbell/Agreement/PCA comparator is gross structural; combined CAGR is not a fully net production forecast.',
        'ALT_status': 'Validated 65FAST/35ALT is research reference only; ALT lacks production/Pine-live parity and must not be treated as production-ready.',
        'weight_policy': 'Only fixed Corridor weights 0/10/15/20%, FX capital weights 25/33/40/50%, and leverage 1/1.5/2/2.5/3 were tested; no continuous optimization.',
    }
    (out/'00_methodology.json').write_text(json.dumps(methodology,indent=2,default=str))
    print('completed',out)

if __name__ == '__main__':
    main()
