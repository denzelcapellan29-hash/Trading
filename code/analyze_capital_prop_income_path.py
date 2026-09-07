#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

WPY = 365.25/7.0
PANEL = 'combined_equity_fx_long_no_put_2026-08-29/03_2013plus_aligned_corridor_fx.csv'


def load_panel(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        df = pd.read_csv(zf.open(PANEL))
    df = df.rename(columns={df.columns[0]:'date'})
    df['date'] = pd.to_datetime(df['date'])
    df['OWNED_COMBINED'] = 0.5*df['EQ_C20'] + 0.5*df['FAST']
    return df


def edge_adjust(r: np.ndarray, frac: float) -> np.ndarray:
    lr = np.log1p(np.asarray(r,float))
    return np.expm1((lr-lr.mean()) + frac*lr.mean())


def hist_metrics(r):
    r=np.asarray(r,float)
    eq=np.cumprod(1+r)
    dd=eq/np.maximum.accumulate(eq)-1
    return {
        'weeks':len(r),
        'cagr':float(eq[-1]**(WPY/len(r))-1),
        'ann_vol':float(np.std(r,ddof=1)*math.sqrt(WPY)),
        'ann_arith':float(np.mean(r)*WPY),
        'sharpe':float(np.mean(r)/np.std(r,ddof=1)*math.sqrt(WPY)),
        'max_dd':float(dd.min()),
        'worst_week':float(r.min()),
        'best_week':float(r.max()),
    }


def generate_indices(n_hist:int, n_paths:int, years:int, block:int, seed:int):
    rng=np.random.default_rng(seed)
    T=int(round(years*WPY))
    idx=np.empty((T,n_paths),dtype=np.uint16)
    t=0
    while t<T:
        starts=rng.integers(0,n_hist,size=n_paths,dtype=np.uint16)
        m=min(block,T-t)
        for j in range(m):
            idx[t+j]=(starts+j)%n_hist
        t+=m
    return idx


def simulate_joint(idx, owned_hist, fast_hist, *, edge_fraction, monthly_contrib,
                   prop_allocation, prop_risk_scale, reward_split, fee_per_100k,
                   inflation, spend_levels, withdrawal_rate=0.04):
    T,N=idx.shape
    owned_ret=edge_adjust(owned_hist,edge_fraction)
    fast_ret=edge_adjust(fast_hist,edge_fraction)
    wealth=np.full(N,10000.0,dtype=np.float64)
    weekly_contrib=monthly_contrib*12.0/WPY
    target1=1.10; target2=1.05; floor=0.90
    units=prop_allocation/100000.0
    prop_enabled=units>0
    state=np.full(N,-1 if prop_enabled else -9,dtype=np.int8) # -1 waiting fee,0 phase1,1 phase2,2 funded
    pbal=np.ones(N,dtype=np.float64)
    fee_debt=np.full(N,fee_per_100k*units if prop_enabled else 0.0)
    prop_fee_incurred=np.full(N,fee_per_100k*units if prop_enabled else 0.0)
    first_funded=np.full(N,-1,dtype=np.int32)
    funded_failures=np.zeros(N,dtype=np.int16)
    challenge_restarts=np.zeros(N,dtype=np.int16)
    total_prop_payout=np.zeros(N,dtype=np.float64)
    total_prop_fee=np.full(N,fee_per_100k*units if prop_enabled else 0.0)
    # rolling 52w net prop cash events
    rollbuf=np.zeros((52,N),dtype=np.float32) if prop_enabled else None
    rolling_prop=np.zeros(N,dtype=np.float64)
    # milestones
    millionaire=np.full(N,-1,dtype=np.int32)
    spend_levels=list(spend_levels)
    owned_gate={s:np.full(N,-1,dtype=np.int16) for s in spend_levels}
    hybrid_gate={s:np.full(N,-1,dtype=np.int16) for s in spend_levels}
    owned_streak={s:np.zeros(N,dtype=np.int8) for s in spend_levels}
    hybrid_streak={s:np.zeros(N,dtype=np.int8) for s in spend_levels}

    for t in range(T):
        # portfolio return first
        rr=owned_ret[idx[t]]
        wealth*=1+rr
        prop_cash_event=np.zeros(N,dtype=np.float64)

        if prop_enabled:
            active=state>=0
            if np.any(active):
                pbal[active]*=1 + prop_risk_scale*fast_ret[idx[t,active]]
                # fail any active state at weekly close (daily/intraday rule unavailable)
                fail=active & (pbal<=floor)
                if np.any(fail):
                    was_funded=fail & (state==2)
                    funded_failures[was_funded]+=1
                    challenge_restarts[fail]+=1
                    newfee=fee_per_100k*units
                    fee_debt[fail]+=newfee
                    prop_fee_incurred[fail]+=newfee
                    total_prop_fee[fail]+=newfee
                    prop_cash_event[fail]-=newfee
                    state[fail]=-1
                    pbal[fail]=1.0

                p1=(state==0)&(pbal>=target1)
                if np.any(p1):
                    state[p1]=1; pbal[p1]=1.0
                p2=(state==1)&(pbal>=target2)
                if np.any(p2):
                    state[p2]=2; pbal[p2]=1.0
                    new=(first_funded<0)&p2
                    first_funded[new]=t+1

                if (t+1)%4==0:
                    elig=(state==2)&(pbal>1.0)
                    if np.any(elig):
                        gross=(pbal[elig]-1.0)*prop_allocation
                        payout=gross*reward_split
                        total_prop_payout[elig]+=payout
                        prop_cash_event[elig]+=payout
                        pbal[elig]=1.0

        # external contribution cash plus prop payout; pay any outstanding prop fee debt first
        cash=np.full(N,weekly_contrib,dtype=np.float64)
        if prop_enabled:
            payout_only=np.maximum(prop_cash_event,0.0)
            cash += payout_only
            pay=np.minimum(cash,fee_debt)
            fee_debt-=pay
            cash-=pay
            # once fee fully paid, start/restart phase 1
            start=(state==-1)&(fee_debt<=1e-9)
            state[start]=0; pbal[start]=1.0
        wealth += cash

        # rolling prop cash for income test is economic prop cash: payouts minus fees incurred
        if prop_enabled:
            slot=t%52
            rolling_prop -= rollbuf[slot]
            rollbuf[slot]=prop_cash_event.astype(np.float32)
            rolling_prop += rollbuf[slot]

        newm=(millionaire<0)&(wealth>=1_000_000)
        millionaire[newm]=t+1

        # annual checkpoint; require two consecutive years above threshold
        if (t+1)%round(WPY)==0:
            y=(t+1)/WPY
            inflation_factor=(1+inflation)**y
            for s in spend_levels:
                spend_nom=s*inflation_factor
                needed=spend_nom/withdrawal_rate
                ok=wealth>=needed
                owned_streak[s]=np.where(ok,owned_streak[s]+1,0)
                hit=(owned_gate[s]<0)&(owned_streak[s]>=2)
                owned_gate[s][hit]=int(round(y))
                hybrid_income=withdrawal_rate*wealth + (rolling_prop if prop_enabled else 0.0)
                hok=hybrid_income>=spend_nom
                hybrid_streak[s]=np.where(hok,hybrid_streak[s]+1,0)
                hhit=(hybrid_gate[s]<0)&(hybrid_streak[s]>=2)
                hybrid_gate[s][hhit]=int(round(y))

    return {
        'wealth':wealth, 'millionaire_week':millionaire,
        'first_funded_week':first_funded,
        'funded_failures':funded_failures,
        'challenge_restarts':challenge_restarts,
        'total_prop_payout':total_prop_payout,
        'total_prop_fee':total_prop_fee,
        'owned_gate':owned_gate,'hybrid_gate':hybrid_gate,
    }


def summarize_hits(arr, horizons=(5,10,12,15,20,25,30)):
    arr=np.asarray(arr)
    good=arr>0
    out={'hit_prob_horizon':float(np.mean(good))}
    if np.any(good):
        out['median_year']=float(np.median(arr[good])/WPY) if arr.dtype.kind not in 'iu' or arr.max()>100 else float(np.median(arr[good]))
        out['p25_year']=float(np.quantile(arr[good],.25)/WPY) if arr.max()>100 else float(np.quantile(arr[good],.25))
        out['p75_year']=float(np.quantile(arr[good],.75)/WPY) if arr.max()>100 else float(np.quantile(arr[good],.75))
    else:
        out['median_year']=out['p25_year']=out['p75_year']=np.nan
    for h in horizons:
        cutoff=h*WPY if arr.max()>100 else h
        out[f'by_{h}y']=float(np.mean(good & (arr<=cutoff)))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-zip',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--paths',type=int,default=50000)
    ap.add_argument('--years',type=int,default=35)
    ap.add_argument('--block-weeks',type=int,default=26)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

    df=load_panel(args.input_zip)
    owned=df['OWNED_COMBINED'].to_numpy(float); fast=df['FAST'].to_numpy(float)
    idx=generate_indices(len(df),args.paths,args.years,args.block_weeks,20260906)
    spend=(48000,72000,96000,120000)
    common=dict(monthly_contrib=1000.0,reward_split=.80,fee_per_100k=499.0,inflation=.025,spend_levels=spend,withdrawal_rate=.04)
    scenarios=[
        ('NoProp_Historical',1.00,0,0.0),
        ('NoProp_75pctEdge',.75,0,0.0),
        ('Prop100k_0p4_Historical',1.00,100000,.40),
        ('Prop100k_0p4_75pctEdge',.75,100000,.40),
        ('Prop400k_0p4_Historical',1.00,400000,.40),
        ('Prop400k_0p4_75pctEdge',.75,400000,.40),
        ('Prop400k_0p5_75pctEdge',.75,400000,.50),
    ]
    rows=[]; income_rows=[]; prop_rows=[]
    for name,edge,alloc,scale in scenarios:
        sim=simulate_joint(idx,owned,fast,edge_fraction=edge,prop_allocation=alloc,prop_risk_scale=scale,**common)
        sm=summarize_hits(sim['millionaire_week'])
        row={'scenario':name,'edge_fraction':edge,'prop_allocation':alloc,'prop_risk_scale':scale,**sm,
             'median_final_wealth':float(np.median(sim['wealth'])),
             'p10_final_wealth':float(np.quantile(sim['wealth'],.10)),
             'p90_final_wealth':float(np.quantile(sim['wealth'],.90))}
        rows.append(row)
        for s in spend:
            og=summarize_hits(sim['owned_gate'][s])
            hg=summarize_hits(sim['hybrid_gate'][s])
            income_rows.append({'scenario':name,'spend_today':s,'method':'owned_only_4pct','inflation':.025,'two_year_confirmation':True,**og})
            income_rows.append({'scenario':name,'spend_today':s,'method':'hybrid_4pct_plus_prop_trailing12m','inflation':.025,'two_year_confirmation':True,**hg})
        if alloc>0:
            ff=sim['first_funded_week']; good=ff>0
            prop_rows.append({
                'scenario':name,'prop_allocation':alloc,'risk_scale':scale,'edge_fraction':edge,
                'funded_by_1y':float(np.mean(good&(ff<=1*WPY))),
                'funded_by_2y':float(np.mean(good&(ff<=2*WPY))),
                'funded_by_3y':float(np.mean(good&(ff<=3*WPY))),
                'median_first_funded_year':float(np.median(ff[good])/WPY) if np.any(good) else np.nan,
                'median_total_prop_payout':float(np.median(sim['total_prop_payout'])),
                'median_total_prop_fees':float(np.median(sim['total_prop_fee'])),
                'median_net_prop_cash_per_year':float(np.median(sim['total_prop_payout']-sim['total_prop_fee'])/args.years),
                'prob_any_funded_failure':float(np.mean(sim['funded_failures']>0)),
                'median_challenge_restarts':float(np.median(sim['challenge_restarts'])),
            })
    paths=pd.DataFrame(rows); income=pd.DataFrame(income_rows); props=pd.DataFrame(prop_rows)
    paths.to_csv(args.output_dir/'capital_path_summary.csv',index=False)
    income.to_csv(args.output_dir/'income_independence_summary.csv',index=False)
    props.to_csv(args.output_dir/'prop_lifecycle_summary.csv',index=False)

    # prop risk geometry and simple steady-state payout capacity
    prop_sens=[]
    max_planned_risk=0.08 # canonical FAST weekly planned-risk cap from weekly return file / handoff
    for scale in (.30,.40,.50,.60,.75,1.0):
        rr=fast*scale; hm=hist_metrics(rr)
        for split in (.80,.90):
            prop_sens.append({
                'risk_scale':scale,'reward_split':split,**hm,
                'scaled_max_planned_weekly_stop_risk':max_planned_risk*scale,
                'expected_reward_100k_from_arith_mean':hm['ann_arith']*100000*split,
                'expected_reward_400k_from_arith_mean':hm['ann_arith']*400000*split,
                'four_month_expected_geometric_return':(1+hm['cagr'])**(1/3)-1,
            })
    ps=pd.DataFrame(prop_sens); ps.to_csv(args.output_dir/'prop_risk_reward_sensitivity.csv',index=False)

    # deterministic contribution reference at historical/75% edge
    det=[]
    for frac in (1.0,.75,.5):
        cagr=hist_metrics(edge_adjust(owned,frac))['cagr']
        rm=(1+cagr)**(1/12)-1
        for add in (1000,1500,2000,3000,4000):
            bal=10000
            months=None
            for m in range(1,601):
                bal=bal*(1+rm)+add
                if bal>=1_000_000:
                    months=m;break
            det.append({'edge_fraction':frac,'annual_cagr':cagr,'monthly_total_cash_in':add,'years_to_1m':months/12 if months else np.nan})
    pd.DataFrame(det).to_csv(args.output_dir/'deterministic_cashflow_sensitivity.csv',index=False)

    provenance={
        'date':'2026-09-06','source_zip':args.input_zip.name,'source_panel':PANEL,
        'sample_start':str(df.date.min().date()),'sample_end':str(df.date.max().date()),'weeks':len(df),
        'paths':args.paths,'years':args.years,'block_weeks':args.block_weeks,'starting_owned_capital':10000,
        'monthly_contribution':1000,'inflation':.025,'withdrawal_rate':.04,'two_year_income_confirmation':True,
        'prop_reference':'FTMO 2-Step current public rules as of 2026-09-06',
        'prop_rules_used':{'phase1_target':.10,'phase2_target':.05,'max_total_loss':.10,'reward_split':.80,'fee_per_100k':499,'max_pre_scaling_allocation':400000},
        'prop_limitations':'Only weekly-close max-loss proxy is simulated. FTMO maximum daily loss and intraday equity breaches cannot be validated from weekly returns. Prop results are therefore conditional research estimates, not eligibility certification.',
        'historical_owned':hist_metrics(owned),'historical_fast':hist_metrics(fast),
    }
    (args.output_dir/'methodology.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')

    # concise markdown report generated from result tables
    p=paths.set_index('scenario'); inc=income.set_index(['scenario','spend_today','method']); pr=props.set_index('scenario')
    def fyears(x): return 'n/a' if pd.isna(x) else f'{x:.1f}y'
    def pct(x): return f'{100*x:.1f}%'
    lines=[]
    lines += ['# $10k + $1,000/month + Prop Capital — Capital Growth and Income Independence','',
              'Date: 2026-09-06','',
              '## Executive findings','',
              f"- With no prop capital and historical edge retained, the median $1M hit time is **{fyears(p.loc['NoProp_Historical','median_year'])}**; at 75% edge it is **{fyears(p.loc['NoProp_75pctEdge','median_year'])}**.",
              f"- A $100k prop reference account at 0.4x FAST reaches funded status by year 3 in **{pct(pr.loc['Prop100k_0p4_Historical','funded_by_3y'])}** of historical-edge bootstrap paths; median first funding is **{fyears(pr.loc['Prop100k_0p4_Historical','median_first_funded_year'])}**. At 75% edge those become **{pct(pr.loc['Prop100k_0p4_75pctEdge','funded_by_3y'])}** and **{fyears(pr.loc['Prop100k_0p4_75pctEdge','median_first_funded_year'])}**.",
              f"- A $400k pre-scaling prop allocation materially accelerates owned-capital growth only if the accounts are successfully obtained and retained. In the 75%-edge / 0.4x case, median $1M timing is **{fyears(p.loc['Prop400k_0p4_75pctEdge','median_year'])}**, versus **{fyears(p.loc['NoProp_75pctEdge','median_year'])}** without prop.",
              '- Prop notional is never counted as owned wealth. Only net reward cash is allowed to enter the personal account.',
              '', '## Millionaire path — bootstrap results','',
              '| Scenario | Median $1M time | P(hit by 10y) | P(hit by 15y) | P(hit by 20y) |',
              '|---|---:|---:|---:|---:|']
    for name,label in [('NoProp_Historical','No prop — historical edge'),('NoProp_75pctEdge','No prop — 75% edge'),('Prop100k_0p4_Historical','$100k prop 0.4x — historical edge'),('Prop100k_0p4_75pctEdge','$100k prop 0.4x — 75% edge'),('Prop400k_0p4_Historical','$400k prop 0.4x — historical edge'),('Prop400k_0p4_75pctEdge','$400k prop 0.4x — 75% edge'),('Prop400k_0p5_75pctEdge','$400k prop 0.5x — 75% edge')]:
        row=p.loc[name]
        lines.append(f"| {label} | {fyears(row['median_year'])} | {pct(row['by_10y'])} | {pct(row['by_15y'])} | {pct(row['by_20y'])} |")
    lines += ['', '## Living-off-income gates','',
              'Living expenses are expressed in **today\'s dollars** and inflated at 2.5% annually. The owned-only gate requires 4% of owned capital to cover the inflation-adjusted annual spend at **two consecutive annual checkpoints**. The hybrid gate allows trailing 12-month net prop rewards plus 4% of owned capital to cover the same spend for two consecutive years.','',
              '| Spend today | No prop, historical: owned-only | No prop, 75% edge: owned-only | $400k prop 0.4x, 75% edge: hybrid | $400k prop 0.4x, 75% edge: owned-only |',
              '|---:|---:|---:|---:|---:|']
    for s in spend:
        a=inc.loc[('NoProp_Historical',s,'owned_only_4pct'),'median_year']
        b=inc.loc[('NoProp_75pctEdge',s,'owned_only_4pct'),'median_year']
        c=inc.loc[('Prop400k_0p4_75pctEdge',s,'hybrid_4pct_plus_prop_trailing12m'),'median_year']
        d=inc.loc[('Prop400k_0p4_75pctEdge',s,'owned_only_4pct'),'median_year']
        lines.append(f"| ${s/12:,.0f}/mo (${s/1000:.0f}k/yr) | {fyears(a)} | {fyears(b)} | {fyears(c)} | {fyears(d)} |")
    lines += ['', '## Prop risk interpretation','',
              '- The reference firm uses simulated capital and pays real-money rewards; it is not owned account equity.',
              '- The current 2-Step reference rules are a 10% Challenge target, 5% Verification target, 5% maximum daily loss, 10% maximum loss, and 80% base reward ratio.',
              '- At 0.4x FAST, the canonical 8% planned weekly risk cap becomes 3.2%, historical max drawdown is about 5.4%, and worst historical week about 2.4%. At 0.5x these become 4.0%, about 6.8%, and about 3.0%.',
              '- The simulation can test the 10% total-loss rule only at weekly closes. It cannot certify the 5% daily-loss rule; intraday/daily prop-rule replay is a required gate before spending real challenge fees.',
              '- The reference scaling plan requires at least 10% net simulated profit over each prior four-month scaling window. FAST at 0.4x–0.5x historically compounds at roughly 8%–10% annually, so a $2M scaling assumption is excluded from the base case.',
              '', '## Practical interpretation','',
              'The $1,000 monthly contribution is already a powerful accelerant. Prop capital is most valuable as a **cash-flow overlay that is swept into owned capital**, not as a substitute for owned wealth. A mature $400k prop allocation can plausibly contribute tens of thousands of dollars per year if the strategy passes and stays within firm rules, but those payouts are less durable than income supported by owned capital.',
              '', '## Caveats','',
              '- Backtest-based bootstrap results are conditional, not guarantees.',
              '- The equity research stream remains a structural research stream rather than a fully net live forecast.',
              '- Taxes are not modeled; all living-income figures are pre-tax spending equivalents.',
              '- Inflation is assumed at 2.5%.',
              '- Prop daily-loss, intraday equity, symbol availability, execution costs, news restrictions, and platform-specific order semantics require live-rule validation before use.']
    (args.output_dir/'CAPITAL_PROP_INCOME_ANALYSIS_2026-09-06.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__': main()
