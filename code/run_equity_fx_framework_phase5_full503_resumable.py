from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pandas as pd
import sys

# Reuse the frozen Phase-5 implementation verbatim.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_equity_fx_framework_phase5_standalone as p5


def load_cuts(package):
    with zipfile.ZipFile(package) as z:
        member=[n for n in z.namelist() if n.endswith('qr_train_state_cuts_full503.csv')][0]
        cuts=pd.read_csv(z.open(member))
    return {(r.trust_family,r.resolution):float(r.q67) for r in cuts.itertuples(index=False)}


def build_jobs(raw, mf, inf, phase3, qr_cuts, batches):
    fmap={}
    for r in inf.itertuples(index=False):
        if bool(r.is_duplicate_payload): continue
        f=raw/str(r.file)
        if f.exists(): fmap[int(r.batch)]=f
    jobs=[]
    with zipfile.ZipFile(phase3) as z:
        for b in batches:
            if b not in fmap: continue
            ev=pd.read_csv(z.open(f'events_by_batch/batch_{b:02d}_lifecycle.csv.gz'),compression='gzip')
            ev=ev[ev.near_train_q1 & ev.clean_resolution.eq(1) & ev.trust_family.isin([x[0] for x in p5.PROMOTED])].copy()
            bm=mf[mf.batch.eq(b)].copy()
            jobs.append((b,str(fmap[b]),bm,ev,qr_cuts))
    return jobs


def save_batch(out,b,tr,pa):
    pd.DataFrame(tr).to_csv(out/f'batch_{b:02d}_trades.csv.gz',index=False,compression='gzip')
    pd.DataFrame(pa,columns=['trade_id','time','raw_return']).to_csv(out/f'batch_{b:02d}_paths.csv.gz',index=False,compression='gzip')


def aggregate(out, raw):
    trade_files=sorted(out.glob('batch_*_trades.csv.gz'))
    path_files=sorted(out.glob('batch_*_paths.csv.gz'))
    if len(trade_files)!=42 or len(path_files)!=42:
        raise RuntimeError(f'Need 42 checkpoint pairs, have trades={len(trade_files)} paths={len(path_files)}')
    t=pd.concat([pd.read_csv(f) for f in trade_files],ignore_index=True)
    p=pd.concat([pd.read_csv(f) for f in path_files],ignore_index=True)
    t.to_csv(out/'phase5_promoted_state_candidate_trades.csv.gz',index=False,compression='gzip')
    p.to_csv(out/'phase5_candidate_trade_paths.csv.gz',index=False,compression='gzip')
    ref=next(raw.glob('*.csv'))
    calendar=pd.DatetimeIndex(pd.to_datetime(pd.read_csv(ref,usecols=['time']).time,utc=True).dropna().drop_duplicates().sort_values())
    variants=['PROMOTED_ALL','HIGH_QR','HIGH_QR_INTERACTED_Q2','HIGH_QR_TARGET_Q2','HIGH_QR_BOTH_Q2']
    costs=[0,1,2,5,10]; rows=[]; subrows=[]; kept={}
    for v in variants:
        x=p5.dedupe_and_nonoverlap(t,v); kept[v]=x; x.to_csv(out/f'trades_{v.lower()}.csv',index=False)
        for cost in costs:
            s=p5.pf_and_stats(x,cost); s.update(p5.active_book_metrics(x,p,cost,calendar)); s.update({'variant':v,'one_way_cost_bps':cost,'sample':'ALL'}); rows.append(s)
            for per in p5.PERIODS:
                g=x[x.period.eq(per)]; ss=p5.pf_and_stats(g,cost); ss.update({'variant':v,'one_way_cost_bps':cost,'sample':per}); subrows.append(ss)
    pd.DataFrame(rows).to_csv(out/'portfolio_metrics_all.csv',index=False)
    pd.DataFrame(subrows).to_csv(out/'trade_metrics_by_period.csv',index=False)
    diag=[]; x=kept['HIGH_QR']
    for keys,g in x.groupby(['period','trust_family','resolution','direction']):
        s=p5.pf_and_stats(g,2); s.update(dict(zip(['period','trust_family','resolution','direction'],keys))); diag.append(s)
    pd.DataFrame(diag).to_csv(out/'high_qr_state_direction_diagnostics_2bps.csv',index=False)
    methodology={
        'phase':'Phase 5 standalone structural portfolio full-503 replay', 'status':'FULL 503 DEFINITIVE REPLAY',
        'available_batches':list(range(1,43)),'available_stocks':503,'full_universe':503,
        'entry':'next 78m bar open after observed 0.5 ATR hold/accept resolution',
        'target':'nearest frozen-map node still ahead of resolution-confirmation close','target_distance_atr':[.5,1.5],
        'invalidation':'interacted node +/- 0.5 snapshot ATR opposite realized resolution direction','time_exit':None,
        'same_bar_target_stop':'stop-first conservative','entry_gap_rule':'skip if target already passed or structural invalidation already breached before executable next open',
        'geometry':'TRAIN-fixed nearest first quartile inherited from Phase4','lifecycle':'diagnostic only, no gate',
        'qr':'TRAIN-only trust_family x resolution q67 cuts inherited from Phase4','promoted_qr_states':sorted([list(x) for x in p5.PROMOTED]),
        'q_variants':'Q evaluated separately; not redefined as hold probability',
        'duplicate_control':'latest snapshot for duplicate same ticker/entry/direction; max one open trade per ticker',
        'portfolio':'standardized equal-weight active-book diagnostic, 100% gross when active','costs_one_way_bps':costs,
        'pnl_used_to_choose_structural_parameters':False,
    }
    (out/'PHASE5_METHODOLOGY.json').write_text(json.dumps(methodology,indent=2))
    m=pd.DataFrame(rows); sm=pd.DataFrame(subrows)
    print('\nALL METRICS @ 2BPS')
    print(m[m.one_way_cost_bps.eq(2)][['variant','n','win_rate','mean_trade','profit_factor','mean_bars_held','mean_rr_at_entry','book_cagr','book_vol','book_sharpe','book_sortino','book_max_dd','mean_active_positions_when_active']].to_string(index=False))
    print('\nPERIOD @ 2BPS')
    print(sm[sm.one_way_cost_bps.eq(2)][['variant','sample','n','win_rate','mean_trade','profit_factor']].to_string(index=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-dir',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--inference',required=True)
    ap.add_argument('--phase3-data',required=True); ap.add_argument('--phase4-package',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=min(16,cpu_count())); ap.add_argument('--aggregate-only',action='store_true')
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); raw=Path(a.raw_dir)
    if not a.aggregate_only:
        mf=pd.read_csv(a.manifest); inf=pd.read_csv(a.inference); qr_cuts=load_cuts(a.phase4_package)
        done={int(f.name.split('_')[1]) for f in out.glob('batch_*_trades.csv.gz') if (out/f.name.replace('_trades','_paths')).exists()}
        missing=[b for b in range(1,43) if b not in done]
        print('checkpointed',len(done),'missing',missing,flush=True)
        jobs=build_jobs(raw,mf,inf,a.phase3_data,qr_cuts,missing)
        with Pool(a.workers) as pool:
            for k,(b,tr,pa) in enumerate(pool.imap_unordered(p5.process_batch,jobs),1):
                save_batch(out,b,tr,pa)
                print('checkpoint',k,'/',len(jobs),'batch',b,'trades',len(tr),flush=True)
    aggregate(out,raw)

if __name__=='__main__': main()
