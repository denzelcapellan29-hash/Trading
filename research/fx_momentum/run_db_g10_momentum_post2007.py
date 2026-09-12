#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

G10 = ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "NOK", "SEK"]
PAIR_MAP = {
    "EUR": ("EURUSD", False),
    "JPY": ("USDJPY", True),
    "GBP": ("GBPUSD", False),
    "CHF": ("USDCHF", True),
    "AUD": ("AUDUSD", False),
    "NZD": ("NZDUSD", False),
    "CAD": ("USDCAD", True),
    "NOK": ("USDNOK", True),
    "SEK": ("USDSEK", True),
}
DB_GUIDE_DATE = pd.Timestamp("2007-10-01", tz="UTC")
DB_INDEX_LAUNCH = pd.Timestamp("2007-03-27", tz="UTC")


def load_pair_2h(sample_zip: Path, pair: str) -> pd.DataFrame:
    with zipfile.ZipFile(sample_zip) as z:
        name = next(n for n in z.namelist() if re.search(fr"OANDA_{pair},", n))
        d = pd.read_csv(io.BytesIO(z.read(name)))
    parts = []
    for s in range(1, 5):
        pr = f"EXPORT S{s} "
        cols = {
            pr + "Start Timestamp": "start",
            pr + "Open": "open",
            pr + "High": "high",
            pr + "Low": "low",
            pr + "Close": "close",
            pr + "Volume": "volume",
            pr + "Complete Flag": "complete",
        }
        c = d[list(cols)].rename(columns=cols)
        c = c[c["complete"].fillna(0).astype(float).eq(1)].copy()
        c["start_time"] = pd.to_datetime(c["start"], unit="ms", utc=True)
        c["close_time"] = c["start_time"] + pd.Timedelta(hours=2)
        parts.append(c[["start_time", "close_time", "open", "high", "low", "close", "volume"]])
    return (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["close_time", "close"])
        .sort_values("close_time")
        .drop_duplicates("close_time")
        .reset_index(drop=True)
    )


def calendar_month_targets(start: pd.Timestamp, end: pd.Timestamp, mode: str) -> pd.DatetimeIndex:
    months = pd.period_range(start.tz_convert(None).to_period("M"), end.tz_convert(None).to_period("M"), freq="M")
    vals = []
    for p in months:
        y, m = p.year, p.month
        if mode == "month_end":
            day = calendar.monthrange(y, m)[1]
        elif mode == "third_wednesday":
            cal = calendar.monthcalendar(y, m)
            weds = [wk[calendar.WEDNESDAY] for wk in cal if wk[calendar.WEDNESDAY] != 0]
            day = weds[2]
        else:
            raise ValueError(mode)
        local = pd.Timestamp(year=y, month=m, day=day, hour=16, tz="Europe/London")
        target = local.tz_convert("UTC")
        if target <= end:
            vals.append(target)
    return pd.DatetimeIndex(vals)


def asof_values(x: pd.DataFrame, targets: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    idx = x["close_time"].astype("int64").to_numpy()
    val = x["close"].astype(float).to_numpy()
    ti = targets.astype("int64").to_numpy()
    pos = np.searchsorted(idx, ti, side="right") - 1
    out = np.full(len(targets), np.nan)
    obs = np.full(len(targets), np.datetime64("NaT"), dtype="datetime64[ns]")
    good = pos >= 0
    out[good] = val[pos[good]]
    obs[good] = x["close_time"].dt.tz_convert(None).to_numpy()[pos[good]]
    return pd.Series(out, index=targets), pd.Series(pd.to_datetime(obs, utc=True), index=targets)


def build_price_panel(sample_zip: Path, mode: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = {}
    coverage = {}
    for ccy, (pair, invert) in PAIR_MAP.items():
        x = load_pair_2h(sample_zip, pair)
        raw[ccy] = (x, invert)
        coverage[ccy] = {
            "pair": pair,
            "invert": invert,
            "first_close": str(x["close_time"].min()),
            "last_close": str(x["close_time"].max()),
            "bars": int(len(x)),
        }
    start = max(x[0]["close_time"].min() for x in raw.values())
    end = min(x[0]["close_time"].max() for x in raw.values())
    targets = calendar_month_targets(start, end, mode)
    prices = pd.DataFrame(index=targets)
    observed = pd.DataFrame(index=targets)
    prices["USD"] = 1.0
    observed["USD"] = targets
    for ccy, (x, invert) in raw.items():
        s, obs = asof_values(x, targets)
        prices[ccy] = 1.0 / s if invert else s
        observed[ccy] = obs
    prices = prices[G10]
    observed = observed[G10]
    return prices, observed, coverage


def run_strategy(prices: pd.DataFrame, cost_bps_oneway: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    mom12 = prices / prices.shift(12) - 1.0
    gross_ccy = prices.shift(-1) / prices
    fwd = gross_ccy - 1.0
    logf = np.log(gross_ccy)
    weights = pd.DataFrame(0.0, index=prices.index, columns=G10)
    rank_rows = []
    rows = []
    for dt in prices.index:
        m = mom12.loc[dt]
        if m.isna().any():
            continue
        ordered = m.sort_values(ascending=False, kind="mergesort")
        top = list(ordered.index[:3])
        bot = list(ordered.index[-3:])
        weights.loc[dt, top] = 1 / 3
        weights.loc[dt, bot] = -1 / 3
        rr = {"signal_time": dt, "top3": "|".join(top), "bottom3": "|".join(bot)}
        rr.update({f"mom12_{c}": float(m[c]) for c in G10})
        rank_rows.append(rr)
        if dt == prices.index[-1] or fwd.loc[dt].isna().any():
            continue
        g = gross_ccy.loc[dt]
        exact_cross = float(np.mean([g[a] / g[b] - 1.0 for a in top for b in bot]))
        linear = float(fwd.loc[dt, top].mean() - fwd.loc[dt, bot].mean())
        log_spread = float(logf.loc[dt, top].mean() - logf.loc[dt, bot].mean())
        top_leg = float(fwd.loc[dt, top].mean())
        short_leg = float(-fwd.loc[dt, bot].mean())
        next_rets = fwd.loc[dt]
        signal_rank = m.rank(method="average")
        next_rank = next_rets.rank(method="average")
        ic = float(signal_rank.corr(next_rank, method="pearson"))
        rows.append({"signal_time":dt,"exact_cross_spot_return":exact_cross,
                     "linear_currency_basket_return":linear,"log_cross_spread":log_spread,
                     "top3_usd_spot_return":top_leg,"bottom3_short_usd_spot_return":short_leg,
                     "cross_sectional_rank_ic":ic})
    base = pd.DataFrame(rows).set_index("signal_time")
    turnover = weights.diff().abs().sum(axis=1)
    active = weights.abs().sum(axis=1).gt(0)
    if active.any():
        first_active = active.idxmax()
        turnover.loc[first_active] = weights.loc[first_active].abs().sum()
    base["turnover_oneway_notional"] = turnover.reindex(base.index)
    base["cost_return"] = base["turnover_oneway_notional"] * cost_bps_oneway / 10000.0
    base["net_spot_return"] = base["exact_cross_spot_return"] - base["cost_return"]
    base["next_signal_time"] = pd.Series(prices.index[1:], index=prices.index[:-1]).reindex(base.index).values
    ranks = pd.DataFrame(rank_rows).set_index("signal_time")
    return base, ranks

def metrics(r: pd.Series) -> dict:
    r = r.dropna().astype(float)
    n = len(r)
    if n == 0:
        return {"n_months": 0}
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    years = n / 12.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 and years > 0 else np.nan
    ann_arith = r.mean() * 12
    ann_vol = r.std(ddof=1) * math.sqrt(12) if n > 1 else np.nan
    sharpe = r.mean() / r.std(ddof=1) * math.sqrt(12) if n > 1 and r.std(ddof=1) > 0 else np.nan
    downside = math.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    sortino = r.mean() / downside * math.sqrt(12) if downside > 0 else np.nan
    ulcer = math.sqrt(np.mean((100 * dd.to_numpy()) ** 2)) / 100.0
    return {
        "n_months": n,
        "start": str(r.index.min()),
        "end": str(r.index.max()),
        "cagr": float(cagr),
        "ann_arithmetic_return": float(ann_arith),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd.min()),
        "ulcer_index": float(ulcer),
        "hit_rate": float((r > 0).mean()),
        "skew_monthly": float(r.skew()),
        "best_month": float(r.max()),
        "worst_month": float(r.min()),
        "terminal_growth_1": float(eq.iloc[-1]),
    }


def subset_by_signal(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    x = df
    if start:
        x = x[x.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        x = x[x.index < pd.Timestamp(end, tz="UTC")]
    return x


def period_table(df: pd.DataFrame, return_col: str) -> pd.DataFrame:
    periods = [
        ("pre_guide_available_2005_to_2006", "2005-01-01", "2007-01-01"),
        ("post_index_launch_2007_03plus", "2007-03-27", None),
        ("post_guide_2007_10plus", "2007-10-01", None),
        ("2008_2009", "2008-01-01", "2010-01-01"),
        ("2010_2016", "2010-01-01", "2017-01-01"),
        ("2017_2021", "2017-01-01", "2022-01-01"),
        ("2022_2026", "2022-01-01", None),
        ("2010_plus", "2010-01-01", None),
    ]
    rows = []
    for name, start, end in periods:
        x = subset_by_signal(df, start, end)
        m = metrics(x[return_col])
        m["period"] = name
        rows.append(m)
    cols = ["period"] + [c for c in rows[0] if c != "period"]
    return pd.DataFrame(rows)[cols]


def annual_returns(df: pd.DataFrame, col: str) -> pd.DataFrame:
    x = df[[col]].dropna().copy()
    x["year"] = x.index.year
    return x.groupby("year")[col].apply(lambda s: (1+s).prod()-1).rename("return").reset_index()


def position_stats(ranks: pd.DataFrame, start="2007-10-01") -> pd.DataFrame:
    x = ranks[ranks.index >= pd.Timestamp(start, tz="UTC")]
    rows=[]
    for c in G10:
        top=x["top3"].str.split("|").apply(lambda z: c in z)
        bot=x["bottom3"].str.split("|").apply(lambda z: c in z)
        rows.append({"currency":c,"months_top3":int(top.sum()),"months_bottom3":int(bot.sum()),
                     "pct_top3":float(top.mean()),"pct_bottom3":float(bot.mean()),
                     "pct_selected":float((top|bot).mean())})
    return pd.DataFrame(rows)



def run_horizon_exact(prices: pd.DataFrame, horizon_months: int) -> pd.Series:
    mom = prices / prices.shift(horizon_months) - 1.0
    gross_ccy = prices.shift(-1) / prices
    rows = {}
    for dt in prices.index[:-1]:
        m = mom.loc[dt]
        if m.isna().any() or gross_ccy.loc[dt].isna().any():
            continue
        ordered = m.sort_values(ascending=False, kind="mergesort")
        top = list(ordered.index[:3]); bot = list(ordered.index[-3:])
        g = gross_ccy.loc[dt]
        rows[dt] = float(np.mean([g[a] / g[b] - 1.0 for a in top for b in bot]))
    return pd.Series(rows, name="exact_cross_spot_return").sort_index()


def horizon_surface(prices: pd.DataFrame) -> pd.DataFrame:
    horizons = [1, 3, 6, 9, 12, 18, 24]
    periods = [
        ("post_guide", "2007-10-01", None),
        ("2008_2016", "2008-01-01", "2017-01-01"),
        ("2017_2021", "2017-01-01", "2022-01-01"),
        ("2022_2026", "2022-01-01", None),
    ]
    rows=[]
    for h in horizons:
        s = run_horizon_exact(prices, h)
        df=s.to_frame()
        for name,start,end in periods:
            x=subset_by_signal(df,start,end)["exact_cross_spot_return"]
            m=metrics(x); m.update({"formation_months":h,"period":name}); rows.append(m)
    return pd.DataFrame(rows)


def mechanism_table(primary: pd.DataFrame) -> pd.DataFrame:
    periods=[("post_guide","2007-10-01",None),("2010_2016","2010-01-01","2017-01-01"),
             ("2017_2021","2017-01-01","2022-01-01"),("2022_2026","2022-01-01",None)]
    rows=[]
    for name,start,end in periods:
        x=subset_by_signal(primary,start,end)
        rows.append({
            "period":name,"n_months":len(x),
            "exact_cross_ann_arithmetic":float(x["exact_cross_spot_return"].mean()*12),
            "top3_usd_leg_ann_arithmetic":float(x["top3_usd_spot_return"].mean()*12),
            "bottom3_short_usd_leg_ann_arithmetic":float(x["bottom3_short_usd_spot_return"].mean()*12),
            "mean_cross_sectional_rank_ic":float(x["cross_sectional_rank_ic"].mean()),
            "median_cross_sectional_rank_ic":float(x["cross_sectional_rank_ic"].median()),
            "pct_months_positive_rank_ic":float((x["cross_sectional_rank_ic"]>0).mean()),
            "avg_oneway_turnover":float(x["turnover_oneway_notional"].mean()),
        })
    return pd.DataFrame(rows)


def circular_block_bootstrap_mean(r: pd.Series, block=12, reps=10000, seed=12345) -> dict:
    arr=r.dropna().to_numpy(float); n=len(arr); rng=np.random.default_rng(seed); vals=np.empty(reps)
    for b in range(reps):
        draw=[]
        while len(draw)<n:
            st=int(rng.integers(0,n)); inds=(st+np.arange(block))%n; draw.extend(arr[inds])
        vals[b]=np.mean(draw[:n])*12
    q=np.quantile(vals,[.025,.05,.5,.95,.975])
    return {"block_months":block,"reps":reps,"seed":seed,"ann_mean_sample":float(arr.mean()*12),
            "prob_bootstrap_ann_mean_gt_zero":float(np.mean(vals>0)),
            "ci90_ann_mean":[float(q[1]),float(q[3])],"ci95_ann_mean":[float(q[0]),float(q[4])],
            "bootstrap_median_ann_mean":float(q[2])}


def synthetic_cross_audit(sample_zip: Path, prices: pd.DataFrame) -> pd.DataFrame:
    tests={"EURJPY":prices["EUR"]/prices["JPY"],"GBPJPY":prices["GBP"]/prices["JPY"],
           "AUDJPY":prices["AUD"]/prices["JPY"],"EURGBP":prices["EUR"]/prices["GBP"],
           "EURCAD":prices["EUR"]/prices["CAD"],"EURNOK":prices["EUR"]/prices["NOK"],
           "EURSEK":prices["EUR"]/prices["SEK"]}
    rows=[]
    for pair,syn in tests.items():
        x=load_pair_2h(sample_zip,pair); direct,_=asof_values(x,prices.index)
        d=pd.concat([syn.rename("synthetic"),direct.rename("direct")],axis=1).dropna().pct_change().dropna()
        rows.append({"pair":pair,"n_months":len(d),"monthly_return_corr":float(d.synthetic.corr(d.direct)),
                     "mean_abs_return_diff":float((d.synthetic-d.direct).abs().mean()),
                     "max_abs_return_diff":float((d.synthetic-d.direct).abs().max())})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample-zip", required=True)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    sample=Path(args.sample_zip); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    all_summary=[]
    metadata={"source_zip":str(sample),"db_rule":{
        "universe":G10,"signal":"12-month spot return vs USD","long":"top 3","short":"bottom 3",
        "ranking_frequency":"monthly","guide_calc_time":"4pm London","guide_rebalance_note":"Two 50% tranches during IMM (exact tranche dates not specified in guide)",
        "important_limitation":"This run measures spot P&L only. DB historical excess returns include carry and transaction costs."
    },"timing_conventions":{}}

    for mode in ["month_end","third_wednesday"]:
        prices, observed, coverage = build_price_panel(sample, mode)
        metadata["timing_conventions"][mode]={
            "description":"4pm London target; latest completed 2H bar close at or before target",
            "first_target":str(prices.index.min()),"last_target":str(prices.index.max())}
        metadata["coverage"] = coverage
        prices.to_csv(out/f"db_g10_momentum_{mode}_usd_prices.csv")
        observed.to_csv(out/f"db_g10_momentum_{mode}_observed_timestamps.csv")

        # Gross primary and bps cost sensitivities
        primary, ranks=run_strategy(prices,0.0)
        primary.to_csv(out/f"db_g10_momentum_{mode}_monthly_returns.csv")
        ranks.to_csv(out/f"db_g10_momentum_{mode}_rank_history.csv")
        if mode=="month_end":
            position_stats(ranks).to_csv(out/"db_g10_momentum_position_frequency_postguide.csv",index=False)
            annual_returns(primary,"exact_cross_spot_return").to_csv(out/"db_g10_momentum_yearly_exact_cross_spot_returns.csv",index=False)
            period_table(primary,"exact_cross_spot_return").to_csv(out/"db_g10_momentum_period_metrics_exact_cross_spot.csv",index=False)
            mechanism_table(primary).to_csv(out/"db_g10_momentum_mechanism_diagnostics.csv",index=False)
            horizon_surface(prices).to_csv(out/"db_g10_momentum_exploratory_horizon_surface.csv",index=False)
            synthetic_cross_audit(sample,prices).to_csv(out/"db_g10_momentum_synthetic_cross_data_audit.csv",index=False)
            boot=circular_block_bootstrap_mean(subset_by_signal(primary,"2007-10-01",None)["exact_cross_spot_return"])
            (out/"db_g10_momentum_postguide_block_bootstrap.json").write_text(json.dumps(boot,indent=2),encoding="utf-8")

        for bps in [0.0,1.0,2.0,5.0]:
            ret,_=run_strategy(prices,bps)
            post=subset_by_signal(ret,"2007-10-01",None)
            m=metrics(post["net_spot_return"])
            m.update({"timing":mode,"oneway_cost_bps":bps,
                      "avg_monthly_turnover":float(post["turnover_oneway_notional"].mean())})
            all_summary.append(m)

    summ=pd.DataFrame(all_summary)
    summ.to_csv(out/"db_g10_momentum_postguide_timing_cost_sensitivity.csv",index=False)
    (out/"db_g10_momentum_run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(summ.to_string(index=False))

if __name__=="__main__":
    main()
