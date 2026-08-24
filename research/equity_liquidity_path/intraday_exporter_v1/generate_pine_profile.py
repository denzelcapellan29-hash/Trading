#!/usr/bin/env python3
"""
Generate TradingView Pine lower-timeframe exporters for custom RTH timeframes.

The default package ships with a 20-minute RTH profile because Premium Pine
can retrieve up to 100,000 lower-timeframe bars. With ~20 RTH bars/session,
that is ~5,000 sessions (~19.8 trading years).

Examples
--------
python generate_pine_profile.py --minutes 15 --out pine_15m
python generate_pine_profile.py --minutes 10 --out pine_10m
python generate_pine_profile.py --minutes 5 --out pine_5m --footprint-bars 40000
"""
from pathlib import Path
import argparse, math

MAX_PLOTS = 63

def ohlcv_script(title,prefix,tf,start,end,calc,include_count):
    lines=[
        "//@version=6",f'indicator("{title}", overlay = false)',"",
        'string reqTicker = ticker.new(syminfo.prefix, syminfo.ticker, session.regular, adjustment.splits)',
        f'[tA, oA, hA, lA, cA, vA] = request.security_lower_tf(reqTicker, "{tf}", [time, open, high, low, close, volume], calc_bars_count = {calc})',"",
        "f_get_i(array<int> a, int i) =>",
        "    array.size(a) > i ? array.get(a, i) : na",
        "f_get_f(array<float> a, int i) =>",
        "    array.size(a) > i ? array.get(a, i) : na",""
    ]
    if include_count:
        lines += [f'plot(array.size(oA), "{prefix}_count", display = display.data_window)',""]
    for s in range(start,end+1):
        i=s-1; tag=f"{prefix}_S{s:02d}"
        lines += [
            f'plot(float(f_get_i(tA, {i})), "{tag}_time", display = display.data_window)',
            f'plot(f_get_f(oA, {i}), "{tag}_open", display = display.data_window)',
            f'plot(f_get_f(hA, {i}), "{tag}_high", display = display.data_window)',
            f'plot(f_get_f(lA, {i}), "{tag}_low", display = display.data_window)',
            f'plot(f_get_f(cA, {i}), "{tag}_close", display = display.data_window)',
            f'plot(f_get_f(vA, {i}), "{tag}_volume", display = display.data_window)',
        ]
    return "\n".join(lines)+"\n"

def footprint_script(title,prefix,tf,start,end,calc,ticks):
    lines=[
        "//@version=6",f'indicator("{title}", overlay = false)',"",
        f'int ticksPerRow = input.int({ticks}, "Footprint ticks per row", minval = 1)',
        'string reqTicker = ticker.new(syminfo.prefix, syminfo.ticker, session.regular, adjustment.splits)',"",
        "f_fp() =>",
        "    footprint fp = request.footprint(ticksPerRow, 70, 300)",
        "    float buyV = na(fp) ? na : footprint.buy_volume(fp)",
        "    float sellV = na(fp) ? na : footprint.sell_volume(fp)",
        "    float deltaV = na(fp) ? na : footprint.delta(fp)",
        "    [buyV, sellV, deltaV]","",
        f'[buyA, sellA, deltaA] = request.security_lower_tf(reqTicker, "{tf}", f_fp(), calc_bars_count = {calc})',"",
        "f_get_f(array<float> a, int i) =>",
        "    array.size(a) > i ? array.get(a, i) : na",""
    ]
    if start == 1:
        lines += [f'plot(array.size(deltaA), "{prefix}_count", display = display.data_window)',""]
    for s in range(start,end+1):
        i=s-1; tag=f"{prefix}_S{s:02d}"
        lines += [
            f'plot(f_get_f(buyA, {i}), "{tag}_buy", display = display.data_window)',
            f'plot(f_get_f(sellA, {i}), "{tag}_sell", display = display.data_window)',
            f'plot(f_get_f(deltaA, {i}), "{tag}_delta", display = display.data_window)',
        ]
    return "\n".join(lines)+"\n"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--minutes",type=int,default=20)
    ap.add_argument("--out",default="generated_pine")
    ap.add_argument("--ohlcv-bars",type=int,default=100000)
    ap.add_argument("--footprint-bars",type=int,default=60000)
    ap.add_argument("--ticks-per-row",type=int,default=100)
    args=ap.parse_args()

    if args.minutes <= 0 or args.minutes >= 390:
        raise SystemExit("--minutes must be between 1 and 389 for RTH intrabars.")

    slots=math.ceil(390/args.minutes)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    prefix=f"R{args.minutes}"

    max_ohlcv_slots=10
    k=0
    for start in range(1,slots+1,max_ohlcv_slots):
        k+=1; end=min(slots,start+max_ohlcv_slots-1)
        title=f"EQ_INTRADAY_RTH{args.minutes}_OHLCV_{k:02d}"
        (out/f"{title}.pine").write_text(
            ohlcv_script(title,prefix,str(args.minutes),start,end,args.ohlcv_bars,k==1),
            encoding="utf-8"
        )

    max_fp_slots=20
    k=0
    for start in range(1,slots+1,max_fp_slots):
        k+=1; end=min(slots,start+max_fp_slots-1)
        title=f"EQ_INTRADAY_RTH{args.minutes}_FOOTPRINT_{k:02d}"
        (out/f"{title}.pine").write_text(
            footprint_script(title,prefix+"FP",str(args.minutes),start,end,args.footprint_bars,args.ticks_per_row),
            encoding="utf-8"
        )

    print(f"Generated {slots} RTH slots at {args.minutes}m.")
    print(f"Approx Premium OHLCV coverage: {args.ohlcv_bars/slots/252:.2f} trading years.")
    print(f"Approx footprint coverage: {args.footprint_bars/slots/252:.2f} trading years.")

if __name__=="__main__":
    main()
