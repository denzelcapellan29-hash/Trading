from __future__ import annotations
"""Frozen Equity C20 target composition.

C20 = 80% * (50% Barbell + 25% Agreement + 25% PCA 8/9/10)
    + 20% Corridor ACCEPT
=> 40% Barbell + 20% Agreement + 20% PCA + 20% Corridor.

Inactive sleeve capital stays cash. Overlapping stock targets net algebraically.
"""
from collections import defaultdict
from dataclasses import dataclass

BARBELL_WEIGHT=.40
AGREEMENT_WEIGHT=.20
PCA_WEIGHT=.20
CORRIDOR_WEIGHT=.20

@dataclass(frozen=True)
class EquityC20Target:
    ticker:str
    target_fraction:float

def compose_equity_c20_targets(*,barbell_tickers:list[str],agreement_weights:dict[str,float],pca_weights:dict[str,float],corridor_directions:dict[str,int]|None=None)->list[EquityC20Target]:
    out=defaultdict(float)
    if barbell_tickers:
        w=BARBELL_WEIGHT/len(barbell_tickers)
        for t in barbell_tickers:out[str(t)]+=w
    for t,w in agreement_weights.items():out[str(t)]+=AGREEMENT_WEIGHT*float(w)
    for t,w in pca_weights.items():out[str(t)]+=PCA_WEIGHT*float(w)
    cd={str(t):int(d) for t,d in (corridor_directions or {}).items() if int(d) in (-1,1)}
    if cd:
        w=CORRIDOR_WEIGHT/len(cd)
        for t,d in cd.items():out[t]+=w*d
    return [EquityC20Target(t,float(w)) for t,w in sorted(out.items()) if abs(w)>1e-15]

def diagnostics(targets:list[EquityC20Target])->dict:
    vals=[x.target_fraction for x in targets]
    return {"holding_count":len(vals),"gross_abs_weight":sum(abs(x) for x in vals),"net_weight":sum(vals),"long_weight":sum(x for x in vals if x>0),"short_weight":-sum(x for x in vals if x<0)}
