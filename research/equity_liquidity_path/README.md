# Equity Liquidity / Node-Path Research

Independent equity market-structure research branch. Frozen equity sleeves (Momentum Barbell, Agreement Reversion, PCA StatArb 8/9/10, and their preferred blends) are benchmarks only and are not used to select or condition events.

Current structural checkpoint: `Equity_Liquidity_Path_Q_Decomposition_v5_2026-08-23`.

Research order:
1. build causal liquidity/node map;
2. validate geometry/path behavior;
3. validate hold vs acceptance resolution;
4. identify coherent event families;
5. only then construct standalone strategies;
6. finally compare with frozen equity portfolios.

## Current structural hierarchy

- **Geometry/distance** selects the actionable node neighborhood.
- **Q-like confluence** describes whether a nearby node tends to resolve cleanly/informatively.
- **Lifecycle / repeated touches** is a separate hold-vs-gateway prior.
- **QR / resolution displacement** grades conditional path quality after the touch resolves.

The working non-optimized Q research label is:

```text
q_score = 1[node_source_diversity >= 2] + 1[node_members >= 3]
```

Broad Q2-vs-Q0 clean-resolution lift is about +1.8 to +2.4 percentage points across TRAIN/VALIDATION/HOLDOUT. In the TRAIN-fixed nearest-distance quartile the lift is about +8.3 to +8.8pp, with path-hit lift +2.5pp TRAIN, +3.5pp validation and +5.25pp holdout. Importantly, high-Q nodes are **less likely to hold**, so Q is not being interpreted as generic support/resistance strength.

Within Q2, fresh 0-1-touch nodes are roughly 20pp more likely to hold than 5+ touch nodes across all periods, confirming depletion/gateway behavior as a separate lifecycle layer.

Q2 + high QR produces roughly 60-64% hold-path hit rates and roughly 65-67% corrected acceptance-path hit rates across chronology.

The corrected anonymous source-bit ablation now includes unresolved/chop. Bits 7/8 are gateway-like but path-informative; bit 3 is hold-oriented but mildly path-negative. No economic labels are assigned until the original source-mask dictionary is recovered.

## Standalone family status

- Equity Corridor Traversal: **advance after exact node prices are restored**.
- Fixed one-week Decision-Node Rotation: **rejected**.
- Daily Compression + simple Flow + Path: **rejected/deferred to intraday data**.

Exact Node1/Node2/Node3 prices, source-family labels, full corrected acceptance targets, and sector robustness remain required before standalone PnL promotion.

Large derived event chunks and v5 output tables are stored in the project Google Drive `data/` folder. The reproducible v5 script is `run_equity_liquidity_q_decomposition_v5.py`.
