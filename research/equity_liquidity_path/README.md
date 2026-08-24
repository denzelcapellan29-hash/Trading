# Equity Liquidity / Node-Path Research

Independent equity market-structure research branch. Frozen equity sleeves (Momentum Barbell, Agreement Reversion, PCA StatArb 8/9/10, and their preferred blends) are benchmarks only and are not used to select or condition events.

Current structural checkpoint: `Equity_Liquidity_Path_Structural_v4_2026-08-23`.

Research order:
1. build causal liquidity/node map;
2. validate geometry/path behavior;
3. validate hold vs acceptance resolution;
4. identify event families;
5. only then construct standalone strategies;
6. finally compare with frozen equity portfolios.

Current findings:
- broad all-node daily map is too dense;
- repeated touches strongly shift nodes from hold/decision behavior toward acceptance/gateway behavior;
- resolution close displacement is a robust QR variable;
- corrected acceptance targets that remain ahead of price still show a stable path edge;
- high-QR corridor holds show node-target traversal but not positive fixed 1/2/3/5-day holding returns;
- naive fixed one-week decision-node rotation is therefore not promoted;
- daily compression plus simple signed-flow strength does not validate as absolute range expansion;
- exact Node1/Node2/Node3 price levels, source-family labels, and sector robustness remain required before standalone PnL promotion.

Large derived event chunks are stored in the project Google Drive `data/` folder rather than GitHub.
