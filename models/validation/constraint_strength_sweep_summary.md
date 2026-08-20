# Constraint-strength sweep summary (Plan-to-8.3 B3)

**Artifact:** `models/validation/constraint_strength_sweep.json` (seeded eval timestamp in file)  
**Rule:** fold_0 only (matched to paper external checkpoint selection)  
**Direction reference:** constrained training signature (`simulations_constrained_full`)  
**Sim seed:** 42 (`simulate_flat_cohort`)

## Seeded table

| \(\alpha_c\) | TUH dir | TUH r | P-ADIC dir | Latent AUC mean | Notes |
|---|---|---|---|---|---|
| 0.0 | 3/10 | -0.350 | 4/10 | 0.539 | Seeded unconstrained fold_0 |
| 0.01 | 8/10 | 0.096 | 8/10 | 0.556 | Near-lock signs; low r |
| 0.05 | 10/10 | 0.299 | 10/10 | 0.527 | Signs locked |
| **0.1** | **10/10** | **0.897** | **10/10** | **0.579** | Paper weight; **peak r** |
| 0.2 | 10/10 | 0.220 | 10/10 | 0.514 | Signs locked; r drops |
| 0.5 | 10/10 | 0.412 | 10/10 | 0.578 | Signs locked; r moderate |

## Interpretation (for manuscript)

1. **Direction signs** require nonzero \(\alpha_c\): \(\alpha_c=0\) fails; \(\alpha_c \ge 0.05\) reaches 10/10 on TUH and P-ADIC.
2. **Effect-vector correlation** peaks at the paper weight 0.1 (not monotone in \(\alpha_c\)).
3. **Latent diagnostic AUC** is **non-monotone**; no clean direction↑ / AUC↓ trade-off curve.
4. Keep title **Option A**. Do not claim a causal encoding trade-off from \(\alpha_c\) alone.
5. Hero figure: \(x=\alpha_c\), dual axis direction agreement (and/or r) vs latent AUC, caption states non-monotone AUC honestly.
