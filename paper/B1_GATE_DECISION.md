# B1 gate decision (unconstrained external battery)

**Date:** 2026-08-09  
**Artifact:** `models/validation/unconstrained_external_battery.json`  
**Checkpoint:** `models/checkpoints_unconstrained/checkpoint_unconstrained.pt` (= `checkpoints_5fold/fold_0_best.pt`)  
**Direction reference:** constrained training signature (`simulations_constrained_full`)

## Pre-registered table outcome

| Pattern | Observed? |
|---------|-----------|
| Constrained ≫ unconstrained on direction | **Yes** (10/10 vs 3/10 TUH, 4/10 OSF, 4/10 P-ADIC, 4/10 CAUEEG) |
| Unconstrained ≥ constrained on latent AUC | **No** (unconst mean AUC **0.539** vs constrained **0.579**) |
| Both weak latent AUC | **Yes** (both below theta/alpha **0.675**) |

## Manuscript consequence (locked for rewrite)

1. **Keep** constraint-consistent direction as a positive, constraint-associated result (matched unconstrained fails the same signature endpoint).
2. **Do not** claim that constraints cause diagnostic compression relative to this matched unconstrained twin (unconstrained latent AUC is lower, not higher).
3. **Frame** encoding gap as: constrained \(\mu_{\mathrm{base}}\) retains partial clinical signal (0.579) below spectral reference (0.675); readout heads do not close the gap (0.593). Treat “constraint↔diagnosis trade-off” as **hypothesis for \(\alpha_c\) sweep**, not as established by B1 alone.
4. **Title:** keep Option A (constraint-consistent direction + diagnostic information loss). Do **not** switch to Option B trade-off title unless \(\alpha_c\) curve shows a clear direction↑ / AUC↓ pattern.
5. **Prior-only:** 10/10 signs vs constrained signature but mean r ≈ 0.636 (Donepezil r=0.572, Memantine r=0.700). Constrained external r ≈ 0.85–0.92 therefore recovers **magnitude structure** beyond the constant prior, not only signs.

## Numbers to cite (source files)

See `paper/citation_ledger.md` updates after this gate.
