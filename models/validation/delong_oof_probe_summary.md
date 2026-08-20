# Phase C DeLong on stored OOF probe scores

- Status: `NEW_NON_LOCKED`
- n = 727 (pos=291, neg=436)
- Method: DeLong et al. (1988) paired ROC AUC comparison via structural components / midrank covariance (Sun & Xu form).
- Multiplicity: No multiplicity correction; primary pairs listed explicitly; remaining pairs are secondary.

## sklearn OOF AUCs (sanity)

- latent_probe: 0.576027
- theta_alpha: 0.672893
- packed_2185: 0.699313
- best_nested_head: 0.592854

## Primary paired comparisons

- **latent_probe** vs **packed_2185**: AUC 0.576027 vs 0.699313; Δ=-0.123286; z=-4.639; p=3.50445e-06
- **packed_2185** vs **best_nested_head**: AUC 0.699313 vs 0.592854; Δ=+0.106458; z=4.089; p=4.33923e-05
- **latent_probe** vs **theta_alpha**: AUC 0.576027 vs 0.672893; Δ=-0.096866; z=-3.629; p=0.000284689
- **theta_alpha** vs **best_nested_head**: AUC 0.672893 vs 0.592854; Δ=+0.080039; z=3.317; p=0.000908528

## All pairwise comparisons

- [primary] latent_probe vs packed_2185: z=-4.639, p=3.50445e-06
- [primary] packed_2185 vs best_nested_head: z=4.089, p=4.33923e-05
- [primary] latent_probe vs theta_alpha: z=-3.629, p=0.000284689
- [primary] theta_alpha vs best_nested_head: z=3.317, p=0.000908528
- [secondary] theta_alpha vs packed_2185: z=-1.108, p=0.267696
- [secondary] latent_probe vs best_nested_head: z=-0.806, p=0.42007

## Claim boundary

Non-locked secondary statistics on stored OOF vectors. Does not change locked CV-mean AUCs (0.579 / 0.675 / 0.699) or locked best-head OOF (0.593).
