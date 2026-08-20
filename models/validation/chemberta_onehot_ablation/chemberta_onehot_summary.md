# Phase C ChemBERTa vs padded one-hot (fold-0 constrained)

- Status: `NEW_NON_LOCKED`
- Locked checkpoint untouched: `C:\Users\UC\Desktop\my_fyp\models\checkpoints_constrained\checkpoint_constrained.pt`
- Protocol: Same fold-0 split, seed 42, alpha_c=0.1, warmup=5, architecture, and epochs; only drug embedding source differs.

## Direction contrast (vs locked training signature)

- **TUH**: ChemBERTa 9/10 (r=0.103610) vs one-hot 10/10 (r=0.176412); Δr=-0.072802
- **OSF**: ChemBERTa 9/10 (r=0.109931) vs one-hot 10/10 (r=0.194115); Δr=-0.084184
- **PADIC**: ChemBERTa 9/10 (r=0.075516) vs one-hot 10/10 (r=0.174428); Δr=-0.098912

## Latent probe contrast (CAUEEG Dementia vs Normal)

- ChemBERTa rematch mean AUC: 0.540
- One-hot mean AUC: 0.559

## Claim boundary

Secondary ablation of chemical identifier vs drug-identity conditioning. Does not claim large-library ChemBERTa generalization. Locked fold-0 ChemBERTa twin remains the primary historical checkpoint.
