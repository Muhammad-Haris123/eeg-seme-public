# Unconstrained external battery

- Checkpoint: `C:\Users\UC\Desktop\my_fyp\models\checkpoints_unconstrained\checkpoint_unconstrained.pt`
- Direction reference: `C:\Users\UC\Desktop\my_fyp\models\simulations_constrained_full`

- **tuh**: direction 3/10 (r=-0.3086687370992229); magnitude p=0.12432822263360348 d=-0.0360687877210067
- **osf**: direction 4/10 (r=-0.33106121182675713); magnitude p=0.12447622333955541 d=-0.15978018569312052
- **padic**: direction 4/10 (r=-0.3269956000179636); magnitude p=0.7906762240689815 d=0.03256956882706302
- **caueeg**: direction 4/10 (r=-0.35602492522427); magnitude p=0.08811415201191262 d=-0.07058833141870934
  - latent probe AUC=0.539349502298423; theta/alpha AUC=0.6752999217675048

If constrained >> unconstrained on direction and unconstrained >= constrained on latent AUC: keep trade-off as main result. If both ~10/10 direction: soften constraint-causal direction; emphasize domain-shift preservation + encoding. If both weak latent AUC: soften constraint-compresses-diagnosis.