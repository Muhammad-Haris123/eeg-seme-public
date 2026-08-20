# 2. Methods

This section defines the case-study generator, constraints, cohorts, and the SEME endpoint families (Signed concordance, Effect-vector fidelity, Magnitude transfer, Encoding retention). Paired post-dose EEG is not part of the design. Configuration keys, scripts, and artifact paths are listed in Supplementary Material S1.

## 2.1 Architecture and pharmacodynamic penalties

The twin is a pharmacodynamically constrained CVAE (`constrained_mlp`): MLP encoders, no graph EEG encoder, no cross-attention, constrained loss enabled. Input EEG is a packed 2185-D vector for 19 standard 10-20 channels (PSD \([0,380)\), band powers \([380,475)\), interleaved coherence/PLV connectivity \([475,2185)\)). \(L_{\mathrm{band}}\) uses delta through beta slices; \(L_{\mathrm{conn}}\) uses window \([475,1330)\), which mixes early-band coherence and PLV under the interleaved flatten (S1). Layer widths and parameter count are in Table A.1. Fig. 1 summarizes data flow.

The EEG encoder maps structured tensors to a 128-D EEG latent. The drug encoder maps a frozen 384-D ChemBERTa embedding (`DeepChem/ChemBERTa-77M-MLM`) to a 64-D drug latent; ChemBERTa is not trained. Fusion concatenates EEG latent, drug latent, and a scalar disease label (AD = 1, HC = 0) into a 256-D representation. The CVAE encoder outputs a diagonal Gaussian in 128 dimensions with reparameterization
\[
z = \mu + \exp(\tfrac{1}{2}\log\sigma^2)\odot\varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,I).
\label{eq:reparam}
\]
The decoder reconstructs the 2185-D vector. At inference, baseline uses a zero drug vector; Donepezil/Memantine use their ChemBERTa embeddings. \(\mu_{\mathrm{base}}\) denotes the zero-drug latent mean.

Signed literature priors enter as penalty terms added to reconstruction and KL [Babiloni2006DonepezilEEG, Jeong2004EEGreview, Winblad2007Memantine, Rive2013Memantine]:

- Donepezil: \(\delta-0.15\), \(\theta-0.20\), \(\alpha+0.25\), \(\beta+0.10\), connectivity \(+0.20\)
- Memantine: \(\delta-0.10\), \(\theta-0.15\), \(\alpha+0.10\), \(\beta+0.05\), connectivity \(+0.15\)

With reconstruction \(L_{\mathrm{MSE}}=\mathrm{mean}((\hat{x}-x)^2)\), KL weight \(\beta=0.01\), band ReLU penalties \(L_{\mathrm{band}}\), connectivity non-decrease penalty \(L_{\mathrm{conn}}\) (prior connectivity magnitudes are stored but not multiplied into \(L_{\mathrm{conn}}\)), and schedule \(\alpha_c(t)\) rising from 0 after a 5-epoch warmup to \(\alpha_c^{\star}=0.1\), the total loss is
\[
L = L_{\mathrm{MSE}} + \beta L_{\mathrm{KL}} + \alpha_c(t)\,L_{\mathrm{band}} + \alpha_c(t)\,L_{\mathrm{conn}}.
\label{eq:total}
\]
The KL term against \(\mathcal{N}(0,I)\) is
\[
L_{\mathrm{KL}} = \frac{1}{B}\sum_{i=1}^{B}\Bigl(-\tfrac{1}{2}\sum_{k=1}^{128}\bigl(1+\log\sigma^2_{ik}-\mu_{ik}^2-\exp(\log\sigma^2_{ik})\bigr)\Bigr).
\label{eq:kl}
\]
For band \(b\in\{\delta,\theta,\alpha,\beta\}\) with reconstructed change \(\Delta_b\) versus baseline and prior \(\pi_b\),
\[
L_{\mathrm{band}} = \sum_{b}|\pi_b|\,\mathrm{ReLU}\bigl(-\mathrm{sign}(\pi_b)\,\Delta_b\bigr),
\label{eq:band}
\]
and with connectivity-window change \(\Delta_c\),
\[
L_{\mathrm{conn}} = \mathrm{ReLU}(-\Delta_c).
\label{eq:conn}
\]
The implemented schedule is
\[
\alpha_c(t)=\begin{cases}
0 & t < T_w,\\
\alpha_c^{\star}\min\bigl(1,(t-T_w)/5\bigr) & t \ge T_w,
\end{cases}
\label{eq:alpha}
\]
with \(T_w=5\). Training is summarized in Algorithm 1. Exact index slices and packing notes are in S1.

## 2.2 Training cohort, freeze, and sensitivity protocol

Training features come from resting-state EEG for 66 subjects (AD 37, HC 29; OpenNeuro ds004504). Preprocessing uses FIR 0.5-40 Hz, notch 50 Hz on the training path, FastICA with empty default exclusion, 4 s epochs at 90% overlap, and per-subject z-scoring before Welch/connectivity (S1). Features are 500 Hz spectral/connectivity packs. \(N=66\) is small relative to 2185-D inputs and multilayer capacity; primary claims are therefore external assessments, not training-set diagnosis.

Splits are subject-level stratified 5-fold (`random_state=42`); within-fold validation is 10%. External cohorts are never in the training loop. Optimization uses Adam (\(10^{-4}\), weight decay \(10^{-5}\)), batch size 16, 100 epochs. Checkpoint selection minimizes internal validation total loss. The locked alias is fold-0 seed-42 epoch 99 (val total 0.0241). External \(r\), signs, and AUCs are post-freeze assessments, not selection criteria. The freeze is a documented project rule, not independently auditable preregistration. Training-seed sensitivity holds fold membership fixed (`split_seed=42`) and varies `train_seed` over \(\{42,7,21,123,2024\}\). Compute summary is in Table A.2.

## 2.3 External cohorts

Cohorts were introduced so each answered a doubt left by the previous setting (Table 1).

- **TUH** (\(n=200\); abnormal 131 / normal 69): locked features use `crop_seconds=60.0` and `apply_ica=false` in `processing_report.json`; current script default `CROP_SECONDS=320.0` was not used to regenerate locked results.
- **OSF** AD/HC (\(n=92\); HC \(n=12\)).
- **P-ADIC** AD/HC (\(n=145\)) [Shor2021PADIC].
- **CAUEEG** Normal/MCI/Dementia (1122 features OK) for magnitude and encoding [Kim2023CAUEEG].

Direction tests use TUH, OSF, and P-ADIC. CAUEEG was not required for the locked 10/10 direction claim.

## 2.4 Evaluation endpoints and statistics

No paired pre/post-dose EEG is available. Direction measures concordance with the **training constrained signature**, a prior-shaped model-derived reference. That circularity is intentional and bounded by controls: prior-only (constant literature-signed effect; mean \(r=0.636\), 10/10), matched unconstrained CVAE (\(\alpha_c=0\)), and a ridge imitation control trained on locked CVAE-simulated targets (not observed post-dose EEG).

**Endpoint families (SEME).** (1) **S / E:** Effect-vector Pearson \(r\) (primary continuous score) and block-level signed agreement (10/10 secondary) on cohort-mean drug-minus-baseline vectors. (2) **M:** Magnitude \(\bar{s}=\tfrac12(\|\mu_{\mathrm{don}}-\mu_{\mathrm{base}}\|_2+\|\mu_{\mathrm{mem}}-\mu_{\mathrm{base}}\|_2)\). (3) **E (encoding):** CAUEEG probes on \(\mu_{\mathrm{base}}\) (disease bit fixed to 0) versus theta/alpha and packed 2185-D logistic references; nested heads use outer 5-fold / inner 3-fold search. (4) Exploratory \(\alpha_c\) sweep on fold-0 twins. (5) Paired DeLong tests on stored subject-level OOF probe scores (\(n=727\)) [DeLong1988]. (6) Matched fold-0 ChemBERTa-versus-padded-one-hot ablation (same split, seed 42, \(\alpha_c=0.1\), DrugEncoder input dim 384; non-locked checkpoints only). Uncertainty for \(r\) uses subject-bootstrap percentile intervals (B = 2000). Magnitude uses Mann-Whitney and Cohen's d. Probe permutation tests use \(N_{\mathrm{perm}}=200\). No multiplicity correction was applied across the escalation sequence or DeLong pairs. Full endpoint definitions are in S1.
