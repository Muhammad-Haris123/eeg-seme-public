# 3. Results

Results follow the SEME axes. The primary positive contrast is constrained versus unconstrained signed concordance. Continuous high \(r\) at one locked checkpoint is reported secondarily. Prior-only quantifies the circularity budget for binary signs.

## 3.1 Controls: unconstrained, prior-only, and linear imitation

**Matched unconstrained twin (primary contrast).** On the locked fold-0 comparison against the constrained training signature, unconstrained agreement was 3/10 on TUH (\(r=-0.309\)), 4/10 on OSF (\(r=-0.331\)), and 4/10 on P-ADIC (\(r=-0.327\)). Constrained agreement on the same cohorts is 10/10 (Section 3.2). Thus the CVAE-with-constraint adds signed concordance that the matched unconstrained twin does not deliver under this endpoint. Unconstrained magnitude contrasts remained null or non-diagnostic, and the unconstrained CAUEEG \(\mu_{\mathrm{base}}\) probe reached mean ROC-AUC 0.539 (below constrained 0.579 and below theta/alpha 0.675). Table 2 summarizes the locked direction contrast.

**Prior-only baseline (circularity budget).** A constant literature-signed 2185-D effect, without a CVAE, yielded 10/10 signs at mean \(r=0.636\) (Donepezil 0.572; Memantine 0.700). Binary signs are therefore partly recoverable from the prior alone. SEME treats this as an explicit circularity budget: the CVAE is not claimed to uniquely explain 10/10 signs.

**Linear imitation control.** A ridge map trained on packed baselines, frozen drug embeddings, and disease labels to predict locked CVAE-simulated drug-minus-baseline vectors recovered 10/10 signs on TUH/OSF/P-ADIC with \(r=0.851\), \(0.639\), and \(0.775\). At the same fold-0 comparison, the locked constrained CVAE remained higher in continuous \(r\) (\(0.908\), \(0.851\), \(0.918\)). This is imitation of CVAE-generated targets, not independent biological validation.

## 3.2 Direction concordance and continuous-\(r\) sensitivity

Locked fold-0 seed-42 constrained correlations were \(r=0.908\) (TUH, \(n=200\)), \(0.851\) (OSF, \(n=92\)), and \(0.918\) (P-ADIC, \(n=145\)), each with secondary signs 10/10 (Table 3; Fig. 2). Subject-bootstrap 95% percentile intervals were 0.874-0.900, 0.790-0.837, and 0.871-0.904. These continuous values are checkpoint-specific illustrations under SEME’s Effect-vector axis, not architecture-wide performance claims.

Five-fold means were \(0.385\pm0.308\), \(0.371\pm0.294\), and \(0.372\pm0.310\); signs were 10/10 on three folds, 9/10 on fold 4, and 8/10 on fold 2. Holding fold membership fixed and varying train seeds \(\{42,7,21,123,2024\}\), non-42 continuous \(r\) typically ranged 0.08-0.43 while signs remained mostly 9/10-10/10 (Table 6). Matched unconstrained multi-seed runs stayed low/negative in continuous \(r\); constrained signs exceeded unconstrained signs in all 15 seed\(\times\)cohort cells without establishing continuous-\(r\) stabilization. An equal-weight fold ensemble (\(r=0.642/0.626/0.627\)) did not clearly exceed prior-only mean \(r=0.636\).

Thus under SEME the more reproducible external component in this battery is directional/sign concordance; continuous high \(r\) at locked seed-42 is not a stable architecture-level property.

## 3.3 Magnitude escalation

Magnitude \(\bar{s}\) was tested in sequence (Table 4; Fig. 3). TUH abnormal versus normal was null (p = 0.272, d = -0.13). OSF AD versus HC was an exploratory positive with HC \(n=12\) caveats (p = 0.000278, d = 0.44). P-ADIC AD versus HC was null (p = 0.968, d = -0.07). CAUEEG Dementia versus Normal was null (p = 0.516, d = 0.00, \(n=727\)), despite clear theta/alpha slowing on the same features (p < 0.001, d = 0.47) and good scale/PCA overlap (Fig. 4). Secondary CAUEEG contrasts and disease-label ablations remained null. Magnitude therefore does not provide reliable external diagnosis in this escalation.

## 3.4 Encoding bake-off (classical features versus latent)

On CAUEEG Dementia versus Normal (\(n=727\)), SEME’s Encoding axis compares classical feature references to frozen-twin \(\mu_{\mathrm{base}}\) readouts on the same labels (Fig. 5; Fig. 6; Table 5). A logistic probe on \(\mu_{\mathrm{base}}\) (disease bit fixed to 0) reached mean ROC-AUC 0.579 (permutation p = 0.010). Theta/alpha reached 0.675 (OOF 0.673) and a packed 2185-D logistic reference reached 0.699 (OOF 0.699); 0.699 is a feature-space reference, not a CVAE score or ceiling. Nested heads peaked at OOF AUC 0.593 (tuned L2 logistic; permutation p = 0.005); MLP and boosting did not close the gap. Paired DeLong tests on stored OOF scores confirmed latent below theta/alpha (z = -3.63, p = 0.000285) and below 2185-D (z = -4.64, p < 0.001), and best head below theta/alpha (z = -3.32, p = 0.000909) and below 2185-D (z = -4.09, p < 0.001); theta/alpha versus 2185-D was not significant (z = -1.11, p = 0.268). Matched unconstrained encoding did not exceed constrained latent AUC (0.539 vs 0.579). The bake-off shows encoding-level attenuation under tested readouts, not a formal information-bottleneck proof.

## 3.5 Constraint-weight sweep

An exploratory fold-0 \(\alpha_c\) sweep (Fig. 7) showed signs rising to 10/10 by \(\alpha_c=0.05\), continuous TUH \(r\) peaking near the paper weight \(\alpha_c=0.1\) (\(r=0.897\)), and non-monotone latent AUC (0.539, 0.527, 0.579, 0.514, 0.578 at 0, 0.05, 0.1, 0.2, 0.5). The sweep does not support a simple direction-versus-diagnosis trade-off and is not a nested selection of \(\alpha_c\).

## 3.6 ChemBERTa versus padded one-hot

A matched fold-0 constrained retrain replaced ChemBERTa vectors with 384-D padded one-hot drug IDs without touching the locked checkpoint. Against the locked training signature, one-hot reached 10/10 signs on TUH, OSF, and P-ADIC (\(r=0.176/0.194/0.174\)), while the rematched ChemBERTa arm reached 9/10 (\(r=0.104/0.110/0.076\)). Both rematched continuous-\(r\) values sit in the low multi-seed regime, not the locked high-\(r\) checkpoint. CAUEEG latent mean AUC was 0.540 (ChemBERTa rematch) versus 0.559 (one-hot). In this two-drug setting, chemical embeddings are not required for constraint-consistent signs; the ablation is not a large-library ChemBERTa generalization test, and the locked ChemBERTa twin remains the primary historical checkpoint.
