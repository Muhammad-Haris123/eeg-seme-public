# Cover letter — Computer Methods and Programs in Biomedicine

**Status:** Path A draft (authors/institutions to be completed at submission)  
**Manuscript working title:** SEME: A Decoupled Evaluation Protocol for Literature-Constrained EEG–Drug Generators  
**Article type:** Original research

---

Dear Editor,

Please consider our manuscript for publication in *Computer Methods and Programs in Biomedicine*.

**Fit.** CMPB publishes computational methods and evaluation frameworks for biomedical data. This manuscript contributes a named evaluation protocol (**SEME**: Signed concordance, Effect-vector fidelity, Magnitude transfer, Encoding retention) for literature-constrained, chemistry-conditioned EEG generative models, with a pharmacodynamically penalized Donepezil/Memantine CVAE as a worked case study—not as a clinically validated pharmacodynamic digital twin.

**What is known.** Generative EEG and digital-twin papers often conflate reconstruction quality, signed response direction, continuous effect-vector fidelity, diagnostic magnitude, and residual label encoding. Chemistry-conditioned models are sometimes scored as if signature concordance implied post-dose pharmacological validation, even when external cohorts lack paired pre/post-drug EEG.

**What this paper adds.** SEME separately scores the four axes above on the same generator, with matched unconstrained twins, a prior-only signed effect (circularity budget), a linear imitation control on CVAE-simulated targets, fold and multi-seed sensitivity, CAUEEG encoding probes against theta/alpha and packed 2185-D logistic references, paired DeLong tests on stored out-of-fold scores, and a ChemBERTa-versus-padded-one-hot ablation under a matched fold-0 protocol. Locked continuous correlations (\(r=0.908/0.851/0.918\)) are reported as checkpoint-specific secondary illustrations; the more stable external finding in this battery is constrained sign concordance relative to unconstrained twins, while prior-only already reaches 10/10 signs at mean \(r=0.636\).

**What we do not claim.** Empirical post-dose pharmacodynamic validation; seed- or fold-stable continuous external fidelity; independent pharmacological discovery; Alzheimer’s diagnostic superiority; ChemBERTa large-library chemical generalization; or constraint-caused diagnostic compression.

**Reproducibility.** Supplementary Material S1 and `paper/CODE_AVAILABILITY.md` list environment pins, frozen checkpoint aliases, and a one-command artifact regeneration path. Public code: https://github.com/Muhammad-Haris123/eeg-seme-public (commit `31e4249`). Locked checkpoints: Zenodo DOI https://doi.org/10.5281/zenodo.22028681. Raw TUH/CAUEEG EEG are not redistributed.

This work has not been published and is not under consideration elsewhere. Author names and affiliations will be completed in the Editorial Manager submission metadata.

Thank you for your consideration.

Sincerely,  
[Corresponding author — to be completed at submission]

---

## Hostile one-paragraph self-test (do not paste into letter)

A referee who only reads the title and abstract should conclude: “This is an evaluation-protocol paper with a constrained CVAE case study,” not “This validates a clinical EEG drug-response twin.”
