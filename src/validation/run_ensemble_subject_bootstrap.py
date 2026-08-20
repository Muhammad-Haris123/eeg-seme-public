"""
Subject-level bootstrap CIs for five-fold ensemble external direction.

Reuses the verified subject_bootstrap_effect_metrics algorithm (same seed,
B, percentile CI). Substitutes fold-specific effects with per-subject
fold-averaged ensemble effects from models/validation/fold_sims.

Writes ONLY under models/validation/fold_ensemble/bootstrap/.
Does NOT modify existing fold-level bootstrap or ensemble point-estimate files.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.validation.direction_metrics import load_constrained_training_effects
from src.validation.run_fold_ensemble_external_direction import ensemble_cohort
from src.validation.stats_ci import (
    _cosine,
    _pearson,
    subject_bootstrap_effect_metrics,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


EXPECTED_N = {"tuh": 200, "osf": 92, "padic": 145}
N_BOOT = 2000
SEED = 42
ALPHA = 0.05
# Absolute tolerance vs locked ensemble point estimates (float noise only)
POINT_ATOL = 1e-9
POINT_RTOL = 1e-6


def _extended_bootstrap(
    don: np.ndarray,
    mem: np.ndarray,
    sig_don: np.ndarray,
    sig_mem: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """
    Same resampling and statistics as subject_bootstrap_effect_metrics,
    plus bootstrap median, invalid counts, and full replicate arrays.
    """
    base = subject_bootstrap_effect_metrics(
        don, mem, sig_don, sig_mem, n_boot=n_boot, seed=seed, alpha=alpha
    )

    # Independent second pass with identical RNG seed to capture distributions
    # and QC stats without modifying stats_ci.py.
    n_subj = int(don.shape[0])
    rng = np.random.default_rng(int(seed))

    def _point(idx: np.ndarray) -> Dict[str, float]:
        ed = don[idx].mean(0)
        em = mem[idx].mean(0)
        r_d = _pearson(ed, sig_don)
        r_m = _pearson(em, sig_mem)
        c_d = _cosine(ed, sig_don)
        c_m = _cosine(em, sig_mem)
        return {
            "donepezil_effect_corr": r_d,
            "memantine_effect_corr": r_m,
            "effect_magnitude_correlation": float(np.nanmean([r_d, r_m])),
            "donepezil_cosine": c_d,
            "memantine_cosine": c_m,
            "cosine_mean": float(np.nanmean([c_d, c_m])),
        }

    keys = list(base["point"].keys())
    boots = {k: np.empty(n_boot, dtype=float) for k in keys}
    for b in range(n_boot):
        idx = rng.integers(0, n_subj, size=n_subj)
        row = _point(idx)
        for k in keys:
            boots[k][b] = row[k]

    # Sanity: second-pass point and CI must match verified helper
    point_check = _point(np.arange(n_subj))
    for k in keys:
        if not math.isclose(
            float(point_check[k]), float(base["point"][k]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(f"Point mismatch for {k} between passes")
        lo_q = 100.0 * (alpha / 2.0)
        hi_q = 100.0 * (1.0 - alpha / 2.0)
        lo = float(np.nanpercentile(boots[k], lo_q))
        hi = float(np.nanpercentile(boots[k], hi_q))
        if not math.isclose(lo, float(base["ci"][k]["ci_low"]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"CI low mismatch for {k}")
        if not math.isclose(hi, float(base["ci"][k]["ci_high"]), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"CI high mismatch for {k}")

    invalid: Dict[str, int] = {}
    for k in keys:
        arr = boots[k]
        n_invalid = int(np.sum(~np.isfinite(arr)))
        invalid[k] = n_invalid
        base["ci"][k]["boot_median"] = float(np.nanmedian(arr))
        base["ci"][k]["n_invalid"] = n_invalid
        base["ci"][k]["n_valid"] = int(np.sum(np.isfinite(arr)))

    base["invalid_replicates"] = invalid
    base["n_invalid_total_across_metrics"] = int(sum(invalid.values()))
    return base, boots


def _close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=POINT_RTOL, atol=POINT_ATOL))


def main() -> int:
    root = _root()
    out_dir = root / "models" / "validation" / "fold_ensemble" / "bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble_json = (
        root
        / "models"
        / "validation"
        / "fold_ensemble"
        / "fold_ensemble_external_direction_results.json"
    )
    if not ensemble_json.is_file():
        raise FileNotFoundError(ensemble_json)
    ensemble = json.loads(ensemble_json.read_text(encoding="utf-8"))

    fold_sims = root / "models" / "validation" / "fold_sims"
    sig = load_constrained_training_effects(root)
    fold_ids = [0, 1, 2, 3, 4]
    cohorts = ["tuh", "osf", "padic"]

    try:
        import subprocess

        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or "(clean)"
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_status = "not a git repository"
        git_hash = None

    results = {
        "timestamp": datetime.now().isoformat(),
        "script": "src/validation/run_ensemble_subject_bootstrap.py",
        "git_hash": git_hash,
        "git_status": git_status,
        "n_boot": N_BOOT,
        "seed": SEED,
        "alpha": ALPHA,
        "method": "subject_bootstrap_percentile",
        "resampling_unit": "subject",
        "ensemble_definition": (
            "For each subject, mean of Donepezil/Memantine effect vectors "
            "(drug - baseline) across folds 0..4 from fold_sims; then "
            "subject-bootstrap the cohort-mean effect vs fixed training signature."
        ),
        "statistic": (
            "Mean of Donepezil and Memantine Pearson r (and cosine) between "
            "cohort-mean ensemble effect vectors and fixed constrained "
            "training signature (same as score_effects_vs_signature / "
            "subject_bootstrap_effect_metrics)."
        ),
        "signature_source": sig["source"],
        "fold_sims_root": str(fold_sims),
        "ensemble_point_estimate_file": str(ensemble_json),
        "folds": fold_ids,
        "no_external_tuning": True,
        "cohorts": {},
        "sanity_checks": {},
    }

    all_boots: Dict[str, np.ndarray] = {}
    check_a = check_b = check_c = True

    for name in cohorts:
        print(f"=== ensemble bootstrap {name} ===")
        don_list, mem_list, sids, meta = ensemble_cohort(fold_sims, name, fold_ids)
        don = np.stack(don_list, axis=0)
        mem = np.stack(mem_list, axis=0)
        n = don.shape[0]
        if n != EXPECTED_N[name]:
            check_c = False
            raise RuntimeError(
                f"Subject count mismatch {name}: got {n}, expected {EXPECTED_N[name]}"
            )
        if meta["n_subjects_common"] != n:
            raise RuntimeError(f"Internal n mismatch for {name}")

        boot, boots = _extended_bootstrap(
            don,
            mem,
            sig["donepezil_effect"],
            sig["memantine_effect"],
            n_boot=N_BOOT,
            seed=SEED,
            alpha=ALPHA,
        )

        if boot["n_boot"] != N_BOOT:
            raise RuntimeError("B != 2000")
        if boot["n_invalid_total_across_metrics"] > 0:
            # Non-trivial invalids: stop before interpreting
            n_inv = boot["n_invalid_total_across_metrics"]
            if n_inv > 0:
                # Any invalid is reported; stop if substantial (>1% of B*metrics)
                if n_inv > 0.01 * N_BOOT * len(boot["point"]):
                    raise RuntimeError(f"Substantial invalid replicates: {n_inv}")

        locked = ensemble["cohorts"][name]
        r_pt = float(boot["point"]["effect_magnitude_correlation"])
        c_pt = float(boot["point"]["cosine_mean"])
        r_lock = float(locked["effect_magnitude_correlation"])
        c_lock = float(locked["effect_magnitude_cosine_mean"])

        if not _close(r_pt, r_lock):
            check_a = False
            raise RuntimeError(
                f"CHECK A FAIL {name}: bootstrap point r={r_pt} vs ensemble {r_lock}"
            )
        if not _close(c_pt, c_lock):
            check_b = False
            raise RuntimeError(
                f"CHECK B FAIL {name}: bootstrap point cos={c_pt} vs ensemble {c_lock}"
            )

        r_ci = boot["ci"]["effect_magnitude_correlation"]
        c_ci = boot["ci"]["cosine_mean"]
        block = {
            "n_subjects": n,
            "n_used": n,
            "subject_ids_sha256": hashlib.sha256(
                "\n".join(sids).encode("utf-8")
            ).hexdigest(),
            "n_boot": N_BOOT,
            "seed": SEED,
            "alpha": ALPHA,
            "method": "subject_bootstrap_percentile",
            "point": boot["point"],
            "ci": boot["ci"],
            "invalid_replicates": boot["invalid_replicates"],
            "n_invalid_total_across_metrics": boot["n_invalid_total_across_metrics"],
            "effect_magnitude_correlation_formatted": boot[
                "effect_magnitude_correlation_formatted"
            ],
            "reporting_note": boot["reporting_note"],
            "locked_ensemble_r": r_lock,
            "locked_ensemble_cosine": c_lock,
            "point_matches_locked_ensemble": True,
            "input_fold_sims": [
                str(fold_sims / f"fold_{k}" / name) for k in fold_ids
            ],
        }
        results["cohorts"][name] = block
        for k, arr in boots.items():
            all_boots[f"{name}__{k}"] = arr.astype(np.float64)

        print(
            f"  n={n} r={r_pt:.6f} "
            f"CI[{r_ci['ci_low']:.6f},{r_ci['ci_high']:.6f}] "
            f"cos={c_pt:.6f} "
            f"CI[{c_ci['ci_low']:.6f},{c_ci['ci_high']:.6f}] "
            f"invalid={boot['n_invalid_total_across_metrics']}"
        )

    results["sanity_checks"] = {
        "CHECK_A_point_r_matches_ensemble": check_a,
        "CHECK_B_point_cosine_matches_ensemble": check_b,
        "CHECK_C_subject_counts": check_c and all(
            results["cohorts"][c]["n_subjects"] == EXPECTED_N[c] for c in cohorts
        ),
        "CHECK_D_B_equals_2000": all(
            results["cohorts"][c]["n_boot"] == 2000 for c in cohorts
        ),
        "CHECK_E_resampling_unit": "subject",
        "CHECK_F_drug_pairing": "donepezil and memantine effects indexed by same subject idx",
        "CHECK_G_fixed_signature": sig["source"],
        "CHECK_H_no_external_tuning": True,
    }

    json_path = out_dir / "ensemble_subject_bootstrap_B2000.json"
    if json_path.exists():
        raise FileExistsError(f"Refusing to overwrite {json_path}")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    npz_path = out_dir / "ensemble_bootstrap_distributions.npz"
    if npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite {npz_path}")
    np.savez_compressed(npz_path, **all_boots)

    lines = [
        "# Five-fold ensemble subject-level bootstrap (B=2000)",
        "",
        f"Script: `{results['script']}`",
        f"Seed: {SEED}; B: {N_BOOT}; method: percentile 95% CI",
        f"Signature: `{sig['source']}`",
        "",
        "| Cohort | n | r | 95% CI | Cosine | 95% CI |",
        "| ------ | --: | --: | -----: | -----: | -----: |",
    ]
    for name in cohorts:
        c = results["cohorts"][name]
        r = c["point"]["effect_magnitude_correlation"]
        rc = c["ci"]["effect_magnitude_correlation"]
        cos = c["point"]["cosine_mean"]
        cc = c["ci"]["cosine_mean"]
        lines.append(
            f"| {name.upper()} | {c['n_subjects']} | {r:.6f} | "
            f"{rc['ci_low']:.6f} to {rc['ci_high']:.6f} | {cos:.6f} | "
            f"{cc['ci_low']:.6f} to {cc['ci_high']:.6f} |"
        )
    lines.append("")
    lines.append("Bootstrap median / SD (r):")
    for name in cohorts:
        rc = results["cohorts"][name]["ci"]["effect_magnitude_correlation"]
        lines.append(
            f"- **{name}**: median={rc['boot_median']:.6f}; "
            f"SD={rc['boot_std']:.6f}; invalid={rc['n_invalid']}"
        )
    md_path = out_dir / "ensemble_subject_bootstrap_summary.md"
    if md_path.exists():
        raise FileExistsError(f"Refusing to overwrite {md_path}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[wrote] {json_path}")
    print(f"[wrote] {npz_path}")
    print(f"[wrote] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
