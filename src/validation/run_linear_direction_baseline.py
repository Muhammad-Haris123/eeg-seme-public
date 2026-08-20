"""
Phase 1 control: fitted regularized linear drug-conditioned direction baseline.

Does NOT retrain the CVAE. Does NOT overwrite locked JSON.

Scientific question (control, not discovery):
  Can a ridge map Δx = W_x x_base + W_d d + W_y y + b reproduce the imposed
  locked-CVAE training-signature direction on TUH / OSF / P-ADIC?

Targets are the ONLY stored per-subject Δx: locked fold-0 CVAE simulations in
models/simulations_constrained_full (drug-minus-baseline, 2185-D).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.utils.feature_processor import flatten_features
from src.models.data_loader import get_subject_level_kfold, load_drug_embeddings
from src.validation.direction_metrics import (
    direction_agree,
    load_constrained_training_effects,
)
from src.validation.run_unconstrained_external_battery import load_osf, load_padic, load_tuh
from src.validation.stats_ci import _cosine, subject_bootstrap_effect_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models" / "validation" / "linear_direction_baseline"
SCRIPT_PATH = Path(__file__).resolve()

FEAT_DIM = 2185
DRUG_DIM = 384
N_SPLITS = 5
RANDOM_SEED = 42
N_BOOT = 2000
ALPHA_GRID = np.logspace(-2, 6, 17)
DRUG_NAMES = ("donepezil", "memantine")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    return str(obj)


def load_training_packed() -> Tuple[np.ndarray, np.ndarray, List[int]]:
    feat_dir = ROOT / "data" / "eeg_features"
    flats, ys, gids = [], [], []
    for group, y in (("AD", 1), ("HC", 0)):
        psd = np.load(feat_dir / f"{group}_psd.npy")
        bp = np.load(feat_dir / f"{group}_band_powers.npy")
        coh = np.load(feat_dir / f"{group}_coherence.npy")
        plv = np.load(feat_dir / f"{group}_plv.npy")
        for i in range(psd.shape[0]):
            flats.append(flatten_features(psd[i], bp[i], coh[i], plv[i]).astype(np.float64))
            ys.append(y)
            gids.append(len(gids))
    X = np.stack(flats, axis=0)
    y = np.asarray(ys, dtype=np.float64)
    if X.shape != (66, FEAT_DIM):
        raise RuntimeError(f"Unexpected training packed shape {X.shape}; expected (66, {FEAT_DIM})")
    return X, y, gids


def load_cvae_subject_effects() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    sim_dir = ROOT / "models" / "simulations_constrained_full"
    meta = json.loads((sim_dir / "simulation_metadata.json").read_text(encoding="utf-8"))
    info = meta["subject_info"]
    for i, row in enumerate(info):
        if int(row["subject_idx"]) != i:
            raise RuntimeError(
                "STOP: simulations_constrained_full subject_info is not gid-aligned. "
                f"index {i} has subject_idx={row['subject_idx']}"
            )
    base = np.load(sim_dir / "simulated_baseline.npy").mean(1)
    don = np.load(sim_dir / "simulated_donepezil.npy").mean(1)
    mem = np.load(sim_dir / "simulated_memantine.npy").mean(1)
    if base.shape != (66, FEAT_DIM):
        raise RuntimeError(f"Unexpected simulation shape {base.shape}")
    return (don - base).astype(np.float64), (mem - base).astype(np.float64), meta


def pack_design(x: np.ndarray, d: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return np.concatenate([x, d, y], axis=1)


def build_rows(
    subject_ids: np.ndarray,
    X_base: np.ndarray,
    y: np.ndarray,
    delta_don: np.ndarray,
    delta_mem: np.ndarray,
    drug_emb: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows_x, rows_d, rows_y, rows_t = [], [], [], []
    sids = []
    drugs = []
    for sid in subject_ids:
        sid = int(sid)
        for name, delta in (("donepezil", delta_don), ("memantine", delta_mem)):
            rows_x.append(X_base[sid])
            rows_d.append(drug_emb[name].astype(np.float64))
            rows_y.append(y[sid])
            rows_t.append(delta[sid])
            sids.append(sid)
            drugs.append(name)
    return (
        pack_design(np.stack(rows_x), np.stack(rows_d), np.asarray(rows_y)),
        np.stack(rows_t),
        np.asarray(sids, dtype=np.int64),
        np.asarray(drugs),
    )


def select_alpha(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xva = scaler.transform(X_val)
    best_a, best_mse = None, np.inf
    curve = {}
    for a in ALPHA_GRID:
        model = Ridge(alpha=float(a), fit_intercept=True)
        model.fit(Xtr, Y_train)
        pred = model.predict(Xva)
        mse = float(np.mean((pred - Y_val) ** 2))
        curve[f"{a:.6g}"] = mse
        if mse < best_mse:
            best_mse = mse
            best_a = float(a)
    if best_a is None:
        raise RuntimeError("Ridge alpha selection failed")
    return best_a, curve


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> Tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(Xs, Y)
    return scaler, model


def predict_delta(
    scaler: StandardScaler,
    model: Ridge,
    x_base: np.ndarray,
    d: np.ndarray,
    y: float,
) -> np.ndarray:
    row = pack_design(x_base.reshape(1, -1), d.reshape(1, -1), np.array([y], dtype=np.float64))
    return model.predict(scaler.transform(row))[0]


def predict_cohort_effects(
    scaler: StandardScaler,
    model: Ridge,
    flats: np.ndarray,
    diseases: List[int],
    drug_emb: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    don, mem = [], []
    for flat, y in zip(flats, diseases):
        don.append(predict_delta(scaler, model, flat.astype(np.float64), drug_emb["donepezil"], float(y)))
        mem.append(predict_delta(scaler, model, flat.astype(np.float64), drug_emb["memantine"], float(y)))
    return np.stack(don), np.stack(mem)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_cohort_means(
    don_effects: np.ndarray,
    mem_effects: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subject-resampled cohort-mean effect vectors. Same RNG as stats_ci (default_rng)."""
    n_subj = int(don_effects.shape[0])
    rng = np.random.default_rng(int(seed))
    don_boot = np.empty((n_boot, don_effects.shape[1]), dtype=np.float64)
    mem_boot = np.empty_like(don_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_subj, size=n_subj)
        don_boot[b] = don_effects[idx].mean(0)
        mem_boot[b] = mem_effects[idx].mean(0)
    return don_boot, mem_boot


def score_vs_signature(
    don_effects: np.ndarray,
    mem_effects: np.ndarray,
    sig_don: np.ndarray,
    sig_mem: np.ndarray,
    signature_name: str,
    don_boot: np.ndarray | None = None,
    mem_boot: np.ndarray | None = None,
) -> Dict[str, Any]:
    ext_don = don_effects.mean(0)
    ext_mem = mem_effects.mean(0)
    d_a, d_t = direction_agree(ext_don, sig_don)
    m_a, m_t = direction_agree(ext_mem, sig_mem)
    r_d = _pearson(ext_don, sig_don)
    r_m = _pearson(ext_mem, sig_mem)
    c_d = _cosine(ext_don, sig_don)
    c_m = _cosine(ext_mem, sig_mem)
    if don_boot is None or mem_boot is None:
        boot = subject_bootstrap_effect_metrics(
            don_effects, mem_effects, sig_don, sig_mem, n_boot=N_BOOT, seed=RANDOM_SEED
        )
        ci_r = boot["ci"]["effect_magnitude_correlation"]
        ci_d = boot["ci"]["donepezil_effect_corr"]
        ci_m = boot["ci"]["memantine_effect_corr"]
        boot_block = {
            "B": N_BOOT,
            "seed": RANDOM_SEED,
            "method": boot.get("method"),
            "effect_magnitude_correlation": {
                "point": boot["point"]["effect_magnitude_correlation"],
                "ci_low": ci_r["ci_low"],
                "ci_high": ci_r["ci_high"],
            },
            "donepezil_effect_corr": {
                "point": boot["point"]["donepezil_effect_corr"],
                "ci_low": ci_d["ci_low"],
                "ci_high": ci_d["ci_high"],
            },
            "memantine_effect_corr": {
                "point": boot["point"]["memantine_effect_corr"],
                "ci_low": ci_m["ci_low"],
                "ci_high": ci_m["ci_high"],
            },
        }
    else:
        r_d_b = np.array([_pearson(don_boot[b], sig_don) for b in range(len(don_boot))])
        r_m_b = np.array([_pearson(mem_boot[b], sig_mem) for b in range(len(mem_boot))])
        r_mean_b = 0.5 * (r_d_b + r_m_b)
        lo_q, hi_q = 2.5, 97.5
        boot_block = {
            "B": N_BOOT,
            "seed": RANDOM_SEED,
            "method": "subject_bootstrap_percentile_shared_means",
            "effect_magnitude_correlation": {
                "point": float(np.nanmean([r_d, r_m])),
                "ci_low": float(np.nanpercentile(r_mean_b, lo_q)),
                "ci_high": float(np.nanpercentile(r_mean_b, hi_q)),
            },
            "donepezil_effect_corr": {
                "point": r_d,
                "ci_low": float(np.nanpercentile(r_d_b, lo_q)),
                "ci_high": float(np.nanpercentile(r_d_b, hi_q)),
            },
            "memantine_effect_corr": {
                "point": r_m,
                "ci_low": float(np.nanpercentile(r_m_b, lo_q)),
                "ci_high": float(np.nanpercentile(r_m_b, hi_q)),
            },
        }
    return {
        "signature_used": signature_name,
        "n_used": int(don_effects.shape[0]),
        "direction_agreement_total": f"{d_a + m_a}/{d_t + m_t}",
        "donepezil_direction_agreement": f"{d_a}/{d_t}",
        "memantine_direction_agreement": f"{m_a}/{m_t}",
        "donepezil_effect_corr": r_d,
        "memantine_effect_corr": r_m,
        "effect_magnitude_correlation": float(np.nanmean([r_d, r_m])),
        "donepezil_cosine": c_d,
        "memantine_cosine": c_m,
        "cosine_mean": float(np.nanmean([c_d, c_m])),
        "bootstrap": boot_block,
    }


def own_signature_from_training(
    scaler: StandardScaler,
    model: Ridge,
    X_base: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    drug_emb: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    don, mem = [], []
    for sid in subject_ids:
        sid = int(sid)
        don.append(
            predict_delta(scaler, model, X_base[sid], drug_emb["donepezil"], float(y[sid]))
        )
        mem.append(
            predict_delta(scaler, model, X_base[sid], drug_emb["memantine"], float(y[sid]))
        )
    don = np.stack(don)
    mem = np.stack(mem)
    return {
        "donepezil_effect": don.mean(0),
        "memantine_effect": mem.mean(0),
        "n_subjects": int(len(subject_ids)),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[linear] loading training packed features and locked CVAE delta-x targets")
    X_base, y, _gids = load_training_packed()
    delta_don, delta_mem, sim_meta = load_cvae_subject_effects()
    drug_emb = load_drug_embeddings()
    for name in DRUG_NAMES:
        if name not in drug_emb:
            raise RuntimeError(f"Missing drug embedding {name}")
        if int(np.asarray(drug_emb[name]).size) != DRUG_DIM:
            raise RuntimeError(f"{name} embedding dim {np.asarray(drug_emb[name]).size} != {DRUG_DIM}")

    osf_a = ROOT / "data" / "ad_labeled_external" / "features"
    osf_b = ROOT / "data" / "external_osf" / "features"
    osf_equiv = True
    osf_note = "data/ad_labeled_external used as canonical; SHA256 match vs data/external_osf/features for AD/HC psd and band_powers."
    for name in ("AD_psd.npy", "HC_psd.npy", "AD_band_powers.npy", "HC_band_powers.npy"):
        ba = (osf_a / name).read_bytes()
        bb = (osf_b / name).read_bytes()
        if ba != bb:
            osf_equiv = False
            osf_note = f"STOP: {name} differs between OSF locations"
    if not osf_equiv:
        raise RuntimeError(osf_note)

    locked_sig = load_constrained_training_effects(ROOT)
    subject_ids, labels, test_folds = get_subject_level_kfold(n_splits=N_SPLITS, random_seed=RANDOM_SEED)
    if not np.array_equal(labels, y.astype(int)):
        raise RuntimeError("STOP: kfold labels do not match packed AD/HC order")

    print("[linear] loading external cohorts (no inspection of direction scores before α freeze)")
    tuh_flats, tuh_y, tuh_sids, _ = load_tuh(ROOT)
    osf_flats, osf_y, osf_sids, _ = load_osf(ROOT)
    padic_flats, padic_y, padic_sids, _ = load_padic(ROOT)
    cohorts = {
        "TUH": {"flats": tuh_flats, "y": tuh_y, "sids": tuh_sids},
        "OSF": {"flats": osf_flats, "y": osf_y, "sids": osf_sids},
        "P-ADIC": {"flats": padic_flats, "y": padic_y, "sids": padic_sids},
    }
    print(
        f"[linear] n TUH={len(tuh_y)} OSF={len(osf_y)} P-ADIC={len(padic_y)}; "
        "α will be selected on training-cohort inner val only"
    )

    fold_records = []
    for fold_idx in range(N_SPLITS):
        test_ids = np.asarray(test_folds[fold_idx], dtype=np.int64)
        train_val_ids = np.concatenate(
            [test_folds[j] for j in range(N_SPLITS) if j != fold_idx]
        ).astype(np.int64)
        train_ids, val_ids = train_test_split(
            train_val_ids.tolist(),
            test_size=0.1,
            stratify=labels[train_val_ids],
            random_state=RANDOM_SEED + fold_idx,
        )
        train_ids = np.asarray(train_ids, dtype=np.int64)
        val_ids = np.asarray(val_ids, dtype=np.int64)

        Xtr, Ytr, _, _ = build_rows(train_ids, X_base, y, delta_don, delta_mem, drug_emb)
        Xva, Yva, _, _ = build_rows(val_ids, X_base, y, delta_don, delta_mem, drug_emb)
        alpha, curve = select_alpha(Xtr, Ytr, Xva, Yva)
        scaler, model = fit_ridge(Xtr, Ytr, alpha)

        own_fit = own_signature_from_training(scaler, model, X_base, y, train_ids, drug_emb)
        own_full = own_signature_from_training(scaler, model, X_base, y, subject_ids, drug_emb)

        np.savez_compressed(
            OUT_DIR / f"fold_{fold_idx}_own_signatures.npz",
            own_fit_donepezil=own_fit["donepezil_effect"].astype(np.float32),
            own_fit_memantine=own_fit["memantine_effect"].astype(np.float32),
            own_full66_donepezil=own_full["donepezil_effect"].astype(np.float32),
            own_full66_memantine=own_full["memantine_effect"].astype(np.float32),
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            alpha=np.float64(alpha),
        )

        rec = {
            "fold": fold_idx,
            "alpha": alpha,
            "alpha_selection": {
                "method": "inner 10% subject split of non-test IDs; minimize MSE of 2185-D Δx",
                "test_size": 0.1,
                "stratify": "disease label",
                "random_state": RANDOM_SEED + fold_idx,
                "grid": ALPHA_GRID.tolist(),
                "val_mse_by_alpha": curve,
                "selected_val_mse": curve[f"{alpha:.6g}"],
            },
            "n_train_subjects": int(len(train_ids)),
            "n_val_subjects": int(len(val_ids)),
            "n_test_subjects_training_cohort": int(len(test_ids)),
            "train_ids": train_ids.tolist(),
            "val_ids": val_ids.tolist(),
            "test_ids": test_ids.tolist(),
            "refit": "Ridge fit on train_ids only (val used only for alpha); CVAE-matched",
            "primary": {},
            "secondary_own_fit_subjects": {},
            "secondary_own_full66": {},
        }

        for cname, c in cohorts.items():
            don_e, mem_e = predict_cohort_effects(scaler, model, c["flats"], c["y"], drug_emb)
            np.savez_compressed(
                OUT_DIR / f"fold_{fold_idx}_{cname.replace('-', '').lower()}_predicted_effects.npz",
                subject_ids=np.asarray(c["sids"]),
                disease_label=np.asarray(c["y"], dtype=np.int64),
                donepezil_effect=don_e.astype(np.float32),
                memantine_effect=mem_e.astype(np.float32),
            )
            don_boot, mem_boot = bootstrap_cohort_means(don_e, mem_e)
            rec["primary"][cname] = score_vs_signature(
                don_e,
                mem_e,
                locked_sig["donepezil_effect"],
                locked_sig["memantine_effect"],
                "locked_CVAE_training_signature_simulations_constrained_full",
                don_boot=don_boot,
                mem_boot=mem_boot,
            )
            rec["secondary_own_fit_subjects"][cname] = score_vs_signature(
                don_e,
                mem_e,
                own_fit["donepezil_effect"],
                own_fit["memantine_effect"],
                "linear_own_signature_mean_predicted_delta_on_fit_subjects",
                don_boot=don_boot,
                mem_boot=mem_boot,
            )
            rec["secondary_own_full66"][cname] = score_vs_signature(
                don_e,
                mem_e,
                own_full["donepezil_effect"],
                own_full["memantine_effect"],
                "linear_own_signature_mean_predicted_delta_on_all_66_training_subjects",
                don_boot=don_boot,
                mem_boot=mem_boot,
            )
            print(
                f"[linear] fold {fold_idx} {cname} primary r="
                f"{rec['primary'][cname]['effect_magnitude_correlation']:.3f} "
                f"signs={rec['primary'][cname]['direction_agreement_total']}"
            )
        fold_records.append(rec)

    def _five_fold_summary(block: str) -> Dict[str, Any]:
        out = {}
        for cname in ("TUH", "OSF", "P-ADIC"):
            rs = [f[block][cname]["effect_magnitude_correlation"] for f in fold_records]
            signs = [f[block][cname]["direction_agreement_total"] for f in fold_records]
            out[cname] = {
                "r_mean": float(np.mean(rs)),
                "r_std": float(np.std(rs, ddof=1)),
                "r_per_fold": rs,
                "signs_per_fold": signs,
            }
        return out

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "fitted_ridge_linear_drug_conditioned_direction_baseline",
        "status": "NEW_NON_LOCKED_PHASE1_CONTROL",
        "scientific_role": (
            "Control analysis. Not independent pharmacodynamic discovery, "
            "not clinical validation, not diagnostic superiority."
        ),
        "model": {
            "form": "Δx = W_x x_base + W_d d + W_y y + b",
            "estimator": "sklearn.linear_model.Ridge",
            "regularization": "L2 (ridge)",
            "fit_intercept": True,
            "feature_scaling": "StandardScaler fit on training-fold design matrix only",
            "input_dim": FEAT_DIM + DRUG_DIM + 1,
            "output_dim": FEAT_DIM,
            "x_base": "packed 2185-D EEG features (psd[0:380), bands[380:475), interleaved coh/PLV[475:2185))",
            "d": "frozen ChemBERTa 384-D embeddings from load_drug_embeddings()",
            "y": "disease label AD=1 / HC=0 (external: TUH abnormal=1, OSF/P-ADIC AD=1)",
            "target_delta": (
                "locked CVAE simulated drug-minus-baseline from "
                "models/simulations_constrained_full (mean over 10 samples). "
                "These are CVAE outputs, not observed post-dose EEG."
            ),
            "alpha_grid": ALPHA_GRID.tolist(),
            "alpha_selection": (
                "Per outer fold: inner 10% stratified subject split of non-test IDs, "
                "random_state=42+fold_idx, matching CVAE train_phase2_upgraded.py. "
                "Select alpha minimizing MSE on inner val. No external cohort used."
            ),
            "fold_protocol": "get_subject_level_kfold(n_splits=5, random_seed=42); StratifiedKFold",
            "seed": RANDOM_SEED,
            "n_training_subjects": 66,
            "n_ad": 37,
            "n_hc": 29,
        },
        "osf_location_check": {
            "canonical": "data/ad_labeled_external (features_calibrated for x_base, matching Layer 5b / unconstrained battery)",
            "compared_to": "data/external_osf/features",
            "equivalent_raw_group_arrays": True,
            "note": osf_note,
        },
        "locked_cvae_fold0_seed42_for_reference_only_not_replaced": {
            "TUH": {"r": 0.908, "signs": "10/10"},
            "OSF": {"r": 0.851, "signs": "10/10"},
            "P-ADIC": {"r": 0.918, "signs": "10/10"},
            "scored_against": "own CVAE training signature",
        },
        "prior_only_existing_locked": {
            "r_mean": 0.6360168032016147,
            "donepezil_r": 0.5715458752815976,
            "memantine_r": 0.7004877311216317,
            "signs": "10/10",
            "scored_against": "CVAE training signature",
            "note": "Prior-only is a fixed vector vs the training signature; not a per-cohort TUH/OSF/P-ADIC r.",
            "source": "models/validation/prior_only_direction.json",
        },
        "simulation_metadata_checkpoint": sim_meta.get("source_checkpoint"),
        "script": str(SCRIPT_PATH),
        "folds": fold_records,
        "five_fold_summary_primary_locked_cvae_signature": _five_fold_summary("primary"),
        "five_fold_summary_secondary_own_fit_subjects": _five_fold_summary("secondary_own_fit_subjects"),
        "five_fold_summary_secondary_own_full66": _five_fold_summary("secondary_own_full66"),
        "interpretation_constraints": [
            "Do not treat own-signature scores as external generalization by themselves.",
            "Do not replace locked CVAE fold-0 r=0.908/0.851/0.918.",
            "Do not claim independent pharmacodynamic discovery.",
            "Do not call the best fold a new hero result.",
        ],
    }

    out_json = OUT_DIR / "linear_direction_baseline_results.json"
    out_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")

    f0 = fold_records[0]
    md = []
    md.append("# Linear direction baseline (Phase 1 control; NEW, non-locked)\n")
    md.append(f"- Timestamp (UTC): {payload['timestamp_utc']}\n")
    md.append(f"- Script: `{SCRIPT_PATH}`\n")
    md.append(
        "- Targets: locked CVAE simulated Δx from `models/simulations_constrained_full` "
        "(not observed post-dose EEG).\n"
    )
    md.append("\n## Comparison table (computed values only)\n")
    md.append(
        "| Model | TUH r | TUH signs | OSF r | OSF signs | P-ADIC r | P-ADIC signs | Scored against |\n"
    )
    md.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    md.append(
        "| Prior-only (existing locked) | — | 10/10 | — | 10/10 | — | 10/10 | CVAE training signature |\n"
    )

    def _row(label: str, block: Dict[str, Any], against: str) -> str:
        def r(c):
            return f"{block[c]['effect_magnitude_correlation']:.3f}"

        def s(c):
            return block[c]["direction_agreement_total"]

        return (
            f"| {label} | {r('TUH')} | {s('TUH')} | {r('OSF')} | {s('OSF')} | "
            f"{r('P-ADIC')} | {s('P-ADIC')} | {against} |\n"
        )

    md.append(
        _row(
            "Linear, fold-0, primary",
            f0["primary"],
            "exact locked CVAE signature",
        )
    )
    md.append(
        _row(
            "Linear, fold-0, secondary (fit subjects)",
            f0["secondary_own_fit_subjects"],
            "own fitted signature (train_ids)",
        )
    )
    md.append(
        _row(
            "Linear, fold-0, secondary (full 66)",
            f0["secondary_own_full66"],
            "own fitted signature (all 66)",
        )
    )
    md.append(
        "| CVAE locked fold-0 seed-42 | 0.908 | 10/10 | 0.851 | 10/10 | 0.918 | 10/10 | own training signature |\n"
    )
    md.append("\nProtocol-matched secondary comparator in the required table is **own fitted signature**.\n")
    md.append("The full-66 own-signature row is reported for CVAE-signature-construction analogy; it does not replace the locked-CVAE-signature comparison.\n")
    md.append("\n## Five-fold primary summary (locked CVAE signature)\n")
    for cname, row in payload["five_fold_summary_primary_locked_cvae_signature"].items():
        md.append(
            f"- {cname}: mean r = {row['r_mean']:.3f} (sd {row['r_std']:.3f}); "
            f"per-fold r = {[round(v, 3) for v in row['r_per_fold']]}; "
            f"signs = {row['signs_per_fold']}\n"
        )
    md.append(
        "\nNo superiority claim is written here. Compare the numbers. "
        "This control does not establish independent pharmacodynamic discovery.\n"
    )
    (OUT_DIR / "linear_direction_baseline_summary.md").write_text("".join(md), encoding="utf-8")
    print(f"[wrote] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
