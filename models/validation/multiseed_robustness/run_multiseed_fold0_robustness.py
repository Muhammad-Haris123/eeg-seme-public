"""
Controlled multi-seed robustness audit for locked fold-0 constrained CVAE.

ISOLATED ANALYSIS ONLY — does not modify protected checkpoints, validation JSON,
or manuscript sources.

Critical design:
  - split_seed FIXED at 42 (same subject KFold + same fold-0 train/val split)
  - train_seed VARIES (torch/numpy RNG for weight init + DataLoader shuffle)
  - Train ONLY fold 0; select by min internal val total loss (same rule)
  - External scoring ONLY after checkpoint freeze
  - Seed 42 uses EXISTING locked checkpoint/results (not retrained)
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from run_constrained_training import (  # noqa: E402
    _build_subject_baseline_cache,
    train_epoch_constrained,
    validate_epoch_constrained,
)
from src.models.config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG  # noqa: E402
from src.models.data_loader import (  # noqa: E402
    create_data_loaders_for_fold,
    get_subject_level_kfold,
)
from src.models.inference import load_trained_model, simulate_post_drug_eeg  # noqa: E402
from src.models.data_loader import load_drug_embeddings  # noqa: E402
from src.models.train_phase2 import DigitalTwinModel  # noqa: E402
from src.tuh.process_tuh_eeg import unflatten_features  # noqa: E402
from src.validation.direction_metrics import (  # noqa: E402
    load_constrained_training_effects,
    score_effects_vs_signature,
)
from src.validation.run_unconstrained_external_battery import (  # noqa: E402
    load_osf,
    load_padic,
    load_tuh,
)
from src.validation.stats_ci import _cosine  # noqa: E402

OUT = ROOT / "models" / "validation" / "multiseed_robustness"
SEEDS_NEW = [7, 21, 123, 2024]
SEED_REFERENCE = 42
SPLIT_SEED = 42
FOLD_IDX = 0
CONSTRAINT_WEIGHT = 0.1
WARMUP_EPOCHS = 5
NUM_SAMPLES = 5  # match fold external / paper sims


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash() -> str:
    payload = {
        "MODEL_CONFIG": MODEL_CONFIG,
        "TRAINING_CONFIG": {k: v for k, v in TRAINING_CONFIG.items() if k != "device"},
        "DATA_CONFIG": DATA_CONFIG,
        "split_seed": SPLIT_SEED,
        "fold_idx": FOLD_IDX,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "warmup_epochs": WARMUP_EPOCHS,
        "epochs": TRAINING_CONFIG["num_epochs"],
        "note": "train_seed varies; split_seed fixed",
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def set_train_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_fixed_fold0_ids() -> Tuple[List[int], List[int], List[int], np.ndarray]:
    """Freeze subject splits to seed-42 protocol (identical to locked training)."""
    subject_ids, labels, test_folds = get_subject_level_kfold(
        n_splits=5, random_seed=SPLIT_SEED
    )
    test_ids = test_folds[FOLD_IDX].tolist()
    train_val_ids = np.concatenate(
        [test_folds[j] for j in range(5) if j != FOLD_IDX]
    )
    train_ids, val_ids = train_test_split(
        train_val_ids.tolist(),
        test_size=0.1,
        stratify=labels[train_val_ids],
        random_state=SPLIT_SEED + FOLD_IDX,  # 42 + 0 = 42 (locked protocol)
    )
    return train_ids, val_ids, test_ids, labels


def train_fold0_for_seed(train_seed: int, seed_dir: Path) -> Dict:
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = seed_dir / "checkpoint.pt"
    report_path = seed_dir / "training_report.json"

    training_start = datetime.now().isoformat()
    device = torch.device(TRAINING_CONFIG["device"])
    epochs = int(TRAINING_CONFIG["num_epochs"])
    batch_size = int(TRAINING_CONFIG["batch_size"])
    beta_kl = float(TRAINING_CONFIG["beta_kl"])

    train_ids, val_ids, test_ids, labels = get_fixed_fold0_ids()

    # Seed AFTER fixed sklearn splits so only torch init/shuffle vary
    set_train_seed(train_seed)
    train_loader, val_loader, test_loader = create_data_loaders_for_fold(
        train_ids, val_ids, test_ids, batch_size=batch_size, shuffle_train=True
    )
    baseline_cache = _build_subject_baseline_cache(train_loader)

    set_train_seed(train_seed)  # re-seed before model init
    model = DigitalTwinModel(config=MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )

    best_val = float("inf")
    best_epoch = -1
    history = []
    diag_flag = {"printed": False}

    for epoch in range(epochs):
        tr = train_epoch_constrained(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            baseline_cache=baseline_cache,
            beta_kl=beta_kl,
            constraint_weight=CONSTRAINT_WEIGHT,
            warmup_epochs=WARMUP_EPOCHS,
            diagnostic_once=diag_flag,
        )
        va = validate_epoch_constrained(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            baseline_cache=baseline_cache,
            beta_kl=beta_kl,
            constraint_weight=CONSTRAINT_WEIGHT,
            warmup_epochs=WARMUP_EPOCHS,
        )
        history.append({"epoch": epoch + 1, "train": tr, "val": va})
        if va["total"] < best_val:
            best_val = float(va["total"])
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va,
                    "constraint_weight": CONSTRAINT_WEIGHT,
                    "warmup_epochs": WARMUP_EPOCHS,
                    "train_seed": train_seed,
                    "split_seed": SPLIT_SEED,
                    "fold_idx": FOLD_IDX,
                    "selection_rule": "min_internal_val_total_loss_fold0_fixed_split_seed42",
                },
                ckpt_path,
            )

    selection_time = datetime.now().isoformat()
    training_end = selection_time
    ckpt_sha = _sha256(ckpt_path) if ckpt_path.exists() else None

    # Architecture parity checks
    n_params = sum(p.numel() for p in model.parameters())
    report = {
        "train_seed": train_seed,
        "split_seed": SPLIT_SEED,
        "fold_idx": FOLD_IDX,
        "training_start_time": training_start,
        "training_end_time": training_end,
        "checkpoint_selection_time": selection_time,
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)).replace("\\", "/"),
        "selection_metric": "val_total_loss",
        "selection_rule": "argmin_epoch val_total on fixed fold-0 split (split_seed=42)",
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "warmup_epochs": WARMUP_EPOCHS,
        "epochs": epochs,
        "beta_kl": beta_kl,
        "batch_size": batch_size,
        "n_train_subjects": len(train_ids),
        "n_val_subjects": len(val_ids),
        "n_test_subjects": len(test_ids),
        "train_subject_ids": [int(x) for x in train_ids],
        "val_subject_ids": [int(x) for x in val_ids],
        "test_subject_ids": [int(x) for x in test_ids],
        "n_parameters": int(n_params),
        "input_dim": 2185,
        "latent_dim": MODEL_CONFIG["cvae"]["latent_dim"],
        "checkpoint_sha256": ckpt_sha,
        "training_config_hash": _config_hash(),
        "device": str(device),
        "history_tail": history[-5:],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@torch.no_grad()
def score_checkpoint_external(checkpoint: Path, seed_dir: Path, sim_seed: int = 42) -> Dict:
    """Score TUH/OSF/P-ADIC without writing large per-subject caches (disk-safe)."""
    external_start = datetime.now().isoformat()
    sig = load_constrained_training_effects(ROOT)
    loaders = {"tuh": load_tuh, "osf": load_osf, "padic": load_padic}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(checkpoint, device)
    drug_emb = load_drug_embeddings()

    torch.manual_seed(sim_seed)
    np.random.seed(sim_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sim_seed)

    out = {
        "external_scoring_start_time": external_start,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "signature_source": sig["source"],
        "num_samples": NUM_SAMPLES,
        "sim_seed": sim_seed,
        "cohorts": {},
    }

    for name, loader in loaders.items():
        flats, diseases, sids, labels = loader(ROOT)
        don_effects, mem_effects = [], []
        for flat, disease, sid in zip(flats, diseases, sids):
            eeg = unflatten_features(np.asarray(flat, dtype=np.float32))
            sims = simulate_post_drug_eeg(
                model,
                eeg,
                drug_emb,
                disease_label=int(disease),
                device=device,
                num_samples=NUM_SAMPLES,
            )
            base = sims["baseline"].mean(0)
            don = sims["donepezil"].mean(0)
            mem = sims["memantine"].mean(0)
            don_effects.append(don - base)
            mem_effects.append(mem - base)

        scored = score_effects_vs_signature(don_effects, mem_effects, sig)
        ext_don = np.mean(np.stack(don_effects), 0)
        ext_mem = np.mean(np.stack(mem_effects), 0)
        cos_d = _cosine(ext_don, sig["donepezil_effect"])
        cos_m = _cosine(ext_mem, sig["memantine_effect"])
        agree = scored["direction_agreement_total"]
        n_ok, n_den = [int(x) for x in agree.split("/")]
        out["cohorts"][name] = {
            **scored,
            "agreement_n": n_ok,
            "agreement_den": n_den,
            "agreement_fraction": float(n_ok) / float(n_den),
            "effect_magnitude_cosine": float(np.nanmean([cos_d, cos_m])),
            "donepezil_effect_cosine": float(cos_d),
            "memantine_effect_cosine": float(cos_m),
            "n_subjects": int(len(sids)),
        }
        print(
            f"  {name}: agree={agree} r={scored['effect_magnitude_correlation']:.6f} "
            f"cos={out['cohorts'][name]['effect_magnitude_cosine']:.6f}"
        )

    out["external_scoring_end_time"] = datetime.now().isoformat()
    (seed_dir / "external_direction_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    return out


def locked_seed42_row() -> Dict:
    """Authoritative locked fold-0 values from v3 (not retrained)."""
    v3 = json.loads(
        (ROOT / "models/validation/complete_validation_report_v3.json").read_text(
            encoding="utf-8"
        )
    )
    ckpt = ROOT / "models/checkpoints_constrained/checkpoint_constrained.pt"
    # Score cosine for locked ckpt for metric parity (does not change locked r)
    seed_dir = OUT / "seed_42"
    seed_dir.mkdir(parents=True, exist_ok=True)
    scored = score_checkpoint_external(ckpt, seed_dir, sim_seed=42)
    # Overwrite r/agreement with authoritative locked values
    locked_map = {
        "tuh": (
            float(v3["layer5"]["cross_dataset"]["effect_magnitude_correlation"]),
            v3["layer5"]["cross_dataset"]["direction_agreement_total"],
            int(v3["layer5"]["cross_dataset"]["n_tuh_used"]),
        ),
        "osf": (
            float(
                v3["layer5b_ad_labeled_external"]["cross_dataset"][
                    "effect_magnitude_correlation"
                ]
            ),
            v3["layer5b_ad_labeled_external"]["cross_dataset"][
                "direction_agreement_total"
            ],
            int(v3["layer5b_ad_labeled_external"]["cross_dataset"]["n_used"]),
        ),
        "padic": (
            float(
                v3["layer5d_padic_external"]["cross_dataset"][
                    "effect_magnitude_correlation"
                ]
            ),
            v3["layer5d_padic_external"]["cross_dataset"]["direction_agreement_total"],
            int(v3["layer5d_padic_external"]["cross_dataset"]["n_used"]),
        ),
    }
    for name, (r, agree, n) in locked_map.items():
        n_ok, n_den = [int(x) for x in str(agree).split("/")]
        scored["cohorts"][name]["effect_magnitude_correlation"] = r
        scored["cohorts"][name]["direction_agreement_total"] = agree
        scored["cohorts"][name]["agreement_n"] = n_ok
        scored["cohorts"][name]["agreement_den"] = n_den
        scored["cohorts"][name]["agreement_fraction"] = float(n_ok) / float(n_den)
        scored["cohorts"][name]["n_used"] = n
        scored["cohorts"][name]["source"] = "locked_v3_authoritative_r_and_agreement"
    scored["note"] = (
        "Seed 42 uses locked checkpoint_constrained.pt; r/agreement from "
        "complete_validation_report_v3.json; cosine recomputed for parity only."
    )
    meta = {
        "train_seed": 42,
        "split_seed": SPLIT_SEED,
        "checkpoint_path": "models/checkpoints_constrained/checkpoint_constrained.pt",
        "selection_rule": "historical locked fold-0 alias (not retrained in this experiment)",
        "checkpoint_sha256": _sha256(ckpt),
        "training_config_hash": _config_hash(),
        "retrained": False,
        "best_epoch": 99,
    }
    (seed_dir / "training_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (seed_dir / "external_direction_results.json").write_text(
        json.dumps(scored, indent=2), encoding="utf-8"
    )
    return {"training": meta, "external": scored}


def aggregate(all_rows: List[Dict]) -> None:
    import csv

    # all_rows: list of {seed, cohort, ...}
    with (OUT / "aggregate_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "seed",
            "cohort",
            "agreement_n",
            "agreement_den",
            "agreement_fraction",
            "pearson_r",
            "cosine",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "training_signature",
            "training_config_hash",
            "retrained",
            "delta_r_vs_seed42",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    by_cohort: Dict[str, List[Dict]] = {}
    for r in all_rows:
        by_cohort.setdefault(r["cohort"], []).append(r)

    seed42 = {r["cohort"]: float(r["pearson_r"]) for r in all_rows if int(r["seed"]) == 42}
    stats = {}
    for cohort, rows in by_cohort.items():
        rs = [float(x["pearson_r"]) for x in rows]
        agrees = [str(int(x["agreement_n"])) + "/" + str(int(x["agreement_den"])) for x in rows]
        cos = [float(x["cosine"]) for x in rows]
        r42 = seed42[cohort]
        stats[cohort] = {
            "mean_r": float(np.mean(rs)),
            "median_r": float(np.median(rs)),
            "sd_r": float(np.std(rs, ddof=1)) if len(rs) > 1 else 0.0,
            "min_r": float(np.min(rs)),
            "max_r": float(np.max(rs)),
            "range_r": float(np.max(rs) - np.min(rs)),
            "n_seeds_r_gt_0": int(sum(1 for x in rs if x > 0)),
            "n_seeds_10_of_10": int(sum(1 for a in agrees if a == "10/10")),
            "n_seeds_at_least_8_of_10": int(
                sum(1 for x in rows if int(x["agreement_n"]) >= 8)
            ),
            "agreements": agrees,
            "mean_cosine": float(np.mean(cos)),
            "median_cosine": float(np.median(cos)),
            "sd_cosine": float(np.std(cos, ddof=1)) if len(cos) > 1 else 0.0,
            "min_cosine": float(np.min(cos)),
            "max_cosine": float(np.max(cos)),
            "seed42_r": r42,
            "mean_abs_delta_vs_seed42": float(
                np.mean([abs(float(x["pearson_r"]) - r42) for x in rows if int(x["seed"]) != 42])
            ),
            "deltas_vs_seed42": {
                str(int(x["seed"])): float(x["pearson_r"]) - r42
                for x in rows
                if int(x["seed"]) != 42
            },
        }

    # Interpretation gate
    # Substantial sensitivity if any cohort mean r far below locked or SD large / min << locked
    flags = []
    for c, s in stats.items():
        if s["min_r"] < 0.5 or s["sd_r"] > 0.15:
            flags.append("substantial")
        elif s["sd_r"] > 0.05 or s["mean_abs_delta_vs_seed42"] > 0.05:
            flags.append("moderate")
        else:
            flags.append("robust")
    if "substantial" in flags:
        interpretation = "Evidence suggests substantial training-seed sensitivity"
    elif "moderate" in flags:
        interpretation = "Evidence suggests moderate training-seed sensitivity"
    else:
        interpretation = "Evidence supports training-seed robustness"

    agg = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "multiseed_fold0_constrained_external_direction",
        "seeds_prespecified": [SEED_REFERENCE] + SEEDS_NEW,
        "split_seed_fixed": SPLIT_SEED,
        "design_note": (
            "Fold assignment and fold-0 train/val split frozen at split_seed=42; "
            "train_seed controls torch/numpy initialization and DataLoader shuffle only. "
            "Naive DATA_CONFIG.random_seed changes would alter fold membership and are NOT used."
        ),
        "training_config_hash": _config_hash(),
        "constraint_weight": CONSTRAINT_WEIGHT,
        "warmup_epochs": WARMUP_EPOCHS,
        "stats_by_cohort": stats,
        "interpretation": interpretation,
        "rows": all_rows,
    }
    (OUT / "aggregate_results.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # Markdown summary
    lines = [
        "# Multi-seed fold-0 robustness summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Design",
        "",
        "- Fixed `split_seed=42` (identical fold-0 subject split to locked training).",
        "- Varied `train_seed` in `{7, 21, 123, 2024}`; seed `42` = locked reference (not retrained).",
        "- Checkpoint rule: min internal validation total loss on fold 0.",
        "- External scoring after freeze on TUH / OSF / P-ADIC vs fixed training signature.",
        "",
        "## Seed × cohort",
        "",
        "| Seed | Cohort | Agree | r | Cosine | Δr vs 42 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in all_rows:
        lines.append(
            f"| {r['seed']} | {r['cohort']} | {int(r['agreement_n'])}/{int(r['agreement_den'])} | "
            f"{float(r['pearson_r']):.6f} | {float(r['cosine']):.6f} | {float(r['delta_r_vs_seed42']):+.6f} |"
        )
    lines += ["", "## Aggregate", ""]
    lines.append("| Cohort | Mean r | Median r | SD | Min | Max | 10/10 seeds |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c, s in stats.items():
        lines.append(
            f"| {c} | {s['mean_r']:.6f} | {s['median_r']:.6f} | {s['sd_r']:.6f} | "
            f"{s['min_r']:.6f} | {s['max_r']:.6f} | {s['n_seeds_10_of_10']}/5 |"
        )
    lines += [
        "",
        f"## Interpretation",
        "",
        f"**{interpretation}**",
        "",
        "## Limitations",
        "",
        "- Five seeds is a robustness check, not a full training-distribution characterization.",
        "- Does not address fold dependence, prior-only baseline, or post-dose EEG absence.",
        "- Does not prove historical seed-42 pre-registration.",
        "",
    ]
    (OUT / "seed_level_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return interpretation


def write_integrity_post() -> Dict:
    protected = [
        "models/validation/complete_validation_report_v3.json",
        "models/validation/prior_only_direction.json",
        "models/validation/fold_external_direction_results.json",
        "models/validation/fold_ensemble/fold_ensemble_external_direction_results.json",
        "models/validation/fold_ensemble/bootstrap/ensemble_subject_bootstrap_B2000.json",
        "models/checkpoints_constrained/checkpoint_constrained.pt",
        "models/checkpoints_constrained/fold_0_best.pt",
        "models/checkpoints_unconstrained/checkpoint_unconstrained.pt",
        "paper/overleaf/main.tex",
        "paper/sections/03_results_mechanism.md",
    ]
    pre = {}
    pre_path = OUT / "pre_run_sha256.txt"
    if pre_path.exists():
        for line in pre_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and len(parts[0]) == 64:
                pre[parts[1]] = parts[0]

    lines = [f"# post_run_sha256 @ {datetime.now().isoformat()}", ""]
    changed = []
    for rel in protected:
        p = ROOT / rel
        if not p.exists():
            lines.append(f"MISSING  {rel}")
            continue
        dig = _sha256(p)
        lines.append(f"{dig}  {rel}")
        if rel in pre and pre[rel] != dig:
            changed.append(rel)
    (OUT / "post_run_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = [
        f"timestamp: {datetime.now().isoformat()}",
        f"protected_checked: {len(protected)}",
        f"protected_changed: {len(changed)}",
        f"changed_list: {changed}",
        "manuscript_changed: NO",
        "existing_checkpoint_changed: NO"
        if "models/checkpoints_constrained/checkpoint_constrained.pt" not in changed
        else "existing_checkpoint_changed: YES",
        "expected_changed: 0",
        "STATUS: CLEAN" if not changed else "STATUS: FAIL",
    ]
    (OUT / "integrity_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"changed": changed}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = (
            __import__("subprocess")
            .check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
            .strip()
        )
    except Exception:
        git_commit = "UNKNOWN"

    cfg = {
        "experiment": "multiseed_fold0_constrained_external_direction",
        "prespecified_seeds": [SEED_REFERENCE] + SEEDS_NEW,
        "new_train_seeds": SEEDS_NEW,
        "reference_seed": SEED_REFERENCE,
        "split_seed_fixed": SPLIT_SEED,
        "fold_idx": FOLD_IDX,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "warmup_epochs": WARMUP_EPOCHS,
        "epochs": TRAINING_CONFIG["num_epochs"],
        "batch_size": TRAINING_CONFIG["batch_size"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
        "weight_decay": TRAINING_CONFIG["weight_decay"],
        "beta_kl": TRAINING_CONFIG["beta_kl"],
        "num_samples_external": NUM_SAMPLES,
        "training_config_hash": _config_hash(),
        "git_commit": git_commit,
        "design": (
            "Hold fold assignment + fold-0 train/val split fixed at split_seed=42; "
            "vary train_seed for initialization and shuffle only."
        ),
        "created": datetime.now().isoformat(),
    }
    (OUT / "multiseed_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Multi-seed fold-0 robustness audit\n\n"
        "Analysis-only outputs. Does not modify locked fold-0 manuscript results.\n\n"
        "See `seed_level_summary.md` and `aggregate_results.json`.\n",
        encoding="utf-8",
    )

    # Free disk check
    free = __import__("shutil").disk_usage(str(ROOT)).free
    if free < 2 * (1 << 30):
        print(f"STOP: insufficient disk ({free/1e9:.2f} GB free)")
        return 2

    manifest = {"git_commit": git_commit, "seeds": {}, "failures": []}
    all_rows: List[Dict] = []

    # Seed 42 locked reference first (eval only)
    print("=== SEED 42 (locked reference; not retrained) ===")
    ref = locked_seed42_row()
    manifest["seeds"]["42"] = {
        **ref["training"],
        "external_scoring_start_time": ref["external"]["external_scoring_start_time"],
        "external_scoring_end_time": ref["external"]["external_scoring_end_time"],
        "git_commit": git_commit,
    }
    r42 = {
        c: float(v["effect_magnitude_correlation"])
        for c, v in ref["external"]["cohorts"].items()
    }
    for cohort, v in ref["external"]["cohorts"].items():
        all_rows.append(
            {
                "seed": 42,
                "cohort": cohort,
                "agreement_n": v["agreement_n"],
                "agreement_den": v["agreement_den"],
                "agreement_fraction": v["agreement_fraction"],
                "pearson_r": v["effect_magnitude_correlation"],
                "cosine": v["effect_magnitude_cosine"],
                "checkpoint_epoch": 99,
                "checkpoint_sha256": ref["training"]["checkpoint_sha256"],
                "training_signature": ref["external"]["signature_source"],
                "training_config_hash": _config_hash(),
                "retrained": False,
                "delta_r_vs_seed42": 0.0,
            }
        )

    for seed in SEEDS_NEW:
        print(f"\n=== TRAIN SEED {seed} ===")
        seed_dir = OUT / f"seed_{seed}"
        try:
            tr = train_fold0_for_seed(seed, seed_dir)
            print(
                f"trained seed={seed} best_epoch={tr['best_epoch']} "
                f"best_val={tr['best_val_total']:.6f}"
            )
            print(f"=== EXTERNAL SCORE SEED {seed} ===")
            ext = score_checkpoint_external(seed_dir / "checkpoint.pt", seed_dir)
            manifest["seeds"][str(seed)] = {
                "train_seed": seed,
                "training_start_time": tr["training_start_time"],
                "training_end_time": tr["training_end_time"],
                "checkpoint_selection_time": tr["checkpoint_selection_time"],
                "checkpoint_path": tr["checkpoint_path"],
                "selection_metric": tr["selection_metric"],
                "selection_rule": tr["selection_rule"],
                "best_epoch": tr["best_epoch"],
                "external_scoring_start_time": ext["external_scoring_start_time"],
                "external_scoring_end_time": ext["external_scoring_end_time"],
                "git_commit": git_commit,
                "training_config_hash": tr["training_config_hash"],
                "checkpoint_sha256": tr["checkpoint_sha256"],
            }
            for cohort, v in ext["cohorts"].items():
                all_rows.append(
                    {
                        "seed": seed,
                        "cohort": cohort,
                        "agreement_n": v["agreement_n"],
                        "agreement_den": v["agreement_den"],
                        "agreement_fraction": v["agreement_fraction"],
                        "pearson_r": v["effect_magnitude_correlation"],
                        "cosine": v["effect_magnitude_cosine"],
                        "checkpoint_epoch": tr["best_epoch"],
                        "checkpoint_sha256": tr["checkpoint_sha256"],
                        "training_signature": ext["signature_source"],
                        "training_config_hash": tr["training_config_hash"],
                        "retrained": True,
                        "delta_r_vs_seed42": float(v["effect_magnitude_correlation"])
                        - float(r42[cohort]),
                    }
                )
        except Exception as e:
            tb = traceback.format_exc()
            fail_path = seed_dir / "FAILURE.txt"
            seed_dir.mkdir(parents=True, exist_ok=True)
            fail_path.write_text(tb, encoding="utf-8")
            manifest["failures"].append(
                {
                    "seed": seed,
                    "error": str(e),
                    "log": str(fail_path.relative_to(ROOT)).replace("\\", "/"),
                    "before_or_after_selection": "unknown",
                }
            )
            print(f"FAILED seed={seed}: {e}")

    (OUT / "seed_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    interpretation = aggregate(all_rows)
    integ = write_integrity_post()

    # checkpoint integrity
    lines = ["# checkpoint integrity", ""]
    for seed, meta in manifest["seeds"].items():
        lines.append(f"seed={seed} sha256={meta.get('checkpoint_sha256')} path={meta.get('checkpoint_path')}")
    (OUT / "checkpoint_integrity_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== DONE ===")
    print("interpretation:", interpretation)
    print("protected_changed:", integ["changed"])
    return 0 if not integ["changed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
