"""
Constraint-weight (alpha_c) fold-0 sweep for Plan-to-8.3 B3.

Trains only fold 0 (matches external checkpoint selection rule) for each
constraint_weight, then evaluates direction (TUH+P-ADIC) and CAUEEG latent probe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_constrained_training import (  # noqa: E402
    _build_subject_baseline_cache,
    train_epoch_constrained,
    validate_epoch_constrained,
)
from run_5fold_cv import (  # noqa: E402
    compute_accuracy_fold,
    compute_reconstruction_metrics_fold,
)
from src.models.train_phase2 import (  # noqa: E402
    DigitalTwinModel,
    prepare_target_features,
    train_epoch,
    validate,
)
from src.models.config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from src.models.data_loader import create_data_loaders_for_fold, get_subject_level_kfold
from src.validation.direction_metrics import (
    extract_baseline_mu,
    load_constrained_training_effects,
    score_effects_vs_signature,
    simulate_flat_cohort,
)
from src.validation.run_unconstrained_external_battery import (
    load_caueeg,
    load_padic,
    load_tuh,
    latent_probe_auc,
    theta_alpha,
)


DEFAULT_WEIGHTS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def train_fold0(
    constraint_weight: float,
    epochs: int,
    warmup_epochs: int,
    out_dir: Path,
    force: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "fold_0_best.pt"
    if best_path.exists() and not force:
        print(f"[cache] {best_path}")
        return best_path

    device = torch.device(TRAINING_CONFIG["device"])
    batch_size = TRAINING_CONFIG["batch_size"]
    random_seed = DATA_CONFIG["random_seed"]
    beta_kl = TRAINING_CONFIG["beta_kl"]

    subject_ids, labels, test_folds = get_subject_level_kfold(n_splits=5, random_seed=random_seed)
    fold_idx = 0
    test_ids = test_folds[fold_idx].tolist()
    train_val_ids = np.concatenate([test_folds[j] for j in range(5) if j != fold_idx])
    train_ids, val_ids = train_test_split(
        train_val_ids.tolist(),
        test_size=0.1,
        stratify=labels[train_val_ids],
        random_state=random_seed + fold_idx,
    )
    train_loader, val_loader, test_loader = create_data_loaders_for_fold(
        train_ids, val_ids, test_ids, batch_size=batch_size, shuffle_train=True
    )

    model = DigitalTwinModel(config=MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )

    use_constraint = float(constraint_weight) > 0.0
    baseline_cache = _build_subject_baseline_cache(train_loader) if use_constraint else None
    best_val = float("inf")
    history = []
    t0 = time.time()

    print(f"\n=== alpha_c={constraint_weight} fold0 epochs={epochs} ===")
    for epoch in range(epochs):
        if use_constraint:
            tr = train_epoch_constrained(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                baseline_cache=baseline_cache,
                beta_kl=beta_kl,
                constraint_weight=float(constraint_weight),
                warmup_epochs=warmup_epochs,
                diagnostic_once={"printed": True},
            )
            va = validate_epoch_constrained(
                model=model,
                val_loader=val_loader,
                device=device,
                epoch=epoch,
                baseline_cache=baseline_cache,
                beta_kl=beta_kl,
                constraint_weight=float(constraint_weight),
                warmup_epochs=warmup_epochs,
            )
        else:
            cfg = {
                "beta_kl": beta_kl,
                "beta_condition": TRAINING_CONFIG.get("beta_condition", 0.0),
                "use_condition_loss": TRAINING_CONFIG.get("use_condition_loss", False),
            }
            train_epoch(model, train_loader, optimizer, device, cfg)
            va = validate(model, val_loader, device, cfg)
            tr = {"mse": float("nan"), "kl": float("nan"), "band": 0.0, "conn": 0.0, "total": float("nan")}
            # normalize validate dict
            if "total" not in va and "loss" in va:
                va = {"total": va["loss"], **va}

        history.append({"epoch": epoch + 1, "train": tr, "val": va})
        vtot = float(va["total"])
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}: val_total={vtot:.4f}")
        if vtot < best_val:
            best_val = vtot
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va,
                    "constraint_weight": float(constraint_weight),
                    "warmup_epochs": warmup_epochs,
                },
                best_path,
            )

    # test metrics on best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    rec = compute_reconstruction_metrics_fold(model, test_loader, device)
    acc = compute_accuracy_fold(train_loader, test_loader, model, device, random_seed=random_seed)
    meta = {
        "constraint_weight": float(constraint_weight),
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "seconds": time.time() - t0,
        "best_val_total": best_val,
        "test_reconstruction_mse": rec["reconstruction_mse"],
        "test_pearson_global": rec["pearson_global"],
        "test_accuracy": acc,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "checkpoint": str(best_path),
    }
    (out_dir / "fold0_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "fold0_history.json").write_text(json.dumps(history[-5:], indent=2), encoding="utf-8")
    print(f"[saved] {best_path} mse={rec['reconstruction_mse']:.4f}")
    return best_path


def eval_checkpoint(ckpt: Path, root: Path, tag: str) -> Dict:
    signature = load_constrained_training_effects(root)
    sim_root = root / "models" / "validation" / "alpha_c_sims" / tag
    out = {"checkpoint": str(ckpt), "tag": tag}

    flats, dis, sids, labs = load_tuh(root)
    sim = simulate_flat_cohort(flats, dis, sids, ckpt, sim_root / "tuh")
    out["tuh_direction"] = score_effects_vs_signature(sim["don_effects"], sim["mem_effects"], signature)

    flats, dis, sids, labs = load_padic(root)
    sim = simulate_flat_cohort(flats, dis, sids, ckpt, sim_root / "padic")
    out["padic_direction"] = score_effects_vs_signature(sim["don_effects"], sim["mem_effects"], signature)

    flats, dis, sids, labs = load_caueeg(root)
    mu = extract_baseline_mu(flats, ckpt, disease_label=0.0)
    y_all = np.asarray(labs)
    mask = np.isin(y_all, ["dementia", "normal"])
    y = (y_all[mask] == "dementia").astype(np.int64)
    out["caueeg_latent_probe"] = latent_probe_auc(mu[mask], y)
    out["caueeg_theta_alpha_probe"] = latent_probe_auc(theta_alpha(flats)[mask].reshape(-1, 1), y)

    # reconstruction already in meta; optional
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=",".join(str(w) for w in DEFAULT_WEIGHTS))
    ap.add_argument("--epochs", type=int, default=int(TRAINING_CONFIG["num_epochs"]))
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    root = _root()
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    ckpt_root = root / "models" / "checkpoints_alpha_c"
    results = {
        "timestamp": datetime.now().isoformat(),
        "weights": weights,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "selection_rule": "fold_0_best only (matched to constrained external checkpoint)",
        "points": [],
    }

    for w in weights:
        tag = f"alpha_{str(w).replace('.', 'p')}"
        out_dir = ckpt_root / tag
        if not args.eval_only:
            ckpt = train_fold0(w, args.epochs, args.warmup_epochs, out_dir, force=args.force)
        else:
            ckpt = out_dir / "fold_0_best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
        point = {"constraint_weight": w, "checkpoint": str(ckpt)}
        meta_path = out_dir / "fold0_train_meta.json"
        if meta_path.exists():
            point["train_meta"] = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if not args.train_only:
            print(f"=== eval alpha_c={w} ===")
            point["external"] = eval_checkpoint(ckpt, root, tag)
        results["points"].append(point)
        # incremental save
        out_json = root / "models" / "validation" / "constraint_strength_sweep.json"
        out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"[checkpoint save] {out_json}")

    print("[done] constraint strength sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
