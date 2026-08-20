"""
Phase C: ChemBERTa vs padded one-hot fold-0 constrained ablation.

Trains TWO matched fold-0 constrained twins into a NEW directory:
  models/validation/chemberta_onehot_ablation/{chemberta,onehot}/

Does NOT overwrite models/checkpoints_constrained/checkpoint_constrained.pt.
Does NOT touch locked validation JSONs.

One-hot conditioning uses orthogonal unit vectors padded to 384-D so the
DrugEncoder input_dim remains matched to the ChemBERTa arm (capacity control).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

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
from src.models.config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from src.models.data_loader import (
    create_data_loaders_for_fold,
    get_subject_level_kfold,
    load_drug_embeddings,
)
from src.models.train_phase2 import DigitalTwinModel, prepare_target_features  # noqa: F401
from src.validation.direction_metrics import (
    extract_baseline_mu,
    load_constrained_training_effects,
    score_effects_vs_signature,
    simulate_flat_cohort,
)
from src.validation.run_unconstrained_external_battery import (
    load_caueeg,
    load_osf,
    load_padic,
    load_tuh,
    latent_probe_auc,
    theta_alpha,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "models" / "validation" / "chemberta_onehot_ablation"
SCRIPT_PATH = Path(__file__).resolve()
DRUG_DIM = 384
CONSTRAINT_WEIGHT = 0.1
WARMUP_EPOCHS = 5


def make_onehot_embeddings(dim: int = DRUG_DIM) -> Dict[str, np.ndarray]:
    """Padded one-hot identifiers for donepezil / memantine (matched input dim)."""
    if dim < 2:
        raise ValueError("dim must be >= 2")
    don = np.zeros(dim, dtype=np.float32)
    mem = np.zeros(dim, dtype=np.float32)
    don[0] = 1.0
    mem[1] = 1.0
    return {"donepezil": don, "memantine": mem}


def train_fold0_arm(
    tag: str,
    drug_embeddings: Dict[str, np.ndarray],
    epochs: int,
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

    # Deterministic init for matched arms
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

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
        train_ids,
        val_ids,
        test_ids,
        batch_size=batch_size,
        shuffle_train=True,
        drug_embeddings=drug_embeddings,
    )

    model = DigitalTwinModel(config=MODEL_CONFIG).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
    )
    baseline_cache = _build_subject_baseline_cache(train_loader)
    best_val = float("inf")
    history = []
    t0 = time.time()

    print(f"\n=== arm={tag} fold0 epochs={epochs} alpha_c={CONSTRAINT_WEIGHT} ===")
    for epoch in range(epochs):
        tr = train_epoch_constrained(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            baseline_cache=baseline_cache,
            beta_kl=beta_kl,
            constraint_weight=float(CONSTRAINT_WEIGHT),
            warmup_epochs=WARMUP_EPOCHS,
            diagnostic_once={"printed": True},
        )
        va = validate_epoch_constrained(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            baseline_cache=baseline_cache,
            beta_kl=beta_kl,
            constraint_weight=float(CONSTRAINT_WEIGHT),
            warmup_epochs=WARMUP_EPOCHS,
        )
        history.append({"epoch": epoch + 1, "train": tr, "val": va})
        vtot = float(va["total"])
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{tag}] epoch {epoch+1}: val_total={vtot:.4f}")
        if vtot < best_val:
            best_val = vtot
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va,
                    "constraint_weight": float(CONSTRAINT_WEIGHT),
                    "warmup_epochs": WARMUP_EPOCHS,
                    "drug_conditioning": tag,
                    "phase_c_ablation": True,
                    "locked_checkpoint_untouched": True,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    rec = compute_reconstruction_metrics_fold(model, test_loader, device)
    acc = compute_accuracy_fold(train_loader, test_loader, model, device, random_seed=random_seed)
    meta = {
        "tag": tag,
        "constraint_weight": float(CONSTRAINT_WEIGHT),
        "warmup_epochs": WARMUP_EPOCHS,
        "epochs": epochs,
        "seconds": time.time() - t0,
        "best_val_total": best_val,
        "best_epoch": int(ckpt["epoch"]),
        "test_reconstruction_mse": rec["reconstruction_mse"],
        "test_pearson_global": rec["pearson_global"],
        "test_accuracy": acc,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "checkpoint": str(best_path),
        "random_seed": random_seed,
        "drug_embedding_dim": int(next(iter(drug_embeddings.values())).shape[0]),
        "drug_embedding_norms": {
            k: float(np.linalg.norm(v)) for k, v in drug_embeddings.items()
        },
    }
    (out_dir / "fold0_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "fold0_history_tail.json").write_text(
        json.dumps(history[-5:], indent=2), encoding="utf-8"
    )
    print(f"[saved] {best_path} mse={rec['reconstruction_mse']:.4f}")
    return best_path


def eval_arm(
    tag: str,
    ckpt: Path,
    drug_embeddings: Dict[str, np.ndarray],
    sim_root: Path,
) -> Dict:
    signature = load_constrained_training_effects(ROOT)
    out: Dict = {"tag": tag, "checkpoint": str(ckpt)}

    for name, loader in (
        ("tuh", load_tuh),
        ("osf", load_osf),
        ("padic", load_padic),
    ):
        flats, dis, sids, labs = loader(ROOT)
        sim = simulate_flat_cohort(
            flats,
            dis,
            sids,
            ckpt,
            sim_root / name,
            drug_embeddings=drug_embeddings,
        )
        out[f"{name}_direction"] = score_effects_vs_signature(
            sim["don_effects"], sim["mem_effects"], signature
        )

    flats, dis, sids, labs = load_caueeg(ROOT)
    mu = extract_baseline_mu(flats, ckpt, disease_label=0.0)
    y_all = np.asarray(labs)
    mask = np.isin(y_all, ["dementia", "normal"])
    y = (y_all[mask] == "dementia").astype(np.int64)
    out["caueeg_latent_probe"] = latent_probe_auc(mu[mask], y)
    out["caueeg_theta_alpha_probe"] = latent_probe_auc(theta_alpha(flats)[mask].reshape(-1, 1), y)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=int(TRAINING_CONFIG["num_epochs"]))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--skip-osf", action="store_true", help="unused; OSF always scored")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    chem_emb = load_drug_embeddings()
    onehot_emb = make_onehot_embeddings(DRUG_DIM)

    # Persist one-hot vectors used for this run (reproducibility)
    emb_dir = OUT_ROOT / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    np.save(emb_dir / "onehot_donepezil.npy", onehot_emb["donepezil"])
    np.save(emb_dir / "onehot_memantine.npy", onehot_emb["memantine"])
    (emb_dir / "onehot_metadata.json").write_text(
        json.dumps(
            {
                "scheme": "padded_onehot",
                "dim": DRUG_DIM,
                "donepezil_index": 0,
                "memantine_index": 1,
                "note": (
                    "Orthogonal unit vectors in first two dimensions; remaining "
                    "dims zero. Keeps DrugEncoder input_dim=384 matched to ChemBERTa."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    arms = {
        "chemberta": chem_emb,
        "onehot": onehot_emb,
    }
    ckpts = {}
    for tag, emb in arms.items():
        arm_dir = OUT_ROOT / tag
        if args.eval_only:
            ckpts[tag] = arm_dir / "fold_0_best.pt"
            if not ckpts[tag].exists():
                raise FileNotFoundError(ckpts[tag])
        else:
            ckpts[tag] = train_fold0_arm(
                tag=tag,
                drug_embeddings=emb,
                epochs=args.epochs,
                out_dir=arm_dir,
                force=args.force,
            )

    # Safety: locked path must still exist and be distinct
    locked = ROOT / "models" / "checkpoints_constrained" / "checkpoint_constrained.pt"
    for tag, p in ckpts.items():
        if p.resolve() == locked.resolve():
            raise RuntimeError(f"Ablation checkpoint collided with locked path: {p}")

    results = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "phase_c_chemberta_vs_onehot_fold0",
        "status": "NEW_NON_LOCKED",
        "script": str(SCRIPT_PATH),
        "constraint_weight": CONSTRAINT_WEIGHT,
        "warmup_epochs": WARMUP_EPOCHS,
        "epochs": args.epochs,
        "random_seed": DATA_CONFIG["random_seed"],
        "locked_checkpoint_untouched": str(locked),
        "matched_protocol": (
            "Same fold-0 split, seed 42, alpha_c=0.1, warmup=5, architecture, "
            "and epochs; only drug embedding source differs."
        ),
        "claim_boundary": (
            "Secondary ablation of chemical identifier vs drug-identity conditioning. "
            "Does not claim large-library ChemBERTa generalization. "
            "Locked fold-0 ChemBERTa twin remains the primary historical checkpoint."
        ),
        "arms": {},
    }

    sim_root = OUT_ROOT / "sims"
    for tag, emb in arms.items():
        print(f"\n=== evaluate {tag} ===")
        results["arms"][tag] = eval_arm(tag, ckpts[tag], emb, sim_root / tag)

    # Compact contrast table
    contrast = {}
    for cohort in ("tuh", "osf", "padic"):
        key = f"{cohort}_direction"
        c = results["arms"]["chemberta"][key]
        o = results["arms"]["onehot"][key]
        contrast[cohort] = {
            "chemberta_signs": c.get("direction_agreement_total"),
            "onehot_signs": o.get("direction_agreement_total"),
            "chemberta_r": c.get("effect_magnitude_correlation"),
            "onehot_r": o.get("effect_magnitude_correlation"),
            "r_diff_chemberta_minus_onehot": float(
                c.get("effect_magnitude_correlation", float("nan"))
                - o.get("effect_magnitude_correlation", float("nan"))
            ),
        }
    results["direction_contrast"] = contrast
    results["latent_probe_contrast"] = {
        "chemberta_auc_mean": float(
            results["arms"]["chemberta"]["caueeg_latent_probe"]["roc_auc_mean"]
        ),
        "onehot_auc_mean": float(
            results["arms"]["onehot"]["caueeg_latent_probe"]["roc_auc_mean"]
        ),
        "chemberta_auc_oof": float(
            results["arms"]["chemberta"]["caueeg_latent_probe"]["roc_auc_oof"]
        ),
        "onehot_auc_oof": float(
            results["arms"]["onehot"]["caueeg_latent_probe"]["roc_auc_oof"]
        ),
    }

    out_json = OUT_ROOT / "chemberta_onehot_results.json"
    out_md = OUT_ROOT / "chemberta_onehot_summary.md"
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Phase C ChemBERTa vs padded one-hot (fold-0 constrained)",
        "",
        f"- Status: `{results['status']}`",
        f"- Locked checkpoint untouched: `{locked}`",
        f"- Protocol: {results['matched_protocol']}",
        "",
        "## Direction contrast (vs locked training signature)",
        "",
    ]
    for cohort, row in contrast.items():
        lines.append(
            f"- **{cohort.upper()}**: ChemBERTa {row['chemberta_signs']} "
            f"(r={row['chemberta_r']:.6f}) vs one-hot {row['onehot_signs']} "
            f"(r={row['onehot_r']:.6f}); Δr={row['r_diff_chemberta_minus_onehot']:+.6f}"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            results["claim_boundary"],
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
