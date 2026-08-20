"""
Pharmacological prior + enhanced drug embedding utilities.

This module augments ChemBERTa structural embeddings (384-D) with a
hand-crafted pharmacodynamic prior vector (32-D) to produce a 416-D
mechanism-guided representation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.models.config import DATA_ROOT, DRUG_EMBEDDINGS_DIR, PROJECT_ROOT


# Pharmacology constants (hardcoded prior knowledge; not learned).
DRUG_PHARMACOLOGY = {
    "donepezil": {
        "smiles": "COc1cc2c(cc1OC)CC(CC2=O)Cc3ccc(cc3)OCc4ccccc4",
        "primary_target": "ACHE",  # Acetylcholinesterase target (Aricept label)
        "secondary_target": "BCHE",  # Butyrylcholinesterase secondary target
        "mechanism": "ache_inhibitor",
        # Binding affinities (Ki, nM) — ChEMBL CHEMBL502
        "ki_ache_nm": 6.7,
        "ki_bche_nm": 7400.0,
        "ki_nmda_nm": 9999.0,  # No significant NMDA binding (ChEMBL profile)
        # EEG effects — Babiloni et al. 2006; Jeong 2004 review
        "eeg_effects": {
            "delta": -0.15,
            "theta": -0.20,
            "alpha": +0.25,
            "beta": +0.10,
            "connectivity": +0.20,
        },
        # Neurotransmitter effects (normalized) — mechanism-based priors
        "acetylcholine_effect": +0.80,  # Strong ACh increase (AChE inhibition)
        "glutamate_effect": 0.00,  # Minimal direct glutamatergic effect
        "gaba_effect": 0.00,  # Minimal direct GABA effect
        "dopamine_effect": 0.10,  # Mild indirect modulation
        # Clinical PK — Aricept prescribing information; Tiseo et al. (1998)
        "half_life_hours": 70.0,
        "peak_effect_hours": 3.0,
        "standard_dose_mg": 10.0,
        "oral_bioavailability": 0.97,
    },
    "memantine": {
        "smiles": "CC12CC(CC(C1)(CC(C2)N)C)C",
        "primary_target": "GRIN2A",  # NMDA receptor subunit 2A
        "secondary_target": "GRIN2B",  # NMDA receptor subunit 2B
        "mechanism": "nmda_antagonist",
        # Binding affinities (Ki, nM) — ChEMBL CHEMBL807
        "ki_nmda_nm": 290.0,
        "ki_ache_nm": 9999.0,  # No meaningful AChE affinity (ChEMBL profile)
        # EEG effects — Winblad et al. 2007; Rive et al. 2013
        "eeg_effects": {
            "delta": -0.10,
            "theta": -0.15,
            "alpha": +0.10,
            "beta": +0.05,
            "connectivity": +0.15,
        },
        # Neurotransmitter priors — NMDA antagonism literature
        "acetylcholine_effect": 0.00,
        "glutamate_effect": -0.70,  # Primary glutamatergic attenuation
        "gaba_effect": +0.20,  # Indirect GABAergic modulation
        "dopamine_effect": +0.10,  # Mild indirect effect
        # Clinical PK — Namenda prescribing information
        "half_life_hours": 70.0,
        "peak_effect_hours": 4.0,
        "standard_dose_mg": 20.0,
        "oral_bioavailability": 0.99,
    },
    "baseline": {
        "smiles": None,
        "primary_target": None,
        "secondary_target": None,
        "mechanism": "no_drug",
        "ki_ache_nm": 9999.0,
        "ki_nmda_nm": 9999.0,
        "eeg_effects": {
            "delta": 0.0,
            "theta": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "connectivity": 0.0,
        },
        "acetylcholine_effect": 0.0,
        "glutamate_effect": 0.0,
        "gaba_effect": 0.0,
        "dopamine_effect": 0.0,
        "half_life_hours": 0.0,
        "peak_effect_hours": 0.0,
        "standard_dose_mg": 0.0,
        "oral_bioavailability": 0.0,
    },
}

MECHANISMS = ["ache_inhibitor", "nmda_antagonist", "combination", "no_drug", "other"]
DRUGS = ["baseline", "donepezil", "memantine"]


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _safe_log_ki(ki_nm: float) -> float:
    return float(np.log10(max(float(ki_nm), 0.0) + 1.0))


def build_pharmacodynamic_vector(drug_name: str) -> np.ndarray:
    """
    Convert a DRUG_PHARMACOLOGY entry into a 32-D fixed prior vector.
    """
    key = drug_name.strip().lower()
    if key not in DRUG_PHARMACOLOGY:
        raise ValueError(f"Unknown drug_name: {drug_name}")
    d = DRUG_PHARMACOLOGY[key]

    vec = np.zeros(32, dtype=np.float32)

    # [0:5] EEG priors
    eeg = d["eeg_effects"]
    vec[0:5] = np.array(
        [eeg["delta"], eeg["theta"], eeg["alpha"], eeg["beta"], eeg["connectivity"]],
        dtype=np.float32,
    )

    # [5:9] Neurotransmitter effects
    vec[5:9] = np.array(
        [
            d["acetylcholine_effect"],
            d["glutamate_effect"],
            d["gaba_effect"],
            d["dopamine_effect"],
        ],
        dtype=np.float32,
    )

    # [9:11] Log Ki values
    vec[9] = _safe_log_ki(d.get("ki_ache_nm", 9999.0))
    vec[10] = _safe_log_ki(d.get("ki_nmda_nm", 9999.0))

    # [11:15] Clinical properties normalized [0, 1]
    vec[11] = np.clip(float(d["half_life_hours"]) / 100.0, 0.0, 1.0)
    vec[12] = np.clip(float(d["peak_effect_hours"]) / 24.0, 0.0, 1.0)
    vec[13] = np.clip(float(d["standard_dose_mg"]) / 100.0, 0.0, 1.0)
    vec[14] = np.clip(float(d["oral_bioavailability"]), 0.0, 1.0)

    # [15:20] Mechanism one-hot
    mechanism = str(d.get("mechanism", "other"))
    mechanism_idx = MECHANISMS.index(mechanism) if mechanism in MECHANISMS else MECHANISMS.index("other")
    vec[15 + mechanism_idx] = 1.0

    # [20:32] reserved zeros
    return vec.astype(np.float32)


def _load_chemberta_embeddings(chemberta_dir: str | Path) -> Dict[str, np.ndarray]:
    """
    Robust loader for existing ChemBERTa embeddings.
    Supports:
    - per-drug files: donepezil_embedding.npy / memantine_embedding.npy
    - stacked array + metadata (drug_embeddings.npy + metadata json)
    - dict-like object array (fallback)
    """
    root = _resolve_path(chemberta_dir)
    if not root.exists():
        raise FileNotFoundError(f"ChemBERTa directory not found: {root}")

    out: Dict[str, np.ndarray] = {}

    # First: individual per-drug files
    for drug in ("donepezil", "memantine"):
        p = root / f"{drug}_embedding.npy"
        if p.exists():
            emb = np.load(p, allow_pickle=True).astype(np.float32).reshape(-1)
            out[drug] = emb

    # Second: stacked embeddings + metadata
    stacked = root / "drug_embeddings.npy"
    metadata = root / "drug_embeddings_metadata.json"
    if stacked.exists():
        arr = np.load(stacked, allow_pickle=True)
        if arr.dtype == object:
            try:
                obj = arr.item()
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        out[str(k).lower()] = np.asarray(v, dtype=np.float32).reshape(-1)
            except Exception:
                pass
        elif arr.ndim == 2 and metadata.exists():
            md = json.loads(metadata.read_text(encoding="utf-8"))
            keys = [str(k).lower() for k in md.get("drug_keys", [])]
            for i, key in enumerate(keys):
                if i < arr.shape[0]:
                    out[key] = np.asarray(arr[i], dtype=np.float32).reshape(-1)

    for required in ("donepezil", "memantine"):
        if required not in out:
            raise KeyError(f"Missing ChemBERTa embedding for {required} in {root}")
        if out[required].shape[0] != 384:
            raise ValueError(f"{required} embedding must be 384-D, got {out[required].shape}")

    return out


def get_enhanced_drug_embedding(
    drug_name: str,
    chemberta_dir: str | Path = "data/drug_embeddings",
) -> np.ndarray:
    """
    Build 416-D enhanced embedding = [ChemBERTa 384-D, pharma prior 32-D].
    """
    key = drug_name.strip().lower()
    if key not in DRUG_PHARMACOLOGY:
        raise ValueError(f"Unknown drug_name: {drug_name}")

    chem_root = _resolve_path(chemberta_dir)
    enhanced_root = chem_root / "enhanced"
    enhanced_root.mkdir(parents=True, exist_ok=True)

    chem = np.zeros(384, dtype=np.float32)
    if key != "baseline":
        embeddings = _load_chemberta_embeddings(chem_root)
        chem = embeddings[key]

    prior = build_pharmacodynamic_vector(key)
    enhanced = np.concatenate([chem, prior], axis=0).astype(np.float32)
    if enhanced.shape != (416,):
        raise ValueError(f"Enhanced embedding must be (416,), got {enhanced.shape}")

    np.save(enhanced_root / f"{key}_enhanced.npy", enhanced)
    np.save(enhanced_root / f"{key}_pharma_prior.npy", prior)

    meta_path = enhanced_root / "enhanced_metadata.json"
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    metadata.update(
        {
            "chemberta_dim": 384,
            "pharma_prior_dim": 32,
            "enhanced_dim": 416,
            "drugs": DRUGS,
            "source_files": {
                "chemberta_dir": str(chem_root),
                "tvb_priors": str(DATA_ROOT / "tvb_validation" / "tvb_ground_truth_summary.json"),
            },
            "vector_layout": {
                "0:5": "eeg_effect_priors",
                "5:9": "neurotransmitter_effects",
                "9:11": "log_binding_affinity",
                "11:15": "clinical_properties",
                "15:20": "mechanism_one_hot",
                "20:32": "reserved",
            },
        }
    )
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return enhanced


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _make_matrix(vectors: Dict[str, np.ndarray], keys: Tuple[str, ...]) -> np.ndarray:
    n = len(keys)
    mat = np.zeros((n, n), dtype=np.float32)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            mat[i, j] = _cosine_similarity(vectors[ki], vectors[kj])
    return mat


def _print_similarity_matrix(title: str, keys: Tuple[str, ...], matrix: np.ndarray) -> None:
    print(title)
    print(f"{'':14}{keys[0].title():>10}{keys[1].title():>12}{keys[2].title():>12}")
    for i, k in enumerate(keys):
        print(
            f"{k.title():14}"
            f"{matrix[i, 0]:10.3f}"
            f"{matrix[i, 1]:12.3f}"
            f"{matrix[i, 2]:12.3f}"
        )


def compute_drug_similarity_matrix(
    chemberta_dir: str | Path = "data/drug_embeddings",
) -> dict:
    """
    Compute pairwise cosine similarities for chemical, prior, and combined vectors.
    """
    keys = ("baseline", "donepezil", "memantine")
    chem_root = _resolve_path(chemberta_dir)

    enhanced = {k: get_enhanced_drug_embedding(k, chem_root) for k in keys}
    chemical = {k: enhanced[k][:384] for k in keys}
    pharma = {k: enhanced[k][384:] for k in keys}

    chem_mat = _make_matrix(chemical, keys)
    pharma_mat = _make_matrix(pharma, keys)
    comb_mat = _make_matrix(enhanced, keys)

    print("\nDRUG SIMILARITY ANALYSIS")
    print("=" * 48)
    _print_similarity_matrix("Chemical similarity (ChemBERTa 384-dim):", keys, chem_mat)
    print()
    _print_similarity_matrix("Pharmacodynamic similarity (prior 32-dim):", keys, pharma_mat)
    print()
    _print_similarity_matrix("Combined similarity (enhanced 416-dim):", keys, comb_mat)

    chem_sim = float(chem_mat[1, 2])
    pharma_sim = float(pharma_mat[1, 2])
    if chem_sim > pharma_sim:
        print("\n[WARNING] Donepezil-Memantine chemical similarity > pharmacodynamic similarity.")
    print("=" * 48)

    return {
        "drug_order": list(keys),
        "chemical": chem_mat,
        "pharmacodynamic": pharma_mat,
        "combined": comb_mat,
    }


def get_eeg_effect_prior(drug_name: str) -> dict:
    """
    Return EEG effect prior dict for use in biophysical loss constraints.
    """
    key = drug_name.strip().lower()
    if key not in DRUG_PHARMACOLOGY:
        raise ValueError(f"Unknown drug_name: {drug_name}")
    return dict(DRUG_PHARMACOLOGY[key]["eeg_effects"])
