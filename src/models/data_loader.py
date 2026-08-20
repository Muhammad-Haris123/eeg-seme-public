"""
Data Loading Utilities for Phase 2 Training.

Loads preprocessed EEG features, drug embeddings, and creates data loaders.
Supports subject-level train/val/test split (no subject in more than one split).

Author: Research Team
Date: 2026
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set

from .config import (
    PROCESSED_EEG_DIR,
    EEG_FEATURES_DIR,
    DRUG_EMBEDDINGS_DIR,
    DISEASE_LABELS,
    DRUG_IDS
)


class EEGDrugDataset(Dataset):
    """
    Dataset for Phase 2 training.
    
    Combines:
    - EEG features (PSD, band powers, connectivity)
    - Drug embeddings
    - Disease labels
    """
    
    def __init__(
        self,
        group: str,
        drug_embeddings: Dict[str, np.ndarray],
        use_all_drugs: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            group: 'AD' or 'HC'
            drug_embeddings: Dictionary mapping drug names to embeddings
            use_all_drugs: If True, create samples for all drugs; if False, use no-drug baseline
        """
        self.group = group
        self.disease_label = DISEASE_LABELS[group]
        
        # Load EEG features
        self.psd = np.load(EEG_FEATURES_DIR / f"{group}_psd.npy")
        self.band_powers = np.load(EEG_FEATURES_DIR / f"{group}_band_powers.npy")
        self.coherence = np.load(EEG_FEATURES_DIR / f"{group}_coherence.npy")
        self.plv = np.load(EEG_FEATURES_DIR / f"{group}_plv.npy")
        
        self.n_subjects = self.psd.shape[0]
        
        # Drug embeddings
        self.drug_embeddings = drug_embeddings
        self.drug_names = list(drug_embeddings.keys())
        
        # Create samples: (subject, drug) pairs
        if use_all_drugs:
            # For each subject, create samples with each drug + no-drug baseline
            self.samples = []
            for subj_idx in range(self.n_subjects):
                # No-drug baseline (use zero vector or mean drug embedding)
                self.samples.append((subj_idx, None))
                # Each drug
                for drug_name in self.drug_names:
                    self.samples.append((subj_idx, drug_name))
        else:
            # Only no-drug baseline
            self.samples = [(i, None) for i in range(self.n_subjects)]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dictionary with:
            - eeg_features: All EEG features
            - drug_embedding: Drug embedding (or zero if no drug)
            - disease_label: Disease label (AD=1, HC=0)
            - drug_id: Drug ID (-1 for no drug)
        """
        subj_idx, drug_name = self.samples[idx]
        
        # EEG features
        psd = torch.FloatTensor(self.psd[subj_idx])  # (channels, frequencies)
        band_powers = torch.FloatTensor(self.band_powers[subj_idx])  # (channels, bands)
        coherence = torch.FloatTensor(self.coherence[subj_idx])  # (bands, channels, channels)
        plv = torch.FloatTensor(self.plv[subj_idx])  # (bands, channels, channels)
        
        # Drug embedding
        if drug_name is None:
            # No-drug baseline: use zero vector
            drug_embedding = torch.zeros(384, dtype=torch.float32)  # ChemBERTa dimension
            drug_id = -1
        else:
            drug_embedding = torch.FloatTensor(self.drug_embeddings[drug_name])
            drug_id = DRUG_IDS.get(drug_name, -1)
        
        # Disease label
        disease_label = torch.tensor(self.disease_label, dtype=torch.float32)
        
        return {
            'psd': psd,
            'band_powers': band_powers,
            'coherence': coherence,
            'plv': plv,
            'drug_embedding': drug_embedding,
            'disease_label': disease_label,
            'drug_id': drug_id,
            'subject_idx': subj_idx
        }


class EEGDrugDatasetUnified(Dataset):
    """
    Unified dataset: all subjects from AD and HC with global subject IDs.
    Each subject has 3 samples: baseline, donepezil, memantine.
    global_subject_id: 0 .. n_ad-1 = AD, n_ad .. n_ad+n_hc-1 = HC.
    """

    def __init__(self, drug_embeddings: Dict[str, np.ndarray]):
        self.drug_embeddings = drug_embeddings
        self.drug_names = list(drug_embeddings.keys())
        # Load all group data
        self.psd_ad = np.load(EEG_FEATURES_DIR / "AD_psd.npy")
        self.psd_hc = np.load(EEG_FEATURES_DIR / "HC_psd.npy")
        self.bp_ad = np.load(EEG_FEATURES_DIR / "AD_band_powers.npy")
        self.bp_hc = np.load(EEG_FEATURES_DIR / "HC_band_powers.npy")
        self.coh_ad = np.load(EEG_FEATURES_DIR / "AD_coherence.npy")
        self.coh_hc = np.load(EEG_FEATURES_DIR / "HC_coherence.npy")
        self.plv_ad = np.load(EEG_FEATURES_DIR / "AD_plv.npy")
        self.plv_hc = np.load(EEG_FEATURES_DIR / "HC_plv.npy")
        self.n_ad = self.psd_ad.shape[0]
        self.n_hc = self.psd_hc.shape[0]
        self.n_subjects = self.n_ad + self.n_hc
        # Samples: (global_subject_id, drug_name) for each subject and each drug
        self.samples = []
        for gid in range(self.n_subjects):
            self.samples.append((gid, None))
            for drug_name in self.drug_names:
                self.samples.append((gid, drug_name))

    def _get_group_arrays(self, global_id: int):
        if global_id < self.n_ad:
            return (
                self.psd_ad, self.bp_ad, self.coh_ad, self.plv_ad,
                DISEASE_LABELS["AD"], global_id
            )
        else:
            local_idx = global_id - self.n_ad
            return (
                self.psd_hc, self.bp_hc, self.coh_hc, self.plv_hc,
                DISEASE_LABELS["HC"], local_idx
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_id, drug_name = self.samples[idx]
        psd_arr, bp_arr, coh_arr, plv_arr, disease_label, local_idx = self._get_group_arrays(global_id)
        psd = torch.FloatTensor(psd_arr[local_idx])
        band_powers = torch.FloatTensor(bp_arr[local_idx])
        coherence = torch.FloatTensor(coh_arr[local_idx])
        plv = torch.FloatTensor(plv_arr[local_idx])
        if drug_name is None:
            drug_embedding = torch.zeros(384, dtype=torch.float32)
            drug_id = -1
        else:
            drug_embedding = torch.FloatTensor(self.drug_embeddings[drug_name])
            drug_id = DRUG_IDS.get(drug_name, -1)
        return {
            "psd": psd,
            "band_powers": band_powers,
            "coherence": coherence,
            "plv": plv,
            "drug_embedding": drug_embedding,
            "disease_label": torch.tensor(disease_label, dtype=torch.float32),
            "drug_id": drug_id,
            "subject_idx": global_id,
        }


class SubsetBySubjectDataset(Dataset):
    """Wrapper that only includes samples whose global subject_id is in allowed_subject_ids."""

    def __init__(self, unified_dataset: EEGDrugDatasetUnified, allowed_subject_ids: Set[int]):
        self.base = unified_dataset
        self.allowed = allowed_subject_ids
        self.indices = [i for i in range(len(unified_dataset)) if unified_dataset.samples[i][0] in allowed_subject_ids]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base[self.indices[idx]]


class AllSubjectsBaselineDataset(Dataset):
    """One sample per subject (baseline only). Used for full-cohort inference."""

    def __init__(self, unified_dataset: EEGDrugDatasetUnified):
        self.base = unified_dataset
        self.indices = [i for i in range(len(unified_dataset)) if unified_dataset.samples[i][1] is None]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.base[self.indices[idx]]


def get_subject_level_splits(
    n_ad: int,
    n_hc: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Stratified split of subject IDs (global 0..n_ad-1 AD, n_ad..n_ad+n_hc-1 HC).
    Returns (train_ids, val_ids, test_ids) as lists of global subject IDs.
    """
    from sklearn.model_selection import train_test_split
    subject_ids = np.arange(n_ad + n_hc)
    labels = np.array([1] * n_ad + [0] * n_hc)
    # First split: train vs (val+test)
    id_train, id_rest = train_test_split(
        subject_ids,
        test_size=(1 - train_split),
        stratify=labels,
        random_state=random_seed,
    )
    id_rest_arr = np.asarray(id_rest)
    labels_rest = labels[id_rest_arr]
    # Second split: val vs test on rest (stratified)
    val_ratio = val_split / (val_split + test_split)
    id_val, id_test = train_test_split(
        id_rest_arr,
        test_size=(1 - val_ratio),
        stratify=labels_rest,
        random_state=random_seed + 1,
    )
    return list(id_train), list(id_val), list(id_test)


def get_subject_level_kfold(
    n_splits: int = 5,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Stratified K-fold split of subject IDs (global 0..65). Each subject appears in
    exactly one test fold. Same convention: 0..n_ad-1 = AD (label 1), n_ad..n_ad+n_hc-1 = HC (label 0).

    Returns:
        subject_ids: (66,) all subject IDs
        labels: (66,) group labels AD=1, HC=0
        test_folds: list of n_splits arrays; test_folds[k] = subject IDs for test in fold k
    """
    from sklearn.model_selection import StratifiedKFold
    drug_embeddings = load_drug_embeddings()
    unified = EEGDrugDatasetUnified(drug_embeddings)
    n_ad, n_hc = unified.n_ad, unified.n_hc
    subject_ids = np.arange(n_ad + n_hc)
    labels = np.array([1] * n_ad + [0] * n_hc)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    test_folds = []
    for _, test_idx in skf.split(subject_ids, labels):
        test_folds.append(subject_ids[test_idx])
    return subject_ids, labels, test_folds


def create_data_loaders_for_fold(
    train_subject_ids: List[int],
    val_subject_ids: List[int],
    test_subject_ids: List[int],
    batch_size: int = 16,
    shuffle_train: bool = True,
    drop_last: bool = False,
    drug_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test loaders from explicit subject ID lists (e.g. for one CV fold).
    No subject appears in more than one of train/val/test.
    test_subject_ids may be empty (e.g. for final-model training).
    drop_last: If True, drop last incomplete train batch (avoids BatchNorm error when batch_size=1).
    drug_embeddings: optional override (e.g. one-hot ablation); default ChemBERTa .npy store.
    """
    if drug_embeddings is None:
        drug_embeddings = load_drug_embeddings()
    unified = EEGDrugDatasetUnified(drug_embeddings)
    train_dataset = SubsetBySubjectDataset(unified, set(train_subject_ids))
    val_dataset = SubsetBySubjectDataset(unified, set(val_subject_ids))
    test_dataset = SubsetBySubjectDataset(unified, set(test_subject_ids)) if test_subject_ids else SubsetBySubjectDataset(unified, set())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return train_loader, val_loader, test_loader


def load_drug_embeddings() -> Dict[str, np.ndarray]:
    """
    Load drug embeddings from Phase 1.
    
    Returns:
        Dictionary mapping drug names to embeddings
    """
    drug_embeddings_path = DRUG_EMBEDDINGS_DIR / "drug_embeddings.npy"
    drug_metadata_path = DRUG_EMBEDDINGS_DIR / "drug_embeddings_metadata.json"
    
    drug_embeddings_array = np.load(drug_embeddings_path)
    
    with open(drug_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    drug_keys = metadata['drug_keys']
    
    drug_embeddings = {}
    for i, drug_key in enumerate(drug_keys):
        drug_embeddings[drug_key] = drug_embeddings_array[i]
    
    return drug_embeddings


def create_data_loaders(
    batch_size: int = 16,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
    shuffle: bool = True,
    split_by_subject: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.

    When split_by_subject=True (default): subjects are split stratified by group (AD/HC);
    no subject appears in more than one of train/val/test. Each subject has 3 samples
    (baseline, donepezil, memantine).

    When split_by_subject=False: legacy sample-level random split (same subject may appear
    in train and test under different drug conditions).

    Args:
        batch_size: Batch size
        train_split: Training split ratio (e.g. 0.8)
        val_split: Validation split ratio (e.g. 0.1)
        test_split: Test split ratio (e.g. 0.1)
        random_seed: Random seed for reproducibility
        shuffle: Whether to shuffle training data
        split_by_subject: If True, split by subject (stratified); if False, random split by sample

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    drug_embeddings = load_drug_embeddings()

    if split_by_subject:
        unified = EEGDrugDatasetUnified(drug_embeddings)
        n_ad = unified.n_ad
        n_hc = unified.n_hc
        train_ids, val_ids, test_ids = get_subject_level_splits(
            n_ad, n_hc, train_split, val_split, test_split, random_seed
        )
        train_dataset = SubsetBySubjectDataset(unified, set(train_ids))
        val_dataset = SubsetBySubjectDataset(unified, set(val_ids))
        test_dataset = SubsetBySubjectDataset(unified, set(test_ids))
    else:
        ad_dataset = EEGDrugDataset("AD", drug_embeddings, use_all_drugs=True)
        hc_dataset = EEGDrugDataset("HC", drug_embeddings, use_all_drugs=True)
        combined_dataset = torch.utils.data.ConcatDataset([ad_dataset, hc_dataset])
        total_size = len(combined_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            combined_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return train_loader, val_loader, test_loader


def create_all_subjects_loader(batch_size: int = 16) -> DataLoader:
    """
    Create a data loader that iterates over all subjects once (baseline sample only).
    Used for full-cohort inference (latents_full, simulations_full).
    """
    drug_embeddings = load_drug_embeddings()
    unified = EEGDrugDatasetUnified(drug_embeddings)
    dataset = AllSubjectsBaselineDataset(unified)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )


def get_split_info(
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
) -> Dict:
    """
    Return subject-level split info (counts and subject IDs) for metadata.
    Uses same stratification as create_data_loaders(split_by_subject=True).
    """
    drug_embeddings = load_drug_embeddings()
    unified = EEGDrugDatasetUnified(drug_embeddings)
    n_ad, n_hc = unified.n_ad, unified.n_hc
    train_ids, val_ids, test_ids = get_subject_level_splits(
        n_ad, n_hc, train_split, val_split, test_split, random_seed
    )
    n_ad_test = sum(1 for i in test_ids if i < n_ad)
    n_hc_test = sum(1 for i in test_ids if i >= n_ad)
    return {
        "split": "subject-level",
        "n_subjects_total": n_ad + n_hc,
        "n_ad": n_ad,
        "n_hc": n_hc,
        "n_train_subjects": len(train_ids),
        "n_val_subjects": len(val_ids),
        "n_test_subjects": len(test_ids),
        "n_AD_test": n_ad_test,
        "n_HC_test": n_hc_test,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "random_seed": random_seed,
        "test_subject_global_ids": test_ids,
    }

