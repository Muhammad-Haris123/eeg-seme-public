"""
Regional coherence from full coherence matrices.

Computes mean coherence over defined channel regions (e.g. alpha occipital,
beta frontal) for AD vs HC comparison. Uses the same 19-channel layout and
band definitions as the pipeline (delta, theta, alpha, beta, gamma).

Author: Research Team
Date: 2026
"""

from typing import Dict, List, Tuple
import numpy as np

# Band order must match FREQUENCY_BANDS keys (delta, theta, alpha, beta, gamma)
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']
ALPHA_BAND_IDX = 2
BETA_BAND_IDX = 3

# Region definitions: list of channel names per region
REGION_OCCIPITAL = ['O1', 'O2']
REGION_OCCIPITAL_TEMPORAL = ['O1', 'O2', 'T5', 'T6']
REGION_FRONTAL = ['F3', 'F4', 'Fz']
PAIR_O1_O2 = ('O1', 'O2')
PAIR_F3_F4 = ('F3', 'F4')


def get_channel_indices(channel_names: List[str], ch_list: List[str]) -> List[int]:
    """
    Resolve channel names to indices using the pipeline's channel order.

    Args:
        channel_names: Full list of channel names (order = index in coherence matrix).
        ch_list: Channel names to find (e.g. ['O1', 'O2']).

    Returns:
        List of indices for ch_list. Raises ValueError if any name not found.
    """
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    indices = []
    for ch in ch_list:
        if ch not in name_to_idx:
            raise ValueError(f"Channel '{ch}' not in channel_names: {channel_names}")
        indices.append(name_to_idx[ch])
    return indices


def _region_mean_coherence(
    coh: np.ndarray,
    indices: List[int],
) -> np.ndarray:
    """
    Mean coherence over all pairs (i,j) with i, j in indices.
    coh: (n_subjects, n_bands, n_channels, n_channels).
    Returns: (n_subjects, n_bands).
    """
    n_sub, n_bands, n_ch, _ = coh.shape
    out = np.zeros((n_sub, n_bands), dtype=coh.dtype)
    for i, ii in enumerate(indices):
        for j, jj in enumerate(indices):
            out += coh[:, :, ii, jj]
    n_pairs = len(indices) * len(indices)
    out /= n_pairs
    return out


def _pair_coherence(coh: np.ndarray, i: int, j: int) -> np.ndarray:
    """Single pair coherence. Returns (n_subjects, n_bands)."""
    return coh[:, :, i, j].copy()


def _global_mean_off_diagonal(coh: np.ndarray) -> np.ndarray:
    """Mean over all off-diagonal elements per subject per band. (n_subjects, n_bands)."""
    n_sub, n_bands, n_ch, _ = coh.shape
    out = np.zeros((n_sub, n_bands), dtype=coh.dtype)
    for s in range(n_sub):
        for b in range(n_bands):
            m = coh[s, b, :, :]
            mask = ~np.eye(n_ch, dtype=bool)
            out[s, b] = np.mean(m[mask])
    return out


def compute_regional_coherence(
    coherence: np.ndarray,
    channel_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute global and regional mean coherence from a coherence matrix.

    Args:
        coherence: Shape (n_subjects, n_bands, n_channels, n_channels).
        channel_names: Channel names in matrix order (length n_channels).

    Returns:
        Dict with keys:
        - 'global': (n_subjects, n_bands) - mean off-diagonal coherence per band.
        - 'alpha_occipital': (n_subjects, n_bands) - mean over O1, O2 pairs.
        - 'alpha_occipital_temporal': (n_subjects, n_bands) - mean over O1, O2, T5, T6.
        - 'beta_frontal': (n_subjects, n_bands) - mean over F3, F4, Fz.
        - 'O1_O2': (n_subjects, n_bands) - single pair O1-O2.
        - 'F3_F4': (n_subjects, n_bands) - single pair F3-F4.
        All values (n_subjects, n_bands).
    """
    if coherence.ndim != 4:
        raise ValueError(f"coherence must be 4D (n_subjects, n_bands, n_channels, n_channels), got {coherence.ndim}D")
    n_sub, n_bands, n_ch, _ = coherence.shape
    if len(channel_names) != n_ch:
        raise ValueError(f"channel_names length {len(channel_names)} != n_channels {n_ch}")

    result = {}

    # Global mean (off-diagonal) - existing behaviour for transparency
    result['global'] = _global_mean_off_diagonal(coherence)

    # Alpha regions
    occ_idx = get_channel_indices(channel_names, REGION_OCCIPITAL)
    result['alpha_occipital'] = _region_mean_coherence(coherence, occ_idx)

    occ_temp_idx = get_channel_indices(channel_names, REGION_OCCIPITAL_TEMPORAL)
    result['alpha_occipital_temporal'] = _region_mean_coherence(coherence, occ_temp_idx)

    i_o1, i_o2 = get_channel_indices(channel_names, [PAIR_O1_O2[0], PAIR_O1_O2[1]])
    result['O1_O2'] = _pair_coherence(coherence, i_o1, i_o2)

    # Beta regions
    frontal_idx = get_channel_indices(channel_names, REGION_FRONTAL)
    result['beta_frontal'] = _region_mean_coherence(coherence, frontal_idx)

    i_f3, i_f4 = get_channel_indices(channel_names, [PAIR_F3_F4[0], PAIR_F3_F4[1]])
    result['F3_F4'] = _pair_coherence(coherence, i_f3, i_f4)

    return result


def summarize_regional_for_report(
    ad_coherence: np.ndarray,
    hc_coherence: np.ndarray,
    channel_names: List[str],
) -> Dict:
    """
    Compute regional coherence for AD and HC and summarize for the validation report.

    Args:
        ad_coherence: (n_subjects_ad, n_bands, n_channels, n_channels).
        hc_coherence: (n_subjects_hc, n_bands, n_channels, n_channels).
        channel_names: Channel names in matrix order.

    Returns:
        Dict with AD/HC group means, differences (HC - AD), and ratios (AD/HC)
        for global, alpha (occipital, occipital_temporal, O1_O2), beta (frontal, F3_F4).
    """
    ad_reg = compute_regional_coherence(ad_coherence, channel_names)
    hc_reg = compute_regional_coherence(hc_coherence, channel_names)

    def group_mean(per_subject_band: np.ndarray) -> np.ndarray:
        """(n_subjects, n_bands) -> (n_bands,)"""
        return np.mean(per_subject_band, axis=0)

    report = {
        'channel_names': channel_names,
        'band_names': BAND_NAMES,
        'regions': {},
    }

    for key in ['global', 'alpha_occipital', 'alpha_occipital_temporal', 'O1_O2', 'beta_frontal', 'F3_F4']:
        ad_mean = group_mean(ad_reg[key])
        hc_mean = group_mean(hc_reg[key])
        diff = hc_mean - ad_mean
        ratio = np.divide(ad_mean, hc_mean, out=np.full_like(ad_mean, np.nan), where=hc_mean != 0)
        report['regions'][key] = {
            'AD_mean_per_band': [round(float(x), 6) for x in ad_mean],
            'HC_mean_per_band': [round(float(x), 6) for x in hc_mean],
            'difference_HC_minus_AD_per_band': [round(float(x), 6) for x in diff],
            'ratio_AD_over_HC_per_band': [round(float(x), 6) for x in ratio],
        }
        report['regions'][key]['AD_mean_alpha'] = round(float(ad_mean[ALPHA_BAND_IDX]), 6)
        report['regions'][key]['HC_mean_alpha'] = round(float(hc_mean[ALPHA_BAND_IDX]), 6)
        report['regions'][key]['AD_mean_beta'] = round(float(ad_mean[BETA_BAND_IDX]), 6)
        report['regions'][key]['HC_mean_beta'] = round(float(hc_mean[BETA_BAND_IDX]), 6)

    # Add focused alpha/beta summary for report section
    def _diff_alpha(reg_name):
        return round(float(group_mean(hc_reg[reg_name])[ALPHA_BAND_IDX] - group_mean(ad_reg[reg_name])[ALPHA_BAND_IDX]), 6)
    def _diff_beta(reg_name):
        return round(float(group_mean(hc_reg[reg_name])[BETA_BAND_IDX] - group_mean(ad_reg[reg_name])[BETA_BAND_IDX]), 6)

    report['alpha_summary'] = {
        'global': {'AD': report['regions']['global']['AD_mean_alpha'], 'HC': report['regions']['global']['HC_mean_alpha'],
                   'difference_HC_minus_AD': _diff_alpha('global')},
        'occipital': {'AD': report['regions']['alpha_occipital']['AD_mean_alpha'], 'HC': report['regions']['alpha_occipital']['HC_mean_alpha'],
                     'difference_HC_minus_AD': _diff_alpha('alpha_occipital')},
        'occipital_temporal': {'AD': report['regions']['alpha_occipital_temporal']['AD_mean_alpha'], 'HC': report['regions']['alpha_occipital_temporal']['HC_mean_alpha'],
                              'difference_HC_minus_AD': _diff_alpha('alpha_occipital_temporal')},
        'O1_O2': {'AD': report['regions']['O1_O2']['AD_mean_alpha'], 'HC': report['regions']['O1_O2']['HC_mean_alpha'],
                  'difference_HC_minus_AD': _diff_alpha('O1_O2')},
    }
    report['beta_summary'] = {
        'global': {'AD': report['regions']['global']['AD_mean_beta'], 'HC': report['regions']['global']['HC_mean_beta'],
                   'difference_HC_minus_AD': _diff_beta('global')},
        'frontal': {'AD': report['regions']['beta_frontal']['AD_mean_beta'], 'HC': report['regions']['beta_frontal']['HC_mean_beta'],
                    'difference_HC_minus_AD': _diff_beta('beta_frontal')},
        'F3_F4': {'AD': report['regions']['F3_F4']['AD_mean_beta'], 'HC': report['regions']['F3_F4']['HC_mean_beta'],
                  'difference_HC_minus_AD': _diff_beta('F3_F4')},
    }

    return report
