"""Inspect P-ADIC MATLAB v7.3 / HDF5 structure."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

ROOT = Path(r"E:\padic_external")

# Documented order from eeg-slowing-transportability (Nihon-style)
PADIC_CHANNELS = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
)


def _decode(handle, ref):
    if not ref:
        return None
    obj = handle[ref]
    arr = np.asarray(obj)
    if arr.dtype == object:
        for child in arr.flat:
            v = _decode(handle, child)
            if v is not None:
                return v
        return None
    if arr.dtype.kind in "ui" and arr.size > 1:
        return "".join(chr(int(v)) for v in arr.flat if int(v) != 0).strip()
    if arr.size == 1:
        v = arr.item()
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode("utf-8", errors="replace")
        return v
    return arr


def _resolve_recording(handle, ref):
    if not ref:
        return None
    obj = handle[ref]
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None and dtype.kind != "O":
        if len(shape) == 2 and 19 in shape and max(shape) > 1000:
            return obj
    arr = np.asarray(obj)
    if arr.dtype == object:
        for child in arr.flat:
            ds = _resolve_recording(handle, child)
            if ds is not None:
                return ds
    return None


def inspect(path: Path) -> None:
    print("=" * 60)
    print(path.name, "size_gb=", round(path.stat().st_size / 1e9, 3))
    with h5py.File(path, "r") as f:
        print("top keys:", list(f.keys()))
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Group):
                print(f"GROUP {k} subkeys={list(obj.keys())}")
            else:
                print(f"DATASET {k} shape={obj.shape} dtype={obj.dtype}")

        # Prefer known group names
        for gname in ("alz_r", "controls_r", "mci_r"):
            if gname not in f:
                continue
            g = f[gname]
            print(f"\n--- group {gname} ---")
            print("fields:", list(g.keys()))
            for field in g.keys():
                ds = g[field]
                print(f"  {field}: shape={getattr(ds, 'shape', None)} dtype={getattr(ds, 'dtype', None)}")

            refs = np.asarray(g["G"])
            print("G shape:", refs.shape)
            n = int(np.prod(refs.shape))
            print("n_cells:", n)

            sfreqs, ages, sexes, shapes, n_samp, ch_axis = [], [], [], [], [], []
            for idx in np.ndindex(refs.shape):
                ds = _resolve_recording(f, refs[idx])
                if ds is None:
                    continue
                shapes.append(tuple(ds.shape))
                if ds.shape[0] == 19:
                    ch_axis.append(0)
                    n_samp.append(ds.shape[1])
                elif ds.shape[1] == 19:
                    ch_axis.append(1)
                    n_samp.append(ds.shape[0])
                else:
                    ch_axis.append(-1)
                    n_samp.append(max(ds.shape))
                if "g" in g:
                    sfreqs.append(float(_decode(f, np.asarray(g["g"])[idx]) or 500.0))
                if "age" in g:
                    ages.append(_decode(f, np.asarray(g["age"])[idx]))
                if "sex" in g:
                    sexes.append(_decode(f, np.asarray(g["sex"])[idx]))

            print("resolved recordings:", len(shapes))
            print("unique shapes (first 10):", sorted(set(shapes))[:10], "n_unique=", len(set(shapes)))
            if n_samp:
                arr = np.array(n_samp)
                print(
                    "n_samples: min/median/max=",
                    int(arr.min()),
                    int(np.median(arr)),
                    int(arr.max()),
                )
                print(
                    "duration_s @500Hz: min/median/max=",
                    round(arr.min() / 500, 1),
                    round(np.median(arr) / 500, 1),
                    round(arr.max() / 500, 1),
                )
            if ch_axis:
                print("channel_axis counts:", {a: ch_axis.count(a) for a in set(ch_axis)})
            if sfreqs:
                print("sfreq unique:", sorted(set(sfreqs))[:20], "n=", len(sfreqs))
            if ages:
                ages_f = []
                for a in ages:
                    try:
                        if a is None:
                            continue
                        v = float(np.asarray(a).reshape(-1)[0])
                        if np.isfinite(v):
                            ages_f.append(v)
                    except Exception:
                        continue
                if ages_f:
                    print(
                        "age: n=",
                        len(ages_f),
                        "mean=",
                        round(float(np.mean(ages_f)), 1),
                        "range=",
                        (min(ages_f), max(ages_f)),
                    )
            if sexes:
                print("sex sample:", [str(s) for s in sexes[:5]], "n=", len(sexes))

            # Peek first recording stats
            first = None
            for idx in np.ndindex(refs.shape):
                first = _resolve_recording(f, refs[idx])
                if first is not None:
                    break
            if first is not None:
                if first.shape[1] == 19:
                    x = np.asarray(first[:5000, :], dtype=np.float64).T
                else:
                    x = np.asarray(first[:, :5000], dtype=np.float64)
                print("peek shape (ch,t):", x.shape)
                print("peek mean/std per ch (first 5):", np.round(x.mean(1)[:5], 3), np.round(x.std(1)[:5], 3))


def main():
    for name in ("alz_c1_new.mat", "controls_c1_new.mat"):
        inspect(ROOT / name)
    print("\nPADIC_CHANNELS (assumed order):", PADIC_CHANNELS)
    from src.utils.config import STANDARD_10_20_CHANNELS

    print("TRAIN_CHANNELS:", tuple(STANDARD_10_20_CHANNELS))
    same = set(PADIC_CHANNELS) == set(STANDARD_10_20_CHANNELS)
    print("same_names:", same)
    print("order_match:", list(PADIC_CHANNELS) == list(STANDARD_10_20_CHANNELS))
    if same:
        remap = [PADIC_CHANNELS.index(ch) for ch in STANDARD_10_20_CHANNELS]
        print("remap_padic_index_to_train:", remap)


if __name__ == "__main__":
    main()
