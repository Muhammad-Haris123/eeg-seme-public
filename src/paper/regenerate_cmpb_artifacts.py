"""
One-command CMPB / SEME artifact regeneration (Path A Week 2).

Regenerates non-destructive paper artifacts from frozen validation JSON / OOF
stores. Does NOT overwrite locked CVAE checkpoints, does NOT reprocess TUH,
does NOT retrain the twin.

Usage (from repo root):
  .\\eeg_twin\\Scripts\\python.exe src\\paper\\regenerate_cmpb_artifacts.py
  .\\eeg_twin\\Scripts\\python.exe src\\paper\\regenerate_cmpb_artifacts.py --pack-zip
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / "eeg_twin" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)


def run(module_or_script: list[str], label: str) -> None:
    cmd = [str(PY), *module_or_script]
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT)
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {label}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate CMPB/SEME paper artifacts")
    ap.add_argument("--pack-zip", action="store_true", help="Also build Overleaf zip")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--skip-delong", action="store_true")
    args = ap.parse_args()

    steps = []
    if not args.skip_delong:
        steps.append((["-m", "src.validation.run_delong_oof_probes"], "DeLong on stored OOF"))

    # Tables / figures that read frozen JSON only
    if not args.skip_figures:
        steps.extend(
            [
                (["src/tables/make_table2_direction_agreement.py"], "Table 2/3 direction"),
                (["src/tables/make_table3_diagnostic_results.py"], "Table 4 diagnostic"),
                (["src/tables/make_table4_encoding_results.py"], "Table 5 encoding"),
                (["src/tables/make_table5_ablation_direction.py"], "Table 2 ablation"),
                (["src/figures/make_fig5_encoding_analysis.py"], "Fig. 5 encoding bake-off"),
                (["src/figures/make_fig6_constraint_strength.py"], "Fig. 6 alpha_c"),
                (["src/figures/make_fig_bakeoff_encoding.py"], "Fig. bake-off panel"),
            ]
        )

    for args_list, label in steps:
        run(args_list, label)

    if args.pack_zip:
        run(["src/paper/pack_overleaf_zip.py"], "Pack Overleaf zip")

    print("\n[ok] regenerate_cmpb_artifacts finished")
    print("Locked checkpoints were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
