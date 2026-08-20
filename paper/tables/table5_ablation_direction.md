# Table 5. Matched unconstrained vs constrained direction agreement

**Caption (place ABOVE the table, Elsevier):** Locked unconstrained battery versus constrained direction agreement against the constrained training signature (locked fold-0 seed-42 checkpoints). Unconstrained r values are from that locked battery, not from the αc-sweep fold-0 re-simulation. Prior-only signed effects also reach 10/10 on signs with lower mean effect-vector correlation (r=0.636).

| Dataset | n | Constrained agree | Constrained r | Unconstrained agree | Unconstrained r |
|---|---:|---|---:|---|---:|
| TUH | 200 | 10/10 | 0.908 | 3/10 | -0.309 |
| OSF | 92 | 10/10 | 0.851 | 4/10 | -0.331 |
| P-ADIC | 145 | 10/10 | 0.918 | 4/10 | -0.327 |

Prior-only vs constrained training signature: 10/10; mean effect-vector r = 0.636 (Donepezil r = 0.572; Memantine r = 0.700).

Sources: `unconstrained_external_battery.json`, `complete_validation_report_v3.json`, `prior_only_direction.json`.
