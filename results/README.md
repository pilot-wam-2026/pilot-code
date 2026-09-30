# Recorded RoboCasa-GR1 Results

Original 340000 checkpoint: **717/1200 = 59.75%**. This is the repaired-protocol
source evaluation, not the manuscript's separate 58.3% experiment.

[Logs and provenance](../docs/LOGS.md) | [Evaluation protocol](../docs/EVALUATION.md) |
[JSON summary](robocasa_340000.json) | [Training metrics](training_metrics_340000.csv)

```bash
python3 -m release.audit_logs
```

| Task | Successes / Episodes | Success |
|---|---:|---:|
| Cup-DrawerClose | 13/50 | 26.0% |
| Potato-MicrowaveClose | 28/50 | 56.0% |
| Milk-MicrowaveClose | 25/50 | 50.0% |
| Bottle-CabinetClose | 32/50 | 64.0% |
| Wine-CabinetClose | 14/50 | 28.0% |
| Can-DrawerClose | 31/50 | 62.0% |
| Cuttingboard-Basket | 39/50 | 78.0% |
| Cuttingboard-Cardboardbox | 23/50 | 46.0% |
| Cuttingboard-Pan | 45/50 | 90.0% |
| Cuttingboard-Pot | 46/50 | 92.0% |
| Cuttingboard-Tieredbasket | 22/50 | 44.0% |
| Placemat-Basket | 42/50 | 84.0% |
| Placemat-Bowl | 25/50 | 50.0% |
| Placemat-Plate | 27/50 | 54.0% |
| Placemat-Tieredshelf | 15/50 | 30.0% |
| Plate-Bowl | 26/50 | 52.0% |
| Plate-Cardboardbox | 22/50 | 44.0% |
| Plate-Pan | 19/50 | 38.0% |
| Plate-Plate | 40/50 | 80.0% |
| Tray-Cardboardbox | 41/50 | 82.0% |
| Tray-Plate | 40/50 | 80.0% |
| Tray-Pot | 42/50 | 84.0% |
| Tray-Tieredbasket | 38/50 | 76.0% |
| Tray-Tieredshelf | 22/50 | 44.0% |
| **Total** | **717/1200** | **59.75%** |

All tasks use seeds 9000-9049. The public archive has all 24 simulation logs
and task result files, 1200 episode diagnostics, and 680 training metric rows.
See the log guide for redactions and the raw action streams not included here.
