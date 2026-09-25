# `campaign_plots.py` — operator-dependency figures

Reads `run_campaign.py`'s analysis output and renders a set of publication-ready
figures.

---

## Purpose

Turns the JSON dependency report produced by `run_campaign.py analyze` into
plots.  Called automatically at the end of a campaign run; also usable
standalone to regenerate or iterate on figures without re-running the full
campaign.

---

## Input

`<campaign>/report/dependency.json` — written by `run_campaign.py` analyze
stage.  This file must exist; nothing else is needed.

---

## Output — figures written to `<campaign>/report/plots/`

| File | What it shows |
|---|---|
| `icc_by_method.png` | headline dependency: ICC of SR and ICC of steps-to-convergence per method (lower = less operator-dependent) |
| `sr_by_operator.png` | per-operator mean SR within each method — scatter + errorbars + per-tier colour; converging dots across methods = AQI works |
| `sr_by_tier.png` | mean SR per skill tier per method (skill gradient: expert > intermediate > novice; AQI should flatten the slope) |
| `variance_decomposition.png` | stacked bar: σ²_between-operator vs σ²_within-operator per method |
| `operator_spread.png` | CV and permutation-p per method (operator spread diagnostic) |
| `steps_to_convergence.png` | box + convergence fraction per method (training effort) |

All figures are saved as both `.png` and `.svg`.

---

## Flags

```
python examples/campaign_plots.py --campaign <dir>
```

| Flag | Description |
|---|---|
| `--campaign <dir>` | campaign directory (must contain `report/dependency.json`) |

---

## Examples

```bash
# regenerate all plots after a campaign has finished
python examples/campaign_plots.py --campaign runs/study1

# update one figure without re-running the analysis
# (edit campaign_plots.py then)
python examples/campaign_plots.py --campaign runs/study1
```

---

## Connections

```
run_campaign.py analyze  ──writes──▶  <campaign>/report/dependency.json
                                       (also calls campaign_plots.make_plots()
                                        directly if --plots is set)

campaign_plots.py  ◀──reads──  <campaign>/report/dependency.json
                  ──writes──▶  <campaign>/report/plots/*.png / *.svg
```

`campaign_plots.py` is **standalone** — it can be run at any time after the
campaign's analyze stage.  It imports `matplotlib` lazily so a missing
install degrades gracefully (prints a warning, no crash).
