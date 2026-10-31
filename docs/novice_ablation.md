# Novice test: which part of the quality arms collapses the novice?

**The question.** The first training machine's oracle campaign
([aqi_oracle_quality_results_and_qa_log.md](thesis/aqi_oracle_quality_results_and_qa_log.md))
found this on the old stack:

| arm | SR | ICC | expert / intermediate / novice |
|---|---|---|---|
| ensemble extended | 79.0 % | 0.18 | 94 / 78 / **65** |
| quality-oracle | 73.3 % | 0.63 | 97 / 91 / **32** |
| asks 30 % + oracle | 78.8 % | 0.93 | 100 / 100 / **36** |
| asks 30 % + counterfactual | 73.4 % | 0.97 | 100 / 100 / **10** |

Its log concludes that even a perfect score amplifies operator dependence, so the
problem is the idea of grading corrections, not the score.

**Why that does not follow yet.** The quality arms changed three things at once
compared with ensemble extended:

1. **The oracle score.** It sets the replay priority and the BC weights. It never
   drops a takeover; the campaign's own header says so.
2. **The default hard filters.** The arms do not set `quality_filters`, so
   `safety_veto` and `outcome_filter` are on (`jerk_veto` is a no-op, since
   `jerk_hard` is unset). `outcome_filter` drops every takeover from a failed episode,
   whatever the oracle says. On the second machine, 76–78 % of novice takeovers were in
   failed episodes, against 26 % for experts.
3. **The BC loss.** `bc_loss_weight: 0.1` applies only to quality arms
   (`train_aqi_rlpd.py`: `bc_weight = ... if aqi_cfg.quality_on else 0.0`), so these
   arms also imitate the novice's corrections.

The oracle and the counterfactual arm both collapse the novice, and they share 2 and 3.
So the campaign shows that this *package* collapses the novice, not that the score does.

**A hint in the other direction:** intermediates rose from 78 to 91 % with the oracle.
That is one reason to find out which part does what.

## The test

Novice tier only, the three operators of the oracle campaign, 2 seeds. Everything else
equals `aqi_oracle_quality`; `tests/test_novice_ablation.py` checks that the two
reference arms get identical training commands.

| arm | score | hard filters | BC | cells |
|---|---|---|---|---|
| `logged` (ensemble extended) | – | – | – | 6, copied |
| `quality-oracle` | oracle | default | 0.1 | 6, copied |
| `oracle-nofilter` | oracle | none | 0.1 | 6 new |
| `oracle-nobc` | oracle | none | 0 | 6 new |
| `filter-only` | none (`uniform`) | default | 0 | 6 new |
| `bc-only` | none (`uniform`) | none | 0.1 | 6 new |

`logged` stands in for "no score, no filter, no BC"; it does not use the quality
machinery. Operators, demos and the ruler are copied from the oracle campaign, so each
new cell pairs with its oracle-campaign cells.

Cost: 24 new cells, about 1.5 days on the first machine (its 18-cell probe took about a
day), up to 40 GB. It must run on the code that ran the oracle campaign, **before**
patches 0011–0019, or the copied cells stop being comparable.

```bash
./.venv/bin/python examples/novice_ablation_report.py --count runs/aqi_oracle_quality/aqi_logs
./.venv/bin/python examples/copy_campaign_cells.py --src runs/aqi_oracle_quality --dst runs/quality_novice_ablation --arms logged,quality-oracle --operators nov00,nov01,nov02
```

The launch command is in the header of `examples/campaigns/quality_novice_ablation.yaml`.
The first line needs no training: it shows what each arm of the oracle campaign actually
dropped, per tier and why.

## The reading, fixed before the results

`examples/novice_ablation_report.py --campaign runs/quality_novice_ablation` applies
this. An arm has **collapsed** if its mean novice SR is below the midpoint between
`logged` and `quality-oracle`, and has **recovered** otherwise.

| `filter-only` | `bc-only` | `oracle-nofilter` | `oracle-nobc` | reading |
|---|---|---|---|---|
| collapsed | recovered | recovered | any | **the hard filters are the cause**: this is about filtering by episode outcome |
| recovered | collapsed | collapsed | recovered | **the BC loss is the cause**: this is about imitating novice corrections |
| collapsed | collapsed | any | any | each one collapses the novice alone |
| recovered | recovered | any | collapsed | **the oracle's replay priority is the cause**: this supports the oracle campaign's conclusion |
| recovered | recovered | collapsed | recovered | it takes the oracle score together with the BC loss |
| recovered | recovered | recovered | recovered | no part alone; it takes the hard filters together with the others |
| any other | | | | mixed: read the table |

If `logged` and `quality-oracle` differ by less than 15 points in these cells, there is
nothing to explain and the report says so.

Six cells per arm show direction, not significance; the gap to explain is 34 points. The
report also gives each arm's paired difference against both references (exact sign test),
the SR of the last checkpoint and the share of takeovers and operator steps dropped.

## What each outcome means

- **Filters or BC:** the thesis should say that *outcome filtering* or *imitating novice
  corrections* harms weak operators, not that grading corrections does. Whether a clean
  per-correction filter helps stays open, and the keep-all vs oracle-drop test on the new
  stack becomes worthwhile.
- **The oracle's replay priority:** the oracle campaign's conclusion stands, with a
  mechanism. Drop the quality line; a further test on the new stack is not worth its cost.
- **Next steps either way:** apply patches 0011–0019, then the ladder.

Until this has run, do not write "a paradigm problem" in the thesis.
