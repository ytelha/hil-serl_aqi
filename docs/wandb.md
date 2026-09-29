# Weights & Biases monitoring

What every metric means, how to read a run, how to compare arms and how to
export the data: [`wandb_metrics.md`](wandb_metrics.md).

W&B is integrated into the learner processes. The actor sends episode and timer
statistics to the learner through Agentlace; the learner combines those with its
optimization metrics and writes one W&B run. The actor does not open a separate
W&B run.

## Connect an account

Create an account at <https://wandb.ai>, copy the API key from
<https://wandb.ai/authorize>, then authenticate once from the repository:

```bash
./.venv/bin/wandb login
```

For an unattended machine, set the key in the shell that launches training:

```bash
export WANDB_API_KEY="<your-api-key>"
```

Do not commit the key. By default runs use the `hil-serl` project in the
authenticated user's account. These standard W&B variables customize routing:

```bash
export WANDB_PROJECT="hil-serl"
export WANDB_ENTITY="your-user-or-team"
export WANDB_RUN_GROUP="campaign-name"
```

## Enable logging

For a VS Code learner task, choose **on - log to W&B** at the W&B prompt. Start
the learner first and the actor after the learner prints `LEARNER READY`.

For a campaign, put this in its YAML:

```yaml
logging:
  wandb: "online"  # off | offline | online
```

or override it on the command line:

```bash
./.venv/bin/python examples/run_campaign.py \
  --config examples/campaigns/study.yaml \
  --campaign runs/study1 --wandb online
```

`online` streams live and requires authentication. `offline` requires no key
and stores each run under the cell's `checkpoints/<cell>/wandb/` directory.
`off` disables W&B; the repository's AQI JSONL and campaign artifacts are still
written.

To upload an offline run later:

```bash
./.venv/bin/wandb sync runs/study1/checkpoints/<cell>/wandb/wandb/offline-run-*
```

## What is recorded

The run configuration includes the parsed training flags, host, mode, task,
seed, operator, ensemble, trigger, quality, replay, and budget settings. Logged
time series include:

- actor/critic/temperature losses and other agent update metrics;
- episode return, length, success information, intervention counts, and AQI
  request counts sent by the actor;
- learner and actor timing metrics;
- demo-pretraining progress when enabled;
- quality-aware replay priority statistics in `quality` and `aqi` modes.

W&B intentionally receives summaries rather than every image and transition.
Detailed acquisition scores, intervention segments, quality scores, and
per-episode records remain in `examples/aqi_logs/<run_name>/*.jsonl`; campaign
scores and analysis remain under `runs/<campaign>/`.

## Dashboard and export

In the W&B project, use the run name and config columns to group by method,
operator, or seed. Useful charts include episode return/success, intervention
count, AQI requests, critic/actor losses, replay priority statistics, and timer
throughput.

Download a run's scalar history with the public API:

```python
import wandb

api = wandb.Api()
run = api.run("your-entity/hil-serl/run-id")
history = run.history(pandas=True)
history.to_csv("wandb_history.csv", index=False)
print(dict(run.config))
print(dict(run.summary))
```

For complete history instead of the sampled `history()` result, iterate
`run.scan_history()` and write those rows to CSV or JSON.