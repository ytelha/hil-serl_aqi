import datetime
import os
import tempfile
from copy import copy
from socket import gethostname

import absl.flags as flags
import ml_collections
import wandb


def _resolve_wandb_mode(debug: bool) -> str:
    if debug:
        return "disabled"
    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    if mode == "dryrun":
        mode = "offline"
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(
            "WANDB_MODE must be online, offline, or disabled; "
            f"got {mode!r}."
        )
    return mode


def _as_tags(tag):
    """Tags for wandb.init: None, or a list of strings.

    wandb.init(tags=...) iterates its argument, so a plain string would turn
    into one tag per character. Accept a string, a sequence of strings or None.
    """
    if tag is None:
        return None
    if isinstance(tag, str):
        return [tag] if tag else None
    return [str(t) for t in tag]


WANDB_TAG_MAX = 64   # W&B rejects longer tags


def _env_tags(var: str = "AQI_WANDB_TAGS"):
    """Extra tags from the environment, comma-separated. run_campaign.py sets
    AQI_WANDB_TAGS for each cell's learner (campaign, arm, tier, operator,
    seed, ...), so the tags need no command-line flag."""
    return [t.strip() for t in os.environ.get(var, "").split(",") if t.strip()]


def _merge_tags(*groups):
    """Tag lists joined in order, duplicates dropped, each tag cut to the W&B
    maximum; None when nothing is left."""
    out = []
    for group in groups:
        for t in group or []:
            t = str(t)[:WANDB_TAG_MAX]
            if t and t not in out:
                out.append(t)
    return out or None


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "serl_launcher"  # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        # Which entity to log as (default: your own user)
        config.exp_descriptor = ""  # Run name (doesn't have to be unique)
        # Unique identifier for run (will be automatically generated unless
        # provided)
        config.unique_identifier = ""
        config.group = None
        return config

    def __init__(
        self,
        wandb_config,
        variant,
        wandb_output_dir=None,
        debug=False,
    ):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"  # NOQA

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = os.environ.get("WANDB_DIR") or tempfile.mkdtemp()
        os.makedirs(wandb_output_dir, exist_ok=True)

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        mode = _resolve_wandb_mode(debug)
        project = os.environ.get("WANDB_PROJECT", self.config.project)
        entity = os.environ.get("WANDB_ENTITY") or self.config.entity
        group = os.environ.get("WANDB_RUN_GROUP") or self.config.group

        self.run = wandb.init(
            config=self._variant,
            project=project,
            entity=entity,
            group=group,
            tags=_merge_tags(_as_tags(getattr(self.config, "tag", None)),
                             _env_tags()),
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            name=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )

        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
        self.run.config.update(flag_dict)

    def log(self, data: dict, step: int = None):
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        self.run.log(data, step=step)

    def finish(self):
        self.run.finish()
