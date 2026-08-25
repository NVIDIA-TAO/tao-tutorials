"""Parsed configuration for a PAS CLIP DEFT experiment."""

import dataclasses
import json
import os

import yaml
from omegaconf import MISSING, OmegaConf

from pas_deft.config_fields import BOOL_FIELD, DATACLASS_FIELD, FLOAT_FIELD, INT_FIELD, STR_FIELD


def bool_str(value) -> str:
    """Convert a config value to the 'true'/'false' string some pipeline functions expect."""
    return "true" if str(value).strip().lower() in ("true", "1", "yes", "y") else "false"


def _abs_data_path(value: str) -> str:
    """Absolutize a PAS data path, leaving blanks and absolute paths untouched.

    These values are handed to both host-side code (which reads the files) and
    the TAO container specs (which read them again through a bind mount), so a
    single string has to be valid on both sides. The notebook mounts the data
    root at its own host path to make that true, which only holds if the path
    is absolute — a relative ``data/...`` resolves against the container's
    working directory (``/opt/nvidia``) and silently finds nothing.
    """
    value = str(value or "")
    return os.path.abspath(value) if value else value


# ── Dataclass schema — mirrors deft_config.yaml's sections ─────────────────────────────────────

@dataclasses.dataclass
class ExperimentSection:
    """``experiment:`` — top-level experiment identity and TAO spec locations."""

    name: str = STR_FIELD(value=MISSING, description="Name of this PAS DEFT experiment.")
    results_path: str = STR_FIELD(
        value=MISSING,
        description=(
            "Root output directory, relative to the notebook's working directory "
            "(it is bind-mounted into the TAO containers)."
        ),
    )
    train_config: str = STR_FIELD(value=MISSING, description="Path to the TAO CLIP train spec YAML.")
    eval_config: str = STR_FIELD(value=MISSING, description="Path to the TAO CLIP eval spec YAML.")
    visualize: bool = BOOL_FIELD(
        value=False,
        description="Fallback for visualization.enabled when no visualization: block is given.",
    )
    visualize_embeddings: bool = BOOL_FIELD(
        value=False,
        description="Fallback for visualization.embeddings when no visualization: block is given.",
    )
    tao_pytorch_root: str = STR_FIELD(
        value="",
        description="Root of the tao-pytorch checkout; inferred from train_config/eval_config if blank.",
    )


@dataclasses.dataclass
class VisualizationSection:
    """``visualization:`` — contact-sheet and t-SNE embedding visualization."""

    enabled: bool = BOOL_FIELD(value=False, description="Enable contact-sheet visualization of weak/mined samples.")
    embeddings: bool = BOOL_FIELD(value=False, description="Enable t-SNE embedding visualization.")
    max_samples_per_group: int = INT_FIELD(
        value=12, valid_min=1, description="Max contact-sheet samples per (dataset, query_type) group.",
    )
    max_total_samples: int = INT_FIELD(value=96, valid_min=1, description="Max total contact-sheet samples.")
    tile_size: int = INT_FIELD(value=192, valid_min=1, description="Contact-sheet tile size in pixels.")


@dataclasses.dataclass
class IterationSection:
    """``iteration:`` — the DEFT loop's iteration range."""

    start: int = INT_FIELD(value=MISSING, valid_min=1, description="First DEFT iteration to run (1-based).")
    end: int = INT_FIELD(value=MISSING, valid_min=1, description="Last DEFT iteration to run (inclusive).")


@dataclasses.dataclass
class TrainingSection:
    """``training:`` — checkpoint carry-over policy between iterations."""

    init_checkpoint: str = STR_FIELD(
        value="", description="Checkpoint TAO CLIP training starts from at iteration 1.",
    )
    continual_model: bool = BOOL_FIELD(
        value=False,
        description="Carry the trained model forward across iterations instead of resetting to init_checkpoint.",
    )
    continual_dataset: bool = BOOL_FIELD(
        value=True,
        description="Accumulate each iteration's dataset into training instead of replacing it.",
    )


@dataclasses.dataclass
class HistoryAwareSection:
    """``mining.history_aware:`` — cross-iteration selection ledger."""

    enabled: bool = BOOL_FIELD(
        value=False,
        description="Track a cross-iteration selection ledger so mined pairs are never re-selected.",
    )
    replay_fraction: float = FLOAT_FIELD(
        value=0.20, valid_min=0.0, valid_max=1.0,
        description="Fraction of the target budget spent replaying previously-selected pairs (continual_dataset=false only).",
    )


@dataclasses.dataclass
class CaptionExpansionSection:
    """``mining.recovery.caption_expansion:`` — expand mined anchors to their other captions."""

    enabled: bool = BOOL_FIELD(
        value=False, description="Expand each mined anchor image to its other captions in the source pairs file.",
    )
    mode: str = STR_FIELD(
        value="nearest", valid_options="nearest,all",
        description="'nearest' ranks expansion captions by similarity to the anchor; 'all' takes them in source order.",
    )
    max_pairs_per_image_path: int = INT_FIELD(
        value=2, valid_min=0, description="Max pairs kept per expanded image path (0 = unlimited).",
    )
    max_expanded_pair_fraction: float = FLOAT_FIELD(
        value=0.25, valid_min=0.0, valid_max=1.0,
        description="Max fraction of the target query budget spent on expanded (non-anchor) pairs.",
    )
    dedupe_normalized_caption: bool = BOOL_FIELD(
        value=True,
        description="Drop expansion rows whose normalized caption duplicates one already kept for the same image path.",
    )
    count_expanded_pairs_toward_target: str = STR_FIELD(
        value="auto", valid_options="auto,true,false",
        description="Whether expanded pairs count against target_query_count ('auto' = mode != 'all').",
    )


@dataclasses.dataclass
class MiningSection:
    """``mining:`` — k-NN mining parameters (topn/knn_metric come from mining_spec.yaml)."""

    knn_batch_size: int = INT_FIELD(value=32, valid_min=1, description="Batch size used by the k-NN mining step.")
    topn: int = INT_FIELD(
        value=MISSING, valid_min=1, description="Nearest neighbors mined per target query (from mining_spec.yaml).",
    )
    knn_metric: str = STR_FIELD(value=MISSING, description="Distance metric for k-NN mining (from mining_spec.yaml).")
    history_aware: HistoryAwareSection = DATACLASS_FIELD(
        HistoryAwareSection(), description="Cross-iteration selection ledger settings.",
    )
    caption_expansion: CaptionExpansionSection = DATACLASS_FIELD(
        CaptionExpansionSection(), description="mining.recovery.caption_expansion settings.",
    )


@dataclasses.dataclass
class PasSection:
    """``pas:`` — PAS split materialization and dataset roots."""

    seed_exclude_datasets: str = STR_FIELD(
        value="CUHK_PEDES,ICFG_PEDES", description="Comma-separated real datasets excluded from the seed training split.",
    )
    augmented_suffix: str = STR_FIELD(value="_Aug", description="Dataset-name suffix identifying augmented rows.")
    query_types: str = STR_FIELD(
        value="easy,medium,hard,natural_caption,original_captions",
        description="Comma-separated query types kept when materializing PAS splits.",
    )
    max_seed_rows: int = INT_FIELD(value=0, valid_min=0, description="Cap on seed training rows (0 = no cap).")
    max_aug_pool_rows: int = INT_FIELD(value=0, valid_min=0, description="Cap on augmented mining-pool rows (0 = no cap).")
    mining_pool_mode: str = STR_FIELD(
        value="real_and_augmented", valid_options="real,augmented,real_and_augmented",
        description="Which rows populate the mining pool.",
    )
    val_sample_size: int = INT_FIELD(
        value=512, valid_min=0, description="Number of eval images sampled into the TAO validation list.",
    )
    train_pairs_source_file: str = STR_FIELD(
        value="", description="Source train_pairs.json for the seed training split. Optional.",
    )
    pool_pairs_source_file: str = STR_FIELD(
        value="", description="Source pairs JSON for the mining pool; falls back to train_pairs_source_file when blank.",
    )
    eval_pairs_source_file: str = STR_FIELD(value=MISSING, description="Source test_pairs.json for the PAS eval split.")
    train_image_dir: str = STR_FIELD(value=MISSING, description="Image root for the seed training split.")
    train_caption_dir: str = STR_FIELD(value=MISSING, description="Caption root for the seed training split.")
    source_image_dir: str = STR_FIELD(value=MISSING, description="Image root for the mining pool / mined outputs.")
    source_caption_dir: str = STR_FIELD(value=MISSING, description="Caption root for the mining pool / mined outputs.")
    eval_image_dir: str = STR_FIELD(value=MISSING, description="Image root for the PAS eval split.")
    eval_caption_dir: str = STR_FIELD(value=MISSING, description="Caption root for the PAS eval split.")


@dataclasses.dataclass
class CaptionDiversitySection:
    """``gap_analysis.caption_diversity:`` — coverage-aware weak-caption rotation."""

    enabled: bool = BOOL_FIELD(
        value=False,
        description="Rotate weak-query caption selection to maximize coverage across iterations.",
    )
    history_file: str = STR_FIELD(
        value="caption_selection_history.json",
        description="Filename (under base_experiment_path) tracking previously-selected captions.",
    )
    history_policy: str = STR_FIELD(
        value="auto", valid_options="auto,prefer_unseen,novelty_with_replay",
        description="'auto' resolves to prefer_unseen when continual_dataset else novelty_with_replay.",
    )
    coverage_target: float = FLOAT_FIELD(
        value=1.0, valid_min=0.0,
        description="Fraction of each weak group's unseen captions to plan for across the remaining iterations.",
    )
    min_unique_texts_per_attribute: int = INT_FIELD(
        value=0, valid_min=0, description="Minimum unique captions selected per weak attribute, when available.",
    )
    max_unique_texts_per_attribute: int = INT_FIELD(
        value=0, valid_min=0, description="Cap on unique captions selected per weak attribute (0 = no cap).",
    )
    max_rows_per_unique_text: int = INT_FIELD(
        value=1, valid_min=1, description="Max rows kept for the same normalized caption.",
    )
    max_rows_per_image_path: int = INT_FIELD(
        value=1, valid_min=0, description="Max rows kept for the same image path (0 = no cap).",
    )
    recent_exclude_iters: int = INT_FIELD(
        value=0, valid_min=0, description="Number of recent iterations whose captions are excluded from reselection.",
    )
    replay_fraction_when_noncontinual: float = FLOAT_FIELD(
        value=0.25, valid_min=0.0, valid_max=1.0,
        description="Fraction of each group's budget spent replaying seen captions under novelty_with_replay.",
    )


@dataclasses.dataclass
class GapAnalysisSection:
    """``gap_analysis:`` — weak-attribute selection driving each iteration's mining target."""

    metric_name: str = STR_FIELD(
        value="Rank-1", description="Metric used to rank weak attributes (e.g. Rank-1, Match@14, Zero@14).",
    )
    queries_per_slice: int = INT_FIELD(value=256, valid_min=0, description="Max captions sampled per weak attribute.")
    min_num_queries: int = INT_FIELD(
        value=1, valid_min=0, description="Ignore metric rows with fewer queries than this.",
    )
    query_types: str = STR_FIELD(
        value="easy,medium", description="Comma-separated query types considered for weak-attribute selection.",
    )
    weak_attribute_topk: int = INT_FIELD(
        value=8, valid_min=0, description="Number of weakest attributes selected each iteration.",
    )
    target_query_count: int = INT_FIELD(value=100000, valid_min=0, description="Final mined-query budget.")
    total_queries_map: int = INT_FIELD(
        value=768, valid_min=0, description="Query budget used when analyze_by_map is enabled.",
    )
    analyze_by_map: bool = BOOL_FIELD(
        value=False, description="Rank weak attributes by mAP instead of the per-query metric breakdown.",
    )
    caption_diversity: CaptionDiversitySection = DATACLASS_FIELD(
        CaptionDiversitySection(), description="Coverage-aware caption rotation settings.",
    )


@dataclasses.dataclass
class DeftExperimentConfig:
    """Top-level PAS CLIP DEFT experiment schema — mirrors deft_config.yaml's sections."""

    experiment: ExperimentSection = DATACLASS_FIELD(ExperimentSection())
    visualization: VisualizationSection = DATACLASS_FIELD(VisualizationSection())
    iteration: IterationSection = DATACLASS_FIELD(IterationSection())
    training: TrainingSection = DATACLASS_FIELD(TrainingSection())
    mining: MiningSection = DATACLASS_FIELD(MiningSection())
    pas: PasSection = DATACLASS_FIELD(PasSection())
    gap_analysis: GapAnalysisSection = DATACLASS_FIELD(GapAnalysisSection())


def _validate_field_constraints(instance, path=""):
    """Recursively check every field's valid_min/valid_max/valid_options metadata."""
    for f in dataclasses.fields(instance):
        value = getattr(instance, f.name)
        field_path = f"{path}.{f.name}" if path else f.name
        if dataclasses.is_dataclass(value):
            _validate_field_constraints(value, field_path)
            continue
        meta = f.metadata
        valid_min = meta.get("valid_min", "")
        valid_max = meta.get("valid_max", "")
        valid_options = meta.get("valid_options", "")
        if valid_min != "" and value < valid_min:
            raise ValueError(f"{field_path}={value!r} is below the minimum allowed value {valid_min}")
        if valid_max != "" and value > valid_max:
            raise ValueError(f"{field_path}={value!r} is above the maximum allowed value {valid_max}")
        if valid_options:
            options = {opt.strip() for opt in valid_options.split(",")}
            if str(value) not in options:
                raise ValueError(f"{field_path}={value!r} must be one of: {valid_options}")


def _abs_or_missing(value):
    """Like _abs_data_path, but passes an absent (MISSING) value through untouched.

    Required path fields are looked up with ``.get(key, MISSING)`` rather than ``[key]`` so a
    field missing from the YAML flows through as OmegaConf's MISSING sentinel and surfaces as a
    clear "missing mandatory value" error at merge time, instead of a bare KeyError here.
    """
    return value if value is MISSING else _abs_data_path(value)


def _build_source_dict(raw: dict, mining_spec: dict) -> dict:
    """Reshape deft_config.yaml + mining_spec.yaml into DeftExperimentConfig's exact schema."""
    exp = raw.get("experiment") or {}
    train_config = exp.get("train_config", MISSING)
    eval_config = exp.get("eval_config", MISSING)

    tao_pytorch_root = exp.get("tao_pytorch_root", "")
    if not tao_pytorch_root:
        marker = "/nvidia_tao_pytorch/"
        for path in (train_config, eval_config):
            if path is not MISSING and marker in path:
                tao_pytorch_root = path.split(marker, 1)[0]
                break

    viz = raw.get("visualization") or {}
    visualization = {
        "enabled": bool(viz.get("enabled", exp.get("visualize", False))),
        "embeddings": bool(viz.get("embeddings", exp.get("visualize_embeddings", False))),
    }
    for key in ("max_samples_per_group", "max_total_samples", "tile_size"):
        if key in viz:
            visualization[key] = viz[key]

    mining = raw.get("mining") or {}
    caption_expansion = (mining.get("recovery") or {}).get("caption_expansion") or {}
    mining_out = {k: v for k, v in mining.items() if k not in ("history_aware", "recovery")}
    if "history_aware" in mining:
        mining_out["history_aware"] = dict(mining["history_aware"])
    if caption_expansion:
        caption_expansion = dict(caption_expansion)
        if "count_expanded_pairs_toward_target" in caption_expansion:
            caption_expansion["count_expanded_pairs_toward_target"] = str(
                caption_expansion["count_expanded_pairs_toward_target"]
            ).lower()
        mining_out["caption_expansion"] = caption_expansion
    for key in ("topn", "knn_metric"):
        if not isinstance(mining_spec, dict) or key not in mining_spec:
            raise ValueError(f"mining_spec.yaml is missing required key {key!r}")
    mining_out["topn"] = mining_spec["topn"]
    mining_out["knn_metric"] = mining_spec["knn_metric"]

    pas = raw.get("pas") or {}
    train_pairs_source_file = _abs_data_path(pas.get("train_pairs_source_file", ""))
    pool_pairs_source_file = (
        _abs_data_path(pas.get("pool_pairs_source_file", "")) or train_pairs_source_file
    )
    path_keys = (
        "train_pairs_source_file", "pool_pairs_source_file", "eval_pairs_source_file",
        "train_image_dir", "train_caption_dir", "source_image_dir",
        "source_caption_dir", "eval_image_dir", "eval_caption_dir",
    )
    pas_out = {k: v for k, v in pas.items() if k not in path_keys}
    pas_out.update({
        "train_pairs_source_file": train_pairs_source_file,
        "pool_pairs_source_file": pool_pairs_source_file,
        "eval_pairs_source_file": _abs_or_missing(pas.get("eval_pairs_source_file", MISSING)),
        "train_image_dir": _abs_or_missing(pas.get("train_image_dir", MISSING)),
        "train_caption_dir": _abs_or_missing(pas.get("train_caption_dir", MISSING)),
        "source_image_dir": _abs_or_missing(pas.get("source_image_dir", MISSING)),
        "source_caption_dir": _abs_or_missing(pas.get("source_caption_dir", MISSING)),
        "eval_image_dir": _abs_or_missing(pas.get("eval_image_dir", MISSING)),
        "eval_caption_dir": _abs_or_missing(pas.get("eval_caption_dir", MISSING)),
    })

    gap = raw.get("gap_analysis") or {}
    gap_out = {
        k: v for k, v in gap.items()
        if k not in ("caption_diversity", "total_queries_mAP", "analyze_by_mAP")
    }
    gap_out["caption_diversity"] = dict(gap.get("caption_diversity") or {})
    if "total_queries_mAP" in gap:
        gap_out["total_queries_map"] = gap["total_queries_mAP"]
    if "analyze_by_mAP" in gap:
        gap_out["analyze_by_map"] = gap["analyze_by_mAP"]

    return {
        "experiment": {
            "name": exp.get("name", MISSING),
            "results_path": exp.get("results_path", MISSING),
            "train_config": train_config,
            "eval_config": eval_config,
            "visualize": bool(exp.get("visualize", False)),
            "visualize_embeddings": bool(exp.get("visualize_embeddings", False)),
            "tao_pytorch_root": tao_pytorch_root,
        },
        "visualization": visualization,
        "iteration": dict(raw.get("iteration") or {}),
        "training": dict(raw.get("training") or {}),
        "mining": mining_out,
        "pas": pas_out,
        "gap_analysis": gap_out,
    }


class PasDeftConfig:
    """All parsed, validated parameters for a PAS CLIP DEFT experiment.

    Loads a YAML spec file (plus the sibling ``mining_spec.yaml``) once, merges it onto the
    :class:`DeftExperimentConfig` schema via OmegaConf (catching missing required fields, wrong
    types, and unknown keys), then checks every field's ``valid_min``/``valid_max``/
    ``valid_options`` constraint.

    Usage::

        cfg = PasDeftConfig("configs/deft_config.yaml")
        cfg.pas.eval_image_dir
        cfg.mining.history_aware.enabled
    """

    CLIP_CKPT_RELPATH = "best/clip_best_val_t2i_mAP.pth"
    # Model-only copy of the above with the LightningModule "model." prefix
    # stripped, written by normalize_clip_pretrained_checkpoint. This is what
    # carries into the next iteration's train.pretrained_model_path; the raw
    # checkpoint above is what eval consumes.
    CLIP_PRETRAINED_RELPATH = "pretrained/model_state.pth"

    def __init__(self, config_path: str):
        self.config_path = config_path

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        mining_spec_path = os.path.join(os.path.dirname(config_path), "mining_spec.yaml")
        with open(mining_spec_path) as f:
            mining_spec = yaml.safe_load(f)

        source = _build_source_dict(raw, mining_spec)
        schema = OmegaConf.structured(DeftExperimentConfig)
        merged = OmegaConf.merge(schema, OmegaConf.create(source))
        self.cfg: DeftExperimentConfig = OmegaConf.to_object(merged)
        _validate_field_constraints(self.cfg)

        self.experiment = self.cfg.experiment
        self.visualization = self.cfg.visualization
        self.iteration = self.cfg.iteration
        self.training = self.cfg.training
        self.mining = self.cfg.mining
        self.pas = self.cfg.pas
        self.gap_analysis = self.cfg.gap_analysis

        self._validate_business_rules()

        self.sweep_args_str: str = json.dumps({"config": config_path})

    def _validate_business_rules(self):
        """Cross-field checks the per-field valid_min/valid_max/valid_options can't express."""
        if os.path.isabs(self.base_experiment_path):
            raise ValueError(
                f"experiment.results_path must be relative to the notebook's "
                f"working directory, got {self.base_experiment_path!r}. The "
                f"container reads it at /{os.path.basename(self.base_experiment_path)}, "
                f"which the notebook mounts as "
                f"-v $PWD/{os.path.basename(self.base_experiment_path)}:"
                f"/{os.path.basename(self.base_experiment_path)}."
            )
        if self.base_experiment_path.split("/")[0] != "results":
            raise ValueError(
                f"experiment.results_path must be 'results' or start with 'results/', "
                f"got {self.experiment.results_path!r}. The notebook always mounts its "
                f"HOST_RESULTS_DIR (host ./results/) at the container path /results, "
                f"so any other top-level name is unreachable inside the container."
            )
        if self.iteration.end < self.iteration.start:
            raise ValueError(
                f"iteration.end ({self.iteration.end}) must be >= "
                f"iteration.start ({self.iteration.start})"
            )

    # ── Derived paths ────────────────────────────────────────────────────

    @property
    def base_experiment_path(self) -> str:
        """Root output directory for this experiment (``experiment.results_path``)."""
        return self.experiment.results_path

    @property
    def pas_splits_dir(self) -> str:
        return f"{self.base_experiment_path}/pas_splits"

    @property
    def history_aware_history_file(self) -> str:
        return f"{self.base_experiment_path}/mining_selection_history.json"

    @property
    def caption_history_file(self) -> str:
        return f"{self.base_experiment_path}/{self.gap_analysis.caption_diversity.history_file}"

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def container_path(host_path: str) -> str:
        """Host-relative path → the path the TAO containers see."""
        path = str(host_path or "")
        if not path or os.path.isabs(path):
            return path
        return f"/{os.path.normpath(path)}"

    def training_checkpoint_for_iter(self, iter_num: int) -> str:
        """Resolve the training checkpoint to use at the start of a DEFT iteration.

        Returns a *container* path (leading ``/``), since the value is written
        into the TAO spec as ``train.pretrained_model_path``. The normalized model-only
        checkpoint is used rather than the raw Lightning one.
        """
        if not self.training.continual_model:
            return self.training.init_checkpoint
        if iter_num == 1:
            host_ckpt = f"{self.base_experiment_path}/sft/{self.CLIP_PRETRAINED_RELPATH}"
            return (
                self.container_path(host_ckpt)
                if (bool(self.pas.train_pairs_source_file) and os.path.exists(host_ckpt))
                else self.training.init_checkpoint
            )
        return self.container_path(
            f"{self.base_experiment_path}/iter_{iter_num - 1}/{self.CLIP_PRETRAINED_RELPATH}"
        )

    def __repr__(self) -> str:
        return (
            f"PasDeftConfig(experiment={self.experiment.name!r}, "
            f"path={self.config_path!r})"
        )
