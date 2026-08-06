"""Parsed configuration for a PAS CLIP DEFT experiment."""

import json
import os
import yaml


def _bool_str(value) -> str:
    """Convert a config value to the 'true'/'false' string expected by pipeline functions."""
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


class PasDeftConfig:
    """All parsed parameters for a PAS CLIP DEFT experiment.

    Loads a YAML spec file once and exposes every pipeline parameter as an
    attribute, so callers never touch raw dicts directly.

    Usage::

        cfg = PasDeftConfig("configs/clip_config.yaml")
        # override a sweep hyperparameter:
        cfg = PasDeftConfig("configs/clip_config.yaml", cosmos_sft_lr=1e-5)
    """

    CLIP_CKPT_RELPATH = "best/clip_best_val_t2i_mAP.pth"
    # Model-only copy of the above with the LightningModule "model." prefix
    # stripped, written by normalize_clip_pretrained_checkpoint. This is what
    # carries into the next iteration's train.pretrained_model_path; the raw
    # checkpoint above is what eval consumes.
    CLIP_PRETRAINED_RELPATH = "pretrained/model_state.pth"

    def __init__(
        self,
        config_path: str,
    ):
        self.config_path = config_path

        with open(config_path) as f:
            _cfg = yaml.safe_load(f)

        self.sweep_args_str: str = json.dumps({
            "config": config_path,
        })

        # ── Experiment ─────────────────────────────────────────────────────
        _exp = _cfg["experiment"]
        self.experiment_name: str = _exp["name"]
        self.base_experiment_path: str = _exp["results_path"]
        self.train_config: str = _exp["train_config"]
        self.eval_config: str = _exp["eval_config"]

        if os.path.isabs(self.base_experiment_path):
            raise ValueError(
                f"experiment.results_path must be relative to the notebook's "
                f"working directory, got {self.base_experiment_path!r}. The "
                f"container reads it at /{os.path.basename(self.base_experiment_path)}, "
                f"which the notebook mounts as "
                f"-v $PWD/{os.path.basename(self.base_experiment_path)}:"
                f"/{os.path.basename(self.base_experiment_path)}."
            )

        self.tao_pytorch_root: str = _exp.get("tao_pytorch_root", "")
        if not self.tao_pytorch_root:
            for _p in (self.train_config, self.eval_config):
                _marker = "/nvidia_tao_pytorch/"
                if _marker in _p:
                    self.tao_pytorch_root = _p.split(_marker, 1)[0]
                    break

        # ── Visualization (contact sheets / t-SNE) ─────────────────────────
        _viz = _cfg.get("visualization", {}) or {}
        self.visualize: bool = bool(
            _viz.get("enabled", _exp.get("visualize", False))
        )
        self.visualize_embeddings: bool = bool(
            _viz.get("embeddings", _exp.get("visualize_embeddings", False))
        )
        self.viz_max_samples_per_group: int = int(
            _viz.get("max_samples_per_group", 12) or 12
        )
        self.viz_max_total_samples: int = int(
            _viz.get("max_total_samples", 96) or 96
        )
        self.viz_tile_size: int = int(_viz.get("tile_size", 192) or 192)

        # ── Training ───────────────────────────────────────────────────────
        _train = _cfg["training"]
        self.init_checkpoint: str = _train["init_checkpoint"]
        self.continual_model: bool = bool(_train["continual_model"])
        self.continual_dataset: bool = bool(_train["continual_dataset"])

        # ── Mining ─────────────────────────────────────────────────────────
        _mining = _cfg["mining"]
        self.mining_topn: int = int(_mining.get("topn", 5) or 5)
        self.knn_metric: str = _mining.get("knn_metric", "cosine")

        _history_aware = _mining.get("history_aware", {}) or {}
        self.history_aware_enabled: str = _bool_str(_history_aware.get("enabled", False))
        self.history_aware_history_file: str = (
            f"{self.base_experiment_path}/mining_selection_history.json"
        )
        self.history_aware_replay_fraction: float = float(
            _history_aware.get("replay_fraction", 0.20) or 0.0
        )

        _cap_exp = (_mining.get("recovery") or {}).get("caption_expansion") or {}
        self.caption_expansion_enabled: str = _bool_str(_cap_exp.get("enabled", False))
        self.caption_expansion_mode: str = _cap_exp.get("mode", "nearest")
        self.caption_expansion_max_pairs_per_image_path: int = int(
            _cap_exp.get("max_pairs_per_image_path", 2) or 0
        )
        self.caption_expansion_max_expanded_pair_fraction: float = float(
            _cap_exp.get("max_expanded_pair_fraction", 0.25) or 0.0
        )
        self.caption_expansion_dedupe_normalized_caption: str = _bool_str(
            _cap_exp.get("dedupe_normalized_caption", True)
        )
        self.caption_expansion_count_expanded_pairs_toward_target: str = str(
            _cap_exp.get("count_expanded_pairs_toward_target", "auto")
        ).lower()

        # ── PAS ────────────────────────────────────────────────────────────
        _pas = _cfg["pas"]
        self.pas_splits_dir: str = f"{self.base_experiment_path}/pas_splits"
        self.pas_seed_exclude_datasets: str = _pas.get(
            "seed_exclude_datasets", "CUHK_PEDES,ICFG_PEDES"
        )
        self.pas_augmented_suffix: str = _pas.get("augmented_suffix", "_Aug")
        self.pas_query_types: str = _pas.get(
            "query_types", "easy,medium,hard,natural_caption,original_captions"
        )
        self.pas_max_seed_rows: int = int(_pas.get("max_seed_rows", 0) or 0)
        self.pas_max_aug_pool_rows: int = int(_pas.get("max_aug_pool_rows", 0) or 0)
        self.pas_mining_pool_mode: str = _pas.get("mining_pool_mode", "real_and_augmented")
        self.pas_val_sample_size: int = int(_pas.get("val_sample_size", 512) or 512)
        self.pas_train_pairs_source_file: str = _abs_data_path(
            _pas.get("train_pairs_source_file", "")
        )
        self.pas_pool_pairs_source_file: str = (
            _abs_data_path(_pas.get("pool_pairs_source_file", ""))
            or self.pas_train_pairs_source_file
        )
        self.pas_eval_pairs_source_file: str = _abs_data_path(
            _pas["eval_pairs_source_file"]
        )
        self.pas_train_image_dir: str = _abs_data_path(_pas["train_image_dir"])
        self.pas_train_caption_dir: str = _abs_data_path(_pas["train_caption_dir"])
        self.pas_source_image_dir: str = _abs_data_path(_pas["source_image_dir"])
        self.pas_source_caption_dir: str = _abs_data_path(_pas["source_caption_dir"])
        self.pas_eval_image_dir: str = _abs_data_path(_pas["eval_image_dir"])
        self.pas_eval_caption_dir: str = _abs_data_path(_pas["eval_caption_dir"])

        # ── Gap analysis ───────────────────────────────────────────────────
        _gap = _cfg.get("gap_analysis", {}) or {}
        self.gap_metric_name: str = _gap.get("metric_name", "Rank-1")
        self.queries_per_slice: int = int(_gap.get("queries_per_slice", 256) or 0)
        self.min_gap_num_queries: int = int(_gap.get("min_num_queries", 1) or 0)
        self.gap_query_types: str = _gap.get("query_types", "easy,medium")
        self.weak_attribute_topk: int = int(_gap.get("weak_attribute_topk", 8) or 0)
        self.target_query_count: int = int(_gap.get("target_query_count", 100000) or 0)
        self.gap_total_queries_map: int = int(_gap.get("total_queries_mAP", 768) or 768)
        self.analyze_by_map: bool = bool(_gap.get("analyze_by_mAP", False))

        _cap_div = _gap.get("caption_diversity", {}) or {}
        self.caption_diversity_enabled: str = _bool_str(_cap_div.get("enabled", False))
        self.caption_history_file: str = (
            f"{self.base_experiment_path}/"
            f"{_cap_div.get('history_file', 'caption_selection_history.json')}"
        )
        self.caption_history_policy: str = _cap_div.get("history_policy", "auto")
        self.caption_coverage_target: float = float(
            _cap_div.get("coverage_target", 1.0) or 0.0
        )
        self.min_unique_texts_per_attribute: int = int(
            _cap_div.get("min_unique_texts_per_attribute", 0) or 0
        )
        self.max_unique_texts_per_attribute: int = int(
            _cap_div.get("max_unique_texts_per_attribute", 0) or 0
        )
        self.max_rows_per_unique_text: int = int(
            _cap_div.get("max_rows_per_unique_text", 1) or 1
        )
        self.max_rows_per_image_path: int = int(
            _cap_div.get("max_rows_per_image_path", 1) or 1
        )
        self.recent_exclude_iters: int = int(
            _cap_div.get("recent_exclude_iters", 0) or 0
        )
        self.replay_fraction_when_noncontinual: float = float(
            _cap_div.get("replay_fraction_when_noncontinual", 0.25) or 0.0
        )

        # ── Misc ───────────────────────────────────────────────────────────
        self.iter_start: int = _cfg["iteration"]["start"]
        self.iter_end: int = _cfg["iteration"]["end"]

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
        if not self.continual_model:
            return self.init_checkpoint
        if iter_num == 1:
            host_ckpt = (
                f"{self.base_experiment_path}/sft/{self.CLIP_PRETRAINED_RELPATH}"
            )
            return (
                self.container_path(host_ckpt)
                if (bool(self.pas_train_pairs_source_file)
                    and os.path.exists(host_ckpt))
                else self.init_checkpoint
            )
        return self.container_path(
            f"{self.base_experiment_path}/iter_{iter_num - 1}"
            f"/{self.CLIP_PRETRAINED_RELPATH}"
        )

    def __repr__(self) -> str:
        return (
            f"PasDeftConfig(experiment={self.experiment_name!r}, "
            f"path={self.config_path!r})"
        )
