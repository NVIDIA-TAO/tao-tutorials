"""Config generation and checkpoint resolution utilities for DEFT pipelines."""


def create_clip_train_config(
    base_config_yaml: str,
    new_config_yaml: str,
    output_dir: str,
    checkpoint_path: str,
    sweep_args: str,
    mined_image_dir: str,
    mined_caption_dir: str,
    mined_image_list_file: str,
    caption_file_suffix: str,
    train_image_dir: str = "",
    train_caption_dir: str = "",
    train_image_list_file: str = "",
    train_pairs_file: str = "",
    mined_pairs_file: str = "",
    val_image_list_file: str = "",
    val_image_dir: str = "",
    val_caption_dir: str = "",
    config_overrides: str = "",
) -> str:
    """Create a TAO CLIP training config YAML by patching a source config.

    Args:
        base_config_yaml:       Path to the source YAML config.
        new_config_yaml:        Path where the patched YAML will be written.
        output_dir:             TAO ``results_dir`` root.
        checkpoint_path:        Prior-iteration TAO checkpoint.
        sweep_args:             JSON string of optimizer overrides.
        mined_image_dir:        Mined dataset's ``image_dir``.
        mined_caption_dir:      Mined dataset's ``caption_dir``.
        mined_image_list_file:  Mined dataset's ``image_list_file``.
        caption_file_suffix:    Caption file suffix for the mined entry.
        train_image_list_file:  Optional replacement for seed image list.
        train_pairs_file:       Optional replacement for seed pairs file.
        mined_pairs_file:       Optional mined ``train_pairs_file``.
        val_image_list_file:    Explicit image list for ``dataset.val``.
        val_image_dir:          ``image_dir`` for the val dataset.
        val_caption_dir:        ``caption_dir`` for the val dataset.
        config_overrides:       JSON string of deep-merge overrides.

    Returns:
        Path to the newly written config YAML.
    """
    import json
    import os

    import yaml

    sweep_args = json.loads(sweep_args)

    def _apply_config_overrides(config, raw_overrides):
        if not raw_overrides:
            return
        overrides = json.loads(raw_overrides)
        if not isinstance(overrides, dict):
            return

        def _merge(dst, src):
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    _merge(dst[key], value)
                else:
                    dst[key] = value

        _merge(config, overrides)

    with open(base_config_yaml, "r") as f:
        config_data = yaml.safe_load(f)

    config_data["results_dir"] = output_dir
    _apply_config_overrides(config_data, config_overrides)
    for section_name in ("evaluate", "inference", "export"):
        section = config_data.get(section_name)
        if not isinstance(section, dict):
            continue
        if section.get("checkpoint") in ("???", ""):
            section["checkpoint"] = None
        if section_name == "export" and section.get("onnx_file") in ("???", ""):
            section["onnx_file"] = None

    train_cfg = config_data.setdefault("train", {})
    dataset_cfg = config_data.setdefault("dataset", {})
    train_data_cfg = dataset_cfg.setdefault("train", {})
    datasets = []

    if train_image_list_file:
        seed_entry = {
            "image_list_file": train_image_list_file,
            "caption_file_suffix": ".txt",
        }
        if train_image_dir:
            seed_entry["image_dir"] = train_image_dir
        if train_caption_dir:
            seed_entry["caption_dir"] = train_caption_dir
        if train_pairs_file:
            seed_entry["train_pairs_file"] = train_pairs_file
        datasets.append(seed_entry)

    if mined_image_list_file:
        mined_entry = {
            "image_dir": mined_image_dir,
            "caption_dir": mined_caption_dir,
            "image_list_file": mined_image_list_file,
            "caption_file_suffix": caption_file_suffix,
        }
        if mined_pairs_file:
            mined_entry["train_pairs_file"] = mined_pairs_file
        datasets.append(mined_entry)

    train_data_cfg["datasets"] = datasets

    if val_image_list_file:
        val_data_cfg = dataset_cfg.setdefault("val", {})
        val_datasets = list(val_data_cfg.get("datasets") or [])
        if not val_datasets:
            val_datasets.append({})
        effective_val_image_dir = val_image_dir or train_image_dir
        effective_val_caption_dir = val_caption_dir or train_caption_dir
        if effective_val_image_dir:
            val_datasets[0]["image_dir"] = effective_val_image_dir
        if effective_val_caption_dir:
            val_datasets[0]["caption_dir"] = effective_val_caption_dir
        val_datasets[0]["image_list_file"] = val_image_list_file
        val_data_cfg["datasets"] = val_datasets

    val_data_cfg = dataset_cfg.setdefault("val", {})
    val_datasets = list(val_data_cfg.get("datasets") or [])
    if not val_datasets:
        try:
            val_batch_size = max(int(val_data_cfg.get("batch_size", 1)), 1)
        except (TypeError, ValueError):
            val_batch_size = 1
        try:
            num_gpus = max(int(train_cfg.get("num_gpus", 1)), 1)
        except (TypeError, ValueError):
            num_gpus = 1
        gpu_ids = train_cfg.get("gpu_ids") or []
        if isinstance(gpu_ids, list) and gpu_ids:
            num_gpus = max(num_gpus, len(gpu_ids))
        try:
            num_nodes = max(int(train_cfg.get("num_nodes", 1)), 1)
        except (TypeError, ValueError):
            num_nodes = 1
        min_auto_val_samples = val_batch_size * num_gpus * num_nodes
        try:
            requested_auto_val_samples = int(
                os.environ.get("CLIP_AUTO_VAL_SAMPLES", "128")
            )
        except ValueError:
            requested_auto_val_samples = 128
        max_auto_val_samples = max(
            requested_auto_val_samples,
            min_auto_val_samples,
        )

        auto_val_source = mined_image_list_file or train_image_list_file
        auto_val_image_dir = mined_image_dir if mined_image_list_file else train_image_dir
        auto_val_caption_dir = mined_caption_dir if mined_image_list_file else train_caption_dir
        auto_val_rows = []
        if auto_val_source and auto_val_image_dir and auto_val_caption_dir:
            seen = set()
            with open(auto_val_source, "r", encoding="utf-8") as f:
                for raw_line in f:
                    image_name = raw_line.strip()
                    if not image_name or image_name in seen:
                        continue
                    auto_val_rows.append(image_name)
                    seen.add(image_name)
                    if len(auto_val_rows) >= max_auto_val_samples:
                        break

        if auto_val_rows:
            config_dir = os.path.dirname(new_config_yaml)
            auto_val_list = os.path.join(config_dir, "auto_val_image_list.txt")
            os.makedirs(config_dir, exist_ok=True)
            with open(auto_val_list, "w", encoding="utf-8") as f:
                f.write("\n".join(auto_val_rows) + "\n")
            val_datasets = [{
                "image_dir": auto_val_image_dir,
                "caption_dir": auto_val_caption_dir,
                "image_list_file": auto_val_list,
                "caption_file_suffix": caption_file_suffix,
            }]
            val_data_cfg["datasets"] = val_datasets

        validation_interval = 1
        train_cfg["validation_interval"] = validation_interval
        train_cfg["val_check_interval"] = None
        if auto_val_rows:
            print(
                "Validation datasets were empty; created "
                f"{len(auto_val_rows)}-sample auto validation list at "
                f"{auto_val_list} and set train.validation_interval="
                f"{validation_interval} (minimum validation batch size: "
                f"{min_auto_val_samples})."
            )
        else:
            print(
                "Validation datasets are empty and no auto validation list could "
                "be created; TAO training may require val.datasets."
            )

    train_cfg.pop("checkpointer", None)

    if checkpoint_path:
        train_cfg["pretrained_model_path"] = checkpoint_path

    if "lr" in sweep_args:
        train_cfg.setdefault("optim", {})["lr"] = sweep_args["lr"]

    config_dir = os.path.dirname(new_config_yaml)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    with open(new_config_yaml, "w") as f:
        yaml.safe_dump(
            config_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    n_datasets = len(config_data["dataset"]["train"].get("datasets") or [])
    print(
        f"Created CLIP train config at {new_config_yaml} "
        f"({n_datasets} dataset.train.datasets entries)"
    )
    return new_config_yaml

def create_clip_eval_config(
    base_config_yaml: str,
    new_config_yaml: str,
    output_dir: str,
    checkpoint_path: str,
    eval_image_dir: str = "",
    eval_caption_dir: str = "",
    eval_image_list_file: str = "",
    caption_file_suffix: str = ".txt",
    config_overrides: str = "",
) -> str:
    """Create a TAO CLIP evaluation config YAML by patching a template.

    Args:
        base_config_yaml: Path to the template YAML config file.
        new_config_yaml:  Path where the patched YAML will be written.
        output_dir:       TAO ``results_dir`` root.
        checkpoint_path:  Checkpoint for ``evaluate.checkpoint``.
        eval_image_dir:   Optional evaluation image root.
        eval_caption_dir: Optional evaluation caption root.
        eval_image_list_file: Optional evaluation image list.
        caption_file_suffix: Caption suffix for the evaluation dataset.
        config_overrides: JSON string of deep-merge overrides.

    Returns:
        Path to the newly written config YAML.
    """
    import json
    import os

    import yaml

    def _apply_config_overrides(config, raw_overrides):
        if not raw_overrides:
            return
        overrides = json.loads(raw_overrides)
        if not isinstance(overrides, dict):
            return

        def _merge(dst, src):
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    _merge(dst[key], value)
                else:
                    dst[key] = value

        _merge(config, overrides)

    with open(base_config_yaml, "r") as f:
        config_data = yaml.safe_load(f)

    config_data["results_dir"] = output_dir
    _apply_config_overrides(config_data, config_overrides)
    for section_name in ("evaluate", "inference", "export"):
        section = config_data.get(section_name)
        if not isinstance(section, dict):
            continue
        if section.get("checkpoint") in ("???", ""):
            section["checkpoint"] = None
        if section_name == "export" and section.get("onnx_file") in ("???", ""):
            section["onnx_file"] = None

    eval_cfg = config_data.setdefault("evaluate", {})
    eval_cfg["checkpoint"] = checkpoint_path or None
    eval_cfg["results_dir"] = os.path.join(output_dir, "evaluate")
    if eval_image_list_file:
        eval_cfg["datasets"] = [{
            "image_dir": eval_image_dir,
            "caption_dir": eval_caption_dir,
            "image_list_file": eval_image_list_file,
            "caption_file_suffix": caption_file_suffix,
        }]
        fallback_dataset = dict(eval_cfg["datasets"][0])
        fallback_dataset["train_pairs_file"] = None
        dataset_cfg = config_data.setdefault("dataset", {})
        dataset_cfg.setdefault("train", {})["datasets"] = [dict(fallback_dataset)]
        dataset_cfg.setdefault("val", {})["datasets"] = [dict(fallback_dataset)]

    config_dir = os.path.dirname(new_config_yaml)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    with open(new_config_yaml, "w") as f:
        yaml.safe_dump(
            config_data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Created CLIP eval config at {new_config_yaml}")
    return new_config_yaml

def normalize_clip_pretrained_checkpoint(
    checkpoint_path: str,
    output_path: str = "",
) -> str:
    """Create a model-only CLIP checkpoint for ``train.pretrained_model_path``.

    Strips the ``model.`` LightningModule prefix from state-dict keys so the
    checkpoint can be loaded directly by the SigLIP model.
    """
    import os

    if not checkpoint_path:
        return checkpoint_path

    source_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {source_path}")

    lower_path = source_path.lower()
    if lower_path.endswith(".safetensors"):
        return source_path
    if not (lower_path.endswith(".pth") or lower_path.endswith(".ckpt")):
        return source_path

    if output_path:
        normalized_path = os.path.abspath(output_path)
    else:
        root, ext = os.path.splitext(source_path)
        normalized_path = f"{root}_pretrained{ext or '.pth'}"

    if (
        os.path.isfile(normalized_path)
        and os.path.getmtime(normalized_path) >= os.path.getmtime(source_path)
    ):
        print(f"Using normalized CLIP pretrained checkpoint: {normalized_path}")
        return normalized_path

    import torch

    try:
        checkpoint = torch.load(
            source_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(source_path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        print(
            "Checkpoint is not a state-dict mapping; using original "
            f"checkpoint for pretrained_model_path: {source_path}"
        )
        return source_path

    prefixed_items = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if isinstance(key, str) and key.startswith("model.")
    }
    if not prefixed_items:
        print(f"CLIP checkpoint is already model-loadable: {source_path}")
        return source_path

    os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
    tmp_path = f"{normalized_path}.tmp"
    torch.save(prefixed_items, tmp_path)
    os.replace(tmp_path, normalized_path)
    print(
        "Normalized CLIP pretrained checkpoint: "
        f"{source_path} -> {normalized_path} ({len(prefixed_items)} tensors)"
    )
    return normalized_path


def resolve_prev_clip_train_config(
    base_experiment_path: str,
    iter_num: int,
    continual_dataset: bool,
    base_template: str,
    train_image_list_file: str,
) -> str:
    """Return the source train config for the current CLIP DEFT iteration."""
    import os

    if not continual_dataset:
        return base_template

    if iter_num <= 1:
        sft_cfg = os.path.join(
            base_experiment_path, "sft", "specs", "train_config.yaml",
        )
        if train_image_list_file and os.path.isfile(sft_cfg):
            return sft_cfg
        return base_template

    prev_cfg = os.path.join(
        base_experiment_path, f"iter_{iter_num - 1}",
        "specs", "train_config.yaml",
    )
    if not os.path.isfile(prev_cfg):
        raise FileNotFoundError(
            f"continual_dataset requested but no prior CLIP train config "
            f"found at {prev_cfg}"
        )
    return prev_cfg



def resolve_prev_eval_dir(
    base_experiment_path: str, iter_num: int,
    train_ann_path: str,
    eval_subdir: str = "evaluate",
) -> str:
    """Return the results directory from the previous step."""
    import os

    if iter_num <= 1:
        zs_dir = os.path.join(base_experiment_path, "zs", eval_subdir)
        sft_dir = os.path.join(base_experiment_path, "sft", eval_subdir)
        if train_ann_path and os.path.isdir(sft_dir):
            return sft_dir
        print(
            "Initial SFT eval output not found or not requested; "
            f"using zero-shot eval dir: {zs_dir}"
        )
        return zs_dir
    return os.path.join(
        base_experiment_path, f"iter_{iter_num - 1}",
        eval_subdir,
    )
