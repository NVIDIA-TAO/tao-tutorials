"""Gap analysis for DEFT pipelines.

- :func:`analyze_clip_inference_gaps` — TAO CLIP PAS gap analysis.
- :func:`summarize_pas_eval_metrics` — canonical mAP/Rank-1/Rank-5 for a PAS eval run.
"""

def analyze_clip_inference_gaps(
    results_dir: str,
    gaps_parquet: str,
    kpi_image_dir: str,
    logs_dir: str,
    kpi_pairs_file: str = "",
    metric_name: str = "Rank-1",
    queries_per_slice: int = 200,
    min_num_queries: int = 1,
    query_types: str = "",
    weak_attribute_topk: int = 8,
    target_query_count: int = 100000,
    caption_diversity_enabled: str = "false",
    caption_history_file: str = "",
    iter_num: int = 1,
    total_iters: int = 1,
    continual_dataset: str = "true",
    caption_history_policy: str = "auto",
    caption_coverage_target: float = 1.0,
    min_unique_texts_per_attribute: int = 0,
    max_unique_texts_per_attribute: int = 0,
    max_rows_per_unique_text: int = 1,
    max_rows_per_image_path: int = 1,
    recent_exclude_iters: int = 0,
    replay_fraction_when_noncontinual: float = 0.25,
) -> str:
    """Emit weak PAS text queries from TAO CLIP PAS evaluation metrics.

    Args:
        results_dir:         Directory containing the prior iteration's
                             TAO PAS evaluation output.
        gaps_parquet:        Output path for the weak-sample parquet.
        kpi_image_dir:       Root containing the KPI images.
        logs_dir:            Directory where the stub writes its summary.
        kpi_pairs_file:      TAO-FT ``test_pairs.json``.
        metric_name:         Metric used to choose weak attributes.
        queries_per_slice:   Max captions sampled per weak attribute.
        min_num_queries:     Ignore metric rows with fewer queries.
        query_types:         Optional comma-separated query type filter.
        target_query_count:  Final mined-query budget.
        caption_diversity_enabled: Enable coverage-aware caption rotation.
        caption_history_file: JSON file tracking selected captions.
        iter_num:            Current DEFT iteration number.
        total_iters:         Planned total DEFT iterations.
        continual_dataset:   Whether previous mined datasets remain in training.

    Returns:
        Path to ``gaps_parquet``.
    """
    import csv
    import json
    import math
    import os

    import pandas as pd

    from pas_deft.pairs_io import infer_dataset, iter_json_records, split_csv

    import re
    from collections import Counter, defaultdict

    color_aliases = {
        "black": "black", "white": "white", "gray": "gray", "grey": "gray",
        "red": "red", "maroon": "red", "blue": "blue", "navy": "blue",
        "cyan": "blue", "teal": "blue", "green": "green", "olive": "green",
        "yellow": "yellow", "brown": "brown", "purple": "purple",
        "pink": "pink", "orange": "orange", "beige": "beige/tan",
        "tan": "beige/tan", "khaki": "beige/tan", "blond": "blonde",
        "blonde": "blonde",
    }
    color_order = [
        "black", "white", "gray", "red", "blue", "green", "yellow",
        "brown", "purple", "pink", "orange", "beige/tan", "blonde",
    ]
    clothing_aliases = {
        "shirt": "shirt", "shirts": "shirt", "tshirt": "t-shirt",
        "tee": "t-shirt", "top": "top", "jacket": "jacket",
        "coat": "coat", "hoodie": "hoodie", "sweater": "sweater",
        "blouse": "blouse", "tank": "tank top", "vest": "vest",
        "suit": "suit", "jersey": "jersey", "uniform": "uniform",
        "dress": "dress", "polo": "polo", "cardigan": "cardigan",
        "pants": "pants", "trousers": "pants", "jeans": "jeans",
        "shorts": "shorts", "skirt": "skirt", "leggings": "leggings",
        "slacks": "pants", "shoe": "shoes", "shoes": "shoes",
        "sneaker": "sneakers", "sneakers": "sneakers", "boot": "boots",
        "boots": "boots", "sandal": "sandals", "sandals": "sandals",
        "footwear": "shoes", "bag": "bag", "backpack": "backpack",
        "purse": "purse", "handbag": "handbag", "hat": "hat",
        "cap": "hat", "helmet": "helmet", "mask": "mask",
        "scarf": "scarf", "tie": "tie", "belt": "belt",
        "glasses": "glasses", "sunglasses": "glasses",
    }
    concept_tokens = {
        "Top clothing": {
            "shirt", "shirts", "tshirt", "tee", "top", "jacket", "coat",
            "hoodie", "sweater", "blouse", "tank", "vest", "suit",
            "jersey", "uniform", "dress", "polo", "cardigan",
        },
        "Bottom clothing": {
            "pants", "trousers", "jeans", "shorts", "skirt", "leggings",
            "slacks",
        },
        "Shoe terms": {
            "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
            "sandal", "sandals", "footwear",
        },
        "Accessories": {
            "bag", "backpack", "purse", "handbag", "glasses",
            "sunglasses", "hat", "cap", "helmet", "mask", "umbrella",
            "phone", "watch", "headphones", "headset", "scarf", "tie",
            "belt", "bracelet", "wristband", "necklace",
        },
        "Hair/head": {"hair", "bald", "ponytail", "head"},
        "Body/view/action": {
            "person", "man", "woman", "male", "female", "boy", "girl",
            "child", "walking", "standing", "sitting", "front", "back",
            "side", "profile", "facing", "tall", "short", "thin", "slim",
            "heavy",
        },
        "Pattern/logo": {
            "stripe", "stripes", "striped", "logo", "pattern",
            "patterned", "plain", "floral", "printed", "print",
            "checkered", "plaid",
        },
    }
    concept_order = [
        "Color terms", "Top clothing", "Bottom clothing", "Shoe terms",
        "Accessories", "Hair/head", "Body/view/action", "Pattern/logo",
    ]

    def _truthy(value):
        return str(value).strip().lower() in ("true", "1", "yes", "y", "on")

    def _tokens(text):
        cleaned = (
            str(text or "").lower()
            .replace("t-shirt", "tshirt")
            .replace("t shirt", "tshirt")
        )
        return re.findall(r"[a-z0-9]+", cleaned)

    def _normalize_caption_text(text):
        return " ".join(_tokens(text))

    def _colors(text):
        found = {
            color_aliases[token]
            for token in set(_tokens(text))
            if token in color_aliases
        }
        return [color for color in color_order if color in found]

    def _color_item_pairs(text):
        toks = _tokens(text)
        pairs = set()
        for idx, token in enumerate(toks):
            if token == "top" and idx > 0 and toks[idx - 1] == "tank":
                continue
            item = clothing_aliases.get(token)
            if not item:
                continue
            start = max(0, idx - 3)
            stop = min(len(toks), idx + 2)
            nearby = {
                color_aliases[toks[pos]]
                for pos in range(start, stop)
                if pos != idx and toks[pos] in color_aliases
            }
            for color in nearby:
                pairs.add(f"{color} {item}")
        return sorted(pairs)

    def _attributes(text):
        toks = set(_tokens(text))
        attrs = []
        if _colors(text):
            attrs.append("Color terms")
        for name in concept_order:
            if name == "Color terms":
                continue
            if toks & concept_tokens[name]:
                attrs.append(name)
        return attrs

    metrics_path = os.path.join(results_dir, "nvidia_pas_metrics.csv")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Could not find nvidia_pas_metrics.csv at {metrics_path}")
    if not kpi_pairs_file:
        raise ValueError("PAS CLIP gap analysis requires kpi_pairs_file.")
    if not os.path.isfile(kpi_pairs_file):
        raise FileNotFoundError(f"kpi_pairs_file not found: {kpi_pairs_file}")

    selection_qtype_filter = split_csv(query_types)
    selection_qtype_label = ",".join(sorted(selection_qtype_filter)) if selection_qtype_filter else "all"
    metric_name = str(metric_name or "Rank-1")
    high_is_weak = metric_name.lower().startswith("zero@")
    weak_attribute_topk = int(weak_attribute_topk or 0)
    queries_per_slice = int(queries_per_slice or 0)
    min_num_queries = int(min_num_queries or 0)
    target_query_count = int(target_query_count or 0)

    metric_rows = []
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = str(row.get("Dataset") or "").strip()
            qtype = str(row.get("QueryType") or "").strip()
            if not dataset or dataset.startswith(("AVG_", "WAVG_")):
                continue
            if qtype == "image_to_image" or (selection_qtype_filter and qtype not in selection_qtype_filter):
                continue
            try:
                n_queries = int(float(row.get("num_queries") or 0))
            except ValueError:
                n_queries = 0
            if n_queries < min_num_queries:
                continue
            if metric_name not in row or row.get(metric_name) in (None, ""):
                continue
            try:
                metric_value = float(row[metric_name])
            except (TypeError, ValueError):
                continue
            metric_rows.append({
                "dataset": dataset,
                "query_type": qtype,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "num_queries": n_queries,
                "selection_basis": "metrics_csv_dataset_query_type",
            })
    if not metric_rows:
        raise ValueError(f"No usable PAS metric rows found in {metrics_path}")

    attribute_rows = []

    selection_pair_rows = []
    selection_qtype_counts = Counter()
    skipped_pairs = 0
    for row in iter_json_records(kpi_pairs_file):
        qtype = str(row.get("query_type") or "").strip()
        use_for_selection = (
            not selection_qtype_filter or qtype in selection_qtype_filter
        )
        if not use_for_selection:
            skipped_pairs += 1
            continue
        caption = str(row.get("caption") or "").strip()
        unique_name = str(row.get("unique_name") or "").strip()
        image_path = str(row.get("image_path") or "")
        dataset = str(row.get("dataset") or "").strip() or infer_dataset(image_path)
        if not caption or not unique_name or not dataset:
            skipped_pairs += 1
            continue
        out = dict(row)
        out.update({
            "dataset": dataset,
            "query_type": qtype,
            "caption": caption,
            "unique_name": unique_name,
        })
        selection_pair_rows.append(out)
        selection_qtype_counts[qtype or "(blank)"] += 1

    if not selection_pair_rows:
        raise ValueError(
            "No KPI pair rows matched gap_analysis.query_types="
            f"{selection_qtype_label!r}; cannot select weak attributes."
        )

    metric_by_slice = {
        (row["dataset"], row["query_type"]): row
        for row in metric_rows
    }
    proxy_sums = defaultdict(lambda: {
        "n": 0,
        "metric_sum": 0.0,
        "datasets": Counter(),
        "query_types": Counter(),
    })
    for row in selection_pair_rows:
        metric_row = metric_by_slice.get(
            (row["dataset"], row["query_type"])
        )
        if not metric_row:
            continue
        attrs = _attributes(row.get("caption", ""))
        if not attrs:
            continue
        for attr in attrs:
            sums = proxy_sums[attr]
            sums["n"] += 1
            sums["metric_sum"] += float(metric_row["metric_value"])
            sums["datasets"][row["dataset"]] += 1
            sums["query_types"][row["query_type"]] += 1
    for attr, sums in proxy_sums.items():
        if int(sums["n"]) < min_num_queries:
            continue
        attribute_rows.append({
            "dataset": "ALL",
            "query_type": selection_qtype_label,
            "attribute": attr,
            "raw_attribute": attr,
            "metric_name": metric_name,
            "metric_value": sums["metric_sum"] / max(1, int(sums["n"])),
            "num_queries": int(sums["n"]),
            "selection_basis": "metrics_csv_attribute_proxy",
            "datasets": ";".join(sums["datasets"].keys()),
            "query_types": ";".join(sums["query_types"].keys()),
        })

    metric_rows = sorted(metric_rows, key=lambda r: r["metric_value"], reverse=high_is_weak)
    attribute_rows = sorted(attribute_rows, key=lambda r: r["metric_value"], reverse=high_is_weak)
    if not attribute_rows:
        raise ValueError(
            "No attribute metrics could be derived from KPI "
            f"captions filtered by query_types={selection_qtype_label!r}."
        )
    weak_groups = attribute_rows[:weak_attribute_topk or len(attribute_rows)]
    if not weak_groups:
        raise ValueError("No weak PAS groups were selected.")
    selection_basis = weak_groups[0]["selection_basis"]

    group_keys = [
        (row.get("attribute", ""),)
        for row in weak_groups
    ]
    group_rank = {key: idx for idx, key in enumerate(group_keys)}
    group_meta = {key: row for key, row in zip(group_keys, weak_groups)}
    per_group_budget = queries_per_slice

    def _matching_keys(row):
        attrs = _attributes(row.get("caption", ""))
        keys = [
            (attr,)
            for attr in attrs
            if (attr,) in group_meta
        ]
        return sorted(keys, key=lambda key: group_rank.get(key, 999999))

    caption_diversity_on = _truthy(caption_diversity_enabled)
    current_iter = max(1, int(iter_num or 1))
    planned_iters = max(current_iter, int(total_iters or current_iter))
    remaining_iters = max(1, planned_iters - current_iter + 1)
    continual_dataset_on = _truthy(continual_dataset)
    resolved_history_policy = str(caption_history_policy or "auto").strip().lower()
    if resolved_history_policy == "auto":
        resolved_history_policy = (
            "prefer_unseen" if continual_dataset_on else "novelty_with_replay"
        )
    caption_coverage_target = max(0.0, float(caption_coverage_target or 0.0))
    min_unique_texts_per_attribute = max(
        0, int(min_unique_texts_per_attribute or 0)
    )
    max_unique_texts_per_attribute = max(
        0, int(max_unique_texts_per_attribute or 0)
    )
    max_rows_per_unique_text = max(1, int(max_rows_per_unique_text or 1))
    max_rows_per_image_path = max(0, int(max_rows_per_image_path or 0))
    recent_exclude_iters = max(0, int(recent_exclude_iters or 0))
    replay_fraction_when_noncontinual = min(
        1.0, max(0.0, float(replay_fraction_when_noncontinual or 0.0))
    )

    def _serialize_group_key(key):
        return "||".join(str(part) for part in key)

    def _display_group_key(key):
        return " / ".join(str(part) for part in key if str(part))

    def _entry_iter(entry):
        try:
            return int(entry.get("iter", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            return 0

    history_payload = {"version": 1, "entries": []}
    history_entries = []
    if caption_diversity_on and caption_history_file and os.path.isfile(caption_history_file):
        try:
            with open(caption_history_file, "r", encoding="utf-8") as f:
                loaded_history = json.load(f)
            if isinstance(loaded_history, dict):
                history_payload.update(loaded_history)
                history_entries = list(loaded_history.get("entries") or [])
            elif isinstance(loaded_history, list):
                history_entries = list(loaded_history)
                history_payload["entries"] = history_entries
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"WARNING: could not read caption history file {caption_history_file} "
                f"({exc}); treating caption-diversity history as empty for this run."
            )
            history_entries = []

    history_counts = Counter()
    global_history_counts = Counter()
    prior_captions_by_group = defaultdict(set)
    prior_captions_global = set()
    recent_captions_by_group = defaultdict(set)
    recent_captions_global = set()
    for entry in history_entries:
        if not isinstance(entry, dict):
            continue
        entry_iter = _entry_iter(entry)
        if entry_iter >= current_iter:
            continue
        group_key_text = str(
            entry.get("group_key")
            or entry.get("attribute")
            or entry.get("weak_attribute")
            or ""
        )
        norm = str(
            entry.get("normalized_caption")
            or _normalize_caption_text(entry.get("caption", ""))
        )
        if not group_key_text or not norm:
            continue
        history_counts[(group_key_text, norm)] += 1
        global_history_counts[norm] += 1
        prior_captions_by_group[group_key_text].add(norm)
        prior_captions_global.add(norm)
        if recent_exclude_iters and entry_iter >= current_iter - recent_exclude_iters:
            recent_captions_by_group[group_key_text].add(norm)
            recent_captions_global.add(norm)

    caption_candidate_rows_by_key = defaultdict(dict)
    caption_pool_by_key = defaultdict(set)
    if caption_diversity_on:
        for order, row in enumerate(selection_pair_rows):
            norm = _normalize_caption_text(row.get("caption", ""))
            if not norm:
                continue
            for key in _matching_keys(row):
                caption_pool_by_key[key].add(norm)
                rows_for_text = caption_candidate_rows_by_key[key].setdefault(
                    norm, []
                )
                if len(rows_for_text) < max_rows_per_unique_text:
                    rows_for_text.append((order, row, norm))

    unique_text_budget_by_key = {}
    caption_diversity_stats = []
    if caption_diversity_on:
        for key in group_keys:
            pool = caption_pool_by_key.get(key, set())
            pool_count = len(pool)
            group_key_text = _serialize_group_key(key)
            seen_count = len(pool & prior_captions_by_group.get(group_key_text, set()))
            global_seen_count = len(pool & prior_captions_global)
            unseen_count = max(0, pool_count - global_seen_count)
            if unseen_count:
                auto_budget = int(
                    math.ceil((unseen_count * caption_coverage_target) / remaining_iters)
                )
            else:
                auto_budget = (
                    0 if resolved_history_policy == "prefer_unseen"
                    else pool_count
                )
            if (
                pool_count
                and min_unique_texts_per_attribute
                and (
                    unseen_count
                    or resolved_history_policy != "prefer_unseen"
                )
            ):
                auto_budget = max(
                    auto_budget,
                    min(min_unique_texts_per_attribute, pool_count),
                )
            if max_unique_texts_per_attribute:
                auto_budget = min(auto_budget, max_unique_texts_per_attribute)
            if per_group_budget:
                unique_cap_from_rows = max(
                    1,
                    int(math.ceil(per_group_budget / max_rows_per_unique_text)),
                )
                auto_budget = min(auto_budget, unique_cap_from_rows)
            auto_budget = min(max(0, auto_budget), pool_count)
            unique_text_budget_by_key[key] = auto_budget
            caption_diversity_stats.append({
                "group": _display_group_key(key),
                "group_key": group_key_text,
                "candidate_unique_texts": pool_count,
                "history_unique_texts": seen_count,
                "global_history_unique_texts": global_seen_count,
                "remaining_unseen_unique_texts": unseen_count,
                "unique_text_budget": auto_budget,
                "selected_unique_texts": 0,
                "selected_rows": 0,
            })

    selected_query_keys = set()
    selected_queries = []
    selected_text_counts_by_key = defaultdict(Counter)
    selected_unique_texts_by_key = defaultdict(set)
    selected_text_counts_global = Counter()
    selected_image_path_counts = Counter()
    selected_query_caption_info = {}
    per_group_counts = Counter()

    def _row_query_key(row):
        return (
            str(row.get("unique_name") or ""),
            str(row.get("query_type") or ""),
            str(row.get("caption") or ""),
        )

    def _append_selected(row, key, norm):
        name = str(row.get("unique_name") or "")
        query_key = _row_query_key(row)
        if query_key in selected_query_keys:
            return False
        selected_query_keys.add(query_key)
        selected_queries.append((row, key))
        selected_text_counts_by_key[key][norm] += 1
        selected_unique_texts_by_key[key].add(norm)
        selected_text_counts_global[norm] += 1
        image_key = str(row.get("image_path") or name)
        if image_key:
            selected_image_path_counts[image_key] += 1
        selected_query_caption_info[query_key] = {
            "normalized_text": norm,
            "caption_history_count": history_counts.get(
                (_serialize_group_key(key), norm), 0
            ),
            "group_key": _serialize_group_key(key),
        }
        per_group_counts[key] += 1
        return True

    def _can_select(row, key, norm, enforce_budget, unique_limit,
                    allow_seen, allow_recent, prefer_seen_only=False):
        if enforce_budget and per_group_budget and per_group_counts[key] >= per_group_budget:
            return False
        query_key = _row_query_key(row)
        if query_key in selected_query_keys:
            return False
        group_key_text = _serialize_group_key(key)
        prior_count = history_counts.get((group_key_text, norm), 0)
        global_prior_count = global_history_counts.get(norm, 0)
        if prefer_seen_only and prior_count == 0 and global_prior_count == 0:
            return False
        if not allow_seen and (prior_count > 0 or global_prior_count > 0):
            return False
        if (
            not allow_recent
            and (
                norm in recent_captions_by_group.get(group_key_text, set())
                or norm in recent_captions_global
            )
        ):
            return False
        if (
            max_rows_per_unique_text
            and selected_text_counts_by_key[key][norm] >= max_rows_per_unique_text
        ):
            return False
        if (
            max_rows_per_unique_text
            and selected_text_counts_global[norm] >= max_rows_per_unique_text
        ):
            return False
        is_new_text = norm not in selected_unique_texts_by_key[key]
        if unique_limit and is_new_text and len(selected_unique_texts_by_key[key]) >= unique_limit:
            return False
        image_key = str(row.get("image_path") or row.get("unique_name") or "")
        if (
            max_rows_per_image_path
            and image_key
            and selected_image_path_counts[image_key] >= max_rows_per_image_path
        ):
            return False
        return True

    def _candidate_sort_key(key, item, prefer_seen=False):
        order, _row, norm = item
        group_key_text = _serialize_group_key(key)
        prior_count = history_counts.get((group_key_text, norm), 0)
        global_prior_count = global_history_counts.get(norm, 0)
        recent = (
            norm in recent_captions_by_group.get(group_key_text, set())
            or norm in recent_captions_global
        )
        seen_rank = 0 if (prior_count > 0 or global_prior_count > 0) else 1
        if not prefer_seen:
            seen_rank = 0 if (prior_count == 0 and global_prior_count == 0) else 1
        return (seen_rank, bool(recent), prior_count + global_prior_count, order)

    def _select_candidates_for_group(key, unique_limit, allow_seen,
                                     allow_recent, prefer_seen=False,
                                     prefer_seen_only=False):
        candidate_rows = []
        for rows_for_text in caption_candidate_rows_by_key.get(key, {}).values():
            candidate_rows.extend(rows_for_text)
        candidates = sorted(
            candidate_rows,
            key=lambda item: _candidate_sort_key(key, item, prefer_seen=prefer_seen),
        )
        selected_any = False
        for _order, row, norm in candidates:
            if not _can_select(
                row,
                key,
                norm,
                enforce_budget=True,
                unique_limit=unique_limit,
                allow_seen=allow_seen,
                allow_recent=allow_recent,
                prefer_seen_only=prefer_seen_only,
            ):
                continue
            if _append_selected(row, key, norm):
                selected_any = True
        return selected_any

    def _select(row, enforce_budget):
        keys = _matching_keys(row)
        if not keys:
            return False
        key = keys[0]
        if enforce_budget and per_group_budget and per_group_counts[key] >= per_group_budget:
            return False
        norm = _normalize_caption_text(row.get("caption", ""))
        return _append_selected(row, key, norm)

    if caption_diversity_on:
        for key in group_keys:
            unique_budget = unique_text_budget_by_key.get(key, 0)
            if unique_budget <= 0:
                continue
            if resolved_history_policy == "novelty_with_replay":
                replay_budget = int(
                    math.floor(unique_budget * replay_fraction_when_noncontinual)
                )
                novelty_budget = max(0, unique_budget - replay_budget)
            else:
                replay_budget = 0
                novelty_budget = unique_budget
            if novelty_budget:
                _select_candidates_for_group(
                    key,
                    novelty_budget,
                    allow_seen=False,
                    allow_recent=False,
                )
                _select_candidates_for_group(
                    key,
                    novelty_budget,
                    allow_seen=False,
                    allow_recent=True,
                )
            if replay_budget:
                _select_candidates_for_group(
                    key,
                    unique_budget,
                    allow_seen=True,
                    allow_recent=True,
                    prefer_seen=True,
                    prefer_seen_only=True,
                )
            if (
                resolved_history_policy != "prefer_unseen"
                and len(selected_unique_texts_by_key[key]) < unique_budget
            ):
                _select_candidates_for_group(
                    key,
                    unique_budget,
                    allow_seen=True,
                    allow_recent=True,
                )
    else:
        for row in selection_pair_rows:
            _select(row, enforce_budget=True)

    if not selected_queries:
        for row in selection_pair_rows:
            keys = _matching_keys(row)
            if not keys:
                continue
            norm = _normalize_caption_text(row.get("caption", ""))
            if _append_selected(row, keys[0], norm):
                break

    if caption_diversity_on:
        stats_by_key = {
            stat["group_key"]: stat for stat in caption_diversity_stats
        }
        for key in group_keys:
            stat = stats_by_key.get(_serialize_group_key(key))
            if not stat:
                continue
            stat["selected_unique_texts"] = len(selected_unique_texts_by_key[key])
            stat["selected_rows"] = sum(selected_text_counts_by_key[key].values())

    def _record(row, keys, stage):
        caption = str(row.get("caption") or "")
        name = str(row.get("unique_name") or "")
        dataset = str(row.get("dataset") or "")
        qtype = str(row.get("query_type") or "")
        metas = [group_meta[key] for key in keys if key in group_meta]
        meta = metas[0] if metas else {
            "metric_name": metric_name, "metric_value": "",
            "num_queries": "", "selection_basis": selection_basis,
        }
        weak_attrs = [
            str(m.get("attribute") or "")
            for m in metas
            if str(m.get("attribute") or "")
        ]
        weak_attr_text = ";".join(dict.fromkeys(weak_attrs))
        attrs = _attributes(caption)
        norm_caption = _normalize_caption_text(caption)
        query_info = selected_query_caption_info.get(_row_query_key(row), {})
        if not query_info and keys:
            query_info = {
                "normalized_text": norm_caption,
                "caption_history_count": history_counts.get(
                    (_serialize_group_key(keys[0]), norm_caption), 0
                ),
                "group_key": _serialize_group_key(keys[0]),
            }
        return {
            "filepath": os.path.abspath(os.path.join(kpi_image_dir, name))
            if kpi_image_dir and name else "",
            "text": caption,
            "normalized_text": query_info.get("normalized_text", norm_caption),
            "label": ":".join(x for x in (dataset, qtype, weak_attr_text) if x),
            "dataset": dataset,
            "query_type": qtype,
            "weak_attribute": weak_attr_text,
            "easy_attribute": weak_attr_text,
            "attributes": ";".join(attrs),
            "colors": ";".join(_colors(caption)),
            "color_item_pairs": ";".join(_color_item_pairs(caption)),
            "metric_name": meta.get("metric_name", metric_name),
            "metric_value": meta.get("metric_value", ""),
            "num_queries": meta.get("num_queries", ""),
            "selection_basis": meta.get("selection_basis", selection_basis),
            "selection_stage": stage,
            "caption_history_count": query_info.get("caption_history_count", 0),
            "caption_history_group_key": query_info.get("group_key", ""),
            "unique_name": name,
            "image_path": str(row.get("image_path") or ""),
        }

    records = []
    for row, key in selected_queries:
        records.append(_record(row, [key], "selected_query"))

    columns = [
        "filepath", "text", "label", "dataset", "query_type",
        "weak_attribute", "easy_attribute", "attributes", "colors",
        "color_item_pairs", "metric_name", "metric_value", "num_queries",
        "selection_basis", "selection_stage",
        "normalized_text", "caption_history_count",
        "caption_history_group_key",
        "unique_name", "image_path",
    ]
    gaps_df = pd.DataFrame(records, columns=columns)
    gaps_dir = os.path.dirname(gaps_parquet)
    if gaps_dir:
        os.makedirs(gaps_dir, exist_ok=True)
    gaps_df.to_parquet(gaps_parquet, index=False)

    os.makedirs(logs_dir, exist_ok=True)
    summary_path = os.path.join(logs_dir, "weak_samples_breakdown.txt")
    groups_path = os.path.join(logs_dir, "weak_pas_attributes.csv")
    all_attributes_path = os.path.join(logs_dir, "pas_attribute_metrics.csv")
    samples_csv_path = os.path.join(logs_dir, "weak_pas_samples.csv")
    samples_jsonl_path = os.path.join(logs_dir, "weak_pas_samples.jsonl")
    preview_path = os.path.join(logs_dir, "weak_pas_samples_preview.txt")
    caption_diversity_summary_path = os.path.join(
        logs_dir, "caption_diversity_summary.csv"
    )
    pd.DataFrame(weak_groups).to_csv(groups_path, index=False)
    pd.DataFrame(attribute_rows or metric_rows).to_csv(all_attributes_path, index=False)
    gaps_df.to_csv(samples_csv_path, index=False)
    gaps_df.to_json(samples_jsonl_path, orient="records", lines=True, force_ascii=False)
    if caption_diversity_on:
        pd.DataFrame(caption_diversity_stats).to_csv(
            caption_diversity_summary_path, index=False
        )
        if caption_history_file:
            history_dir = os.path.dirname(caption_history_file)
            if history_dir:
                os.makedirs(history_dir, exist_ok=True)
            existing_entries = [
                entry for entry in history_entries
                if isinstance(entry, dict) and _entry_iter(entry) != current_iter
            ]
            new_entries = []
            for row, key in selected_queries:
                caption = str(row.get("caption") or "")
                norm = _normalize_caption_text(caption)
                if not norm:
                    continue
                new_entries.append({
                    "iter": current_iter,
                    "group": _display_group_key(key),
                    "group_key": _serialize_group_key(key),
                    "caption": caption,
                    "normalized_caption": norm,
                    "weak_attribute": key[0] if key else "",
                    "dataset": str(row.get("dataset") or ""),
                    "query_type": str(row.get("query_type") or ""),
                    "unique_name": str(row.get("unique_name") or ""),
                    "image_path": str(row.get("image_path") or ""),
                })
            history_payload.update({
                "version": 1,
                "last_iter": current_iter,
                "history_policy": resolved_history_policy,
                "continual_dataset": continual_dataset_on,
                "entries": existing_entries + new_entries,
            })
            history_tmp_file = caption_history_file + ".tmp"
            with open(history_tmp_file, "w", encoding="utf-8") as f:
                json.dump(history_payload, f, indent=2, ensure_ascii=False)
            os.replace(history_tmp_file, caption_history_file)

    preview_lines = ["PAS weak query sample preview", ""]
    for idx, row in gaps_df.head(100).iterrows():
        text_preview = str(row.get("text") or "").replace("\n", " ")[:240]
        preview_lines.append(
            f"[{idx}] {row.get('dataset', '')} / {row.get('query_type', '')} "
            f"{row.get('weak_attribute', '')} "
            f"{row.get('metric_name', metric_name)}={row.get('metric_value', '')}: "
            f"{text_preview}"
        )
    if len(preview_lines) == 2:
        preview_lines.append("No weak query rows emitted.")
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write("\n".join(preview_lines) + "\n")

    count_lines = []
    if not gaps_df.empty:
        group_cols = ["dataset", "query_type"]
        group_cols.append("weak_attribute")
        count_df = (
            gaps_df.groupby(group_cols, dropna=False)
            .size()
            .reset_index(name="sampled")
            .sort_values(group_cols)
        )
        for _, row in count_df.iterrows():
            attr = (
                f" / {row['weak_attribute']}"
                if "weak_attribute" in row and row.get("weak_attribute") else ""
            )
            count_lines.append(
                f"  {row['dataset']} / {row['query_type']}{attr}: "
                f"{int(row['sampled'])}"
            )

    rank_direction = "highest" if high_is_weak else "lowest"
    rank_basis_note = f"ranked by {rank_direction} {metric_name} (ties broken by attribute ordering)"

    summary_lines = [
        "PAS weak-attribute gap analysis",
        f"Metrics: {metrics_path}",
        f"KPI pairs: {kpi_pairs_file}",
        f"Weak metric: {metric_name} ({'higher' if high_is_weak else 'lower'} is weaker)",
        f"Selection basis: {selection_basis}",
        f"Weak groups picked ({rank_basis_note}): {len(weak_groups)}",
        f"Attribute selection query types: {selection_qtype_label}",
        f"KPI selection pair rows: {len(selection_pair_rows)}",
        f"Mined output target query budget: {target_query_count or 'unlimited'}",
        f"Weak driver per-attribute row budget: {per_group_budget or 'unlimited'}",
        f"Caption diversity enabled: {caption_diversity_on}",
        f"Caption history policy: {resolved_history_policy}",
        f"Caption history file: {caption_history_file or 'disabled'}",
        f"Caption diversity summary CSV: {caption_diversity_summary_path if caption_diversity_on else 'disabled'}",
        "Note: mined output budget is applied after kNN and source-pair recovery.",
        f"Selected seed query rows: {len(selected_queries)}",
        f"Weak query rows emitted ({rank_basis_note}): {len(gaps_df)}",
        f"Skipped/malformed pair rows: {skipped_pairs}",
        f"Weak samples parquet: {gaps_parquet}",
        f"Weak samples CSV: {samples_csv_path}",
        f"Weak samples JSONL: {samples_jsonl_path}",
        f"Weak samples preview: {preview_path}",
        f"Weak group CSV: {groups_path}",
        f"All attribute metric CSV: {all_attributes_path}",
    ]
    if selection_qtype_counts:
        summary_lines.extend(["", "KPI selection rows by query type:"])
        for qtype, count in selection_qtype_counts.most_common():
            summary_lines.append(f"  {qtype}: {count}")
    if caption_diversity_on and caption_diversity_stats:
        summary_lines.extend(["", "Caption diversity by weak group:"])
        for stat in caption_diversity_stats:
            summary_lines.append(
                f"  {stat['group']}: "
                f"candidate_unique_texts={stat['candidate_unique_texts']}, "
                f"history_unique_texts={stat['history_unique_texts']}, "
                f"global_history_unique_texts={stat['global_history_unique_texts']}, "
                f"remaining_unseen={stat['remaining_unseen_unique_texts']}, "
                f"unique_text_budget={stat['unique_text_budget']}, "
                f"selected_unique_texts={stat['selected_unique_texts']}, "
                f"selected_rows={stat['selected_rows']}"
            )
    summary_lines.extend(["", "Selected weak groups:"])
    for row in weak_groups:
        key = (row.get("attribute", ""),)
        summary_lines.append(
            f"  {row.get('attribute', '')}: "
            f"{row['metric_name']}={float(row['metric_value']):.6g}, "
            f"selection_queries={row['num_queries']}, "
            f"selected_rows={per_group_counts.get(key, 0)}, "
            f"basis={row.get('selection_basis', '')}"
        )
    if count_lines:
        summary_lines.extend(["", "Weak query row counts:"])
        summary_lines.extend(count_lines)
    summary_text = "\n".join(summary_lines) + "\n"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text.strip())
    print(f"Saved weak PAS groups to {groups_path}")
    print(f"Saved weak PAS samples to {samples_csv_path}")
    print(f"Saved weak PAS sample preview to {preview_path}")
    print(f"Saved weak query parquet to {gaps_parquet}")
    return gaps_parquet


def summarize_pas_eval_metrics(
    eval_dir: str,
    query_types: str = "",
    metric_names: str = "mAP,Rank-1,Rank-5",
    aggregate: str = "weighted",
) -> dict:
    """Compute the single headline metrics for a PAS CLIP eval run.

    The PAS eval writes three sibling CSVs per (Dataset, QueryType) round:
    ``nvidia_pas_metrics.csv`` (one row per dataset/query-type, no rollups),
    ``nvidia_pas_metrics_aggregate.csv`` (unweighted ``AVG_*`` rollup rows),
    and ``nvidia_pas_metrics_weighted_aggregate.csv`` (``WAVG_*`` rollup
    rows, each dataset weighted by its ``num_queries``). Different readers
    can pick different files/rows/columns and land on different "the mAP
    for this round" numbers. This function is the one place that decision
    is made, so every caller (zero-shot summary, per-iteration summary,
    ad-hoc analysis) reads the same number without re-deriving it from the
    CSVs by hand.

    Fixed convention:
      * Source file / row selection: the weighted-aggregate CSV's
        ``WAVG_*`` rows by default (each dataset weighted by its
        ``num_queries``), not the unweighted-aggregate CSV's ``AVG_*`` rows
        or a single dataset's row from the per-dataset CSV. Pass
        ``aggregate="unweighted"`` to read ``AVG_*`` from the unweighted
        aggregate CSV instead.
      * Query types: ``image_to_image`` rows are always excluded (mirrors
        gap analysis). If ``query_types`` is given, only those are kept;
        otherwise every remaining query type is kept and combined into one
        number, weighted by ``num_queries``.
      * Metric columns: ``metric_names`` (default ``mAP``, ``Rank-1``,
        ``Rank-5``). Each metric's value is a weighted average over only the
        matched rows where that column is present and parseable; a metric
        with no usable values across all matched rows raises ``ValueError``
        rather than being silently reported as ``0.0``.

    Args:
        eval_dir:      Experiment/round root dir; its ``evaluate/`` subdir
                       must directly contain nvidia_pas_metrics.csv and its
                       ``_aggregate``/``_weighted_aggregate`` siblings.
        query_types:   Optional comma-separated query-type filter.
        metric_names:  Comma-separated metric columns to extract.
        aggregate:     ``"weighted"`` (WAVG_* from the weighted-aggregate
                       CSV) or ``"unweighted"`` (AVG_* from the aggregate
                       CSV).

    Returns:
        Dict with the resolved metric values, the total ``num_queries``
        behind them, and the source CSV path, e.g.::

            {"source_csv": ..., "aggregate": "weighted",
             "query_types": ["text_to_image"], "num_queries": 1234,
             "mAP": 0.42, "Rank-1": 0.30, "Rank-5": 0.58}
    """
    import csv
    import os

    from pas_deft.pairs_io import split_csv

    if aggregate not in ("weighted", "unweighted"):
        raise ValueError(f"aggregate must be 'weighted' or 'unweighted', got {aggregate!r}")
    if aggregate == "weighted":
        prefix = "WAVG_"
        agg_filename = "nvidia_pas_metrics_weighted_aggregate.csv"
    else:
        prefix = "AVG_"
        agg_filename = "nvidia_pas_metrics_aggregate.csv"

    metrics_path = os.path.join(eval_dir, "evaluate", agg_filename)
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Could not find {agg_filename} at {metrics_path}")

    qtype_filter = split_csv(query_types)
    metric_cols = [m.strip() for m in str(metric_names or "").split(",") if m.strip()]
    if not metric_cols:
        raise ValueError(f"metric_names must name at least one metric column, got {metric_names!r}")

    sums = {metric: 0.0 for metric in metric_cols}
    metric_query_counts = {metric: 0 for metric in metric_cols}
    total_queries = 0
    matched_query_types = set()
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = str(row.get("Dataset") or "").strip()
            if not dataset.startswith(prefix):
                continue
            qtype = str(row.get("QueryType") or "").strip()
            if qtype == "image_to_image":
                continue
            if qtype_filter and qtype not in qtype_filter:
                continue
            try:
                n_queries = int(float(row.get("num_queries") or 0))
            except ValueError:
                n_queries = 0
            if n_queries <= 0:
                continue
            for metric in metric_cols:
                value = row.get(metric, "")
                if value in ("", None):
                    continue
                try:
                    sums[metric] += float(value) * n_queries
                except ValueError:
                    continue
                else:
                    metric_query_counts[metric] += n_queries
            total_queries += n_queries
            matched_query_types.add(qtype or "(blank)")

    if total_queries == 0:
        raise ValueError(
            f"No {prefix}* rows matched query_types={query_types or 'all'!r} in {metrics_path}"
        )

    missing_metrics = [metric for metric in metric_cols if metric_query_counts[metric] == 0]
    if missing_metrics:
        raise ValueError(
            f"No usable values for metric(s) {missing_metrics} among {prefix}* rows "
            f"matched query_types={query_types or 'all'!r} in {metrics_path}"
        )

    result = {
        "source_csv": metrics_path,
        "aggregate": aggregate,
        "query_types": sorted(matched_query_types),
        "num_queries": total_queries,
    }
    for metric in metric_cols:
        result[metric] = sums[metric] / metric_query_counts[metric]
    return result
