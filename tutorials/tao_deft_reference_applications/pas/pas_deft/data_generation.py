"""Dataset augmentation utilities for the PAS SDG (Synthetic Data Generation) pipeline.

These functions select which images to send to the hosted SDG service and compute
the residual attribute distribution that SDG should generate against — i.e., what
the mined set didn't already cover.
"""

from __future__ import annotations

import ast
import datetime
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image


MISSING_VALUE = "<missing>"
OUTPUT_MISSING_VALUE = "__missing__"


class DataDistribution:
    """Marginal distribution over a fixed attribute alphabet."""

    def __init__(self, alphabet: List[str], probs=None, num_data: int = 0):
        if not alphabet:
            raise ValueError("DataDistribution requires a non-empty alphabet")
        self._alphabet = [str(v) for v in alphabet]
        if MISSING_VALUE not in self._alphabet:
            self._alphabet.append(MISSING_VALUE)
        if len(set(self._alphabet)) != len(self._alphabet):
            raise ValueError(f"Alphabet contains duplicate values: {self._alphabet}")
        self.num_data = float(num_data)
        if probs is None:
            self._data_counts = np.zeros(len(self._alphabet), dtype=float)
            self._probs = np.zeros(len(self._alphabet), dtype=float)
        else:
            self._probs = np.array(probs, dtype=float)
            if len(self._probs) != len(self._alphabet):
                raise ValueError("Probability vector must match fixed alphabet length")
            self._data_counts = self.num_data * self._probs
            self._sync_probs()

    @property
    def alphabet(self) -> Tuple[str, ...]:
        return tuple(self._alphabet)

    @property
    def data_counts(self) -> np.ndarray:
        return self._data_counts.copy()

    @property
    def probs(self) -> np.ndarray:
        return self._probs.copy()

    def _sync_probs(self):
        if self.num_data == 0:
            self._probs = np.zeros(len(self._alphabet), dtype=float)
        else:
            self._probs = self._data_counts / self.num_data

    def _value_index(self, value) -> int:
        value = str(value)
        if value not in self._alphabet:
            raise ValueError(
                f"Value {value!r} is not in fixed alphabet {tuple(self._alphabet)}"
            )
        return self._alphabet.index(value)

    def _increment_from_assignment(self, value, count: int = 1):
        idx = self._value_index(value)
        self._data_counts[idx] += count
        self.num_data += count
        self._sync_probs()

    def _set_counts_from_bottom_up_result(self, alphabet: List[str], data_counts):
        alphabet = [str(v) for v in alphabet]
        data_counts = np.array(data_counts, dtype=float)
        if tuple(alphabet) != self.alphabet:
            raise ValueError(
                f"Bottom-up result alphabet {tuple(alphabet)} does not match "
                f"fixed alphabet {self.alphabet}"
            )
        if len(data_counts) != len(self._alphabet):
            raise ValueError("Data counts must match fixed alphabet length")
        self._data_counts = data_counts
        self.num_data = float(np.sum(self._data_counts))
        self._sync_probs()

    def _multiply_counts(self, scale_factor: float):
        self._data_counts = self._data_counts * scale_factor
        self.num_data = float(np.sum(self._data_counts))
        self._sync_probs()

    def probability_dict(self) -> Dict[str, float]:
        return {
            OUTPUT_MISSING_VALUE if value == MISSING_VALUE else value: float(prob)
            for value, prob in zip(self.alphabet, self.probs)
            if prob > 0
        }

    def to_dict(self) -> dict:
        return {
            "alphabet": list(self.alphabet),
            "num_data": self.num_data,
            "data_counts": self.data_counts.tolist(),
            "probs": self.probs.astype(float).tolist(),
        }


class DataDistributionNode:
    """Distribution for one attribute, with fully enumerated conditional children."""

    def __init__(self, attr_name: str, alphabet: List[str]):
        self.attr_name = attr_name
        self.distribution = DataDistribution(alphabet=alphabet)
        self.children: Dict[str, Dict[str, "DataDistributionNode"]] = {}
        self.child_templates: Dict[str, "DataDistributionNode"] = {}

    def _non_missing_values(self) -> List[str]:
        return [v for v in self.distribution.alphabet if v != MISSING_VALUE]

    def add_child_template(self, child: "DataDistributionNode"):
        if self.child_templates:
            raise ValueError(
                f"Schema node {self.attr_name} can have at most one conditional child attribute"
            )
        self.child_templates[child.attr_name] = child
        self.children[child.attr_name] = {
            parent_value: child.clone_empty()
            for parent_value in self._non_missing_values()
        }

    def clone_empty(self) -> "DataDistributionNode":
        clone = DataDistributionNode(self.attr_name, alphabet=list(self.distribution.alphabet))
        for child in self.child_templates.values():
            clone.add_child_template(child.clone_empty())
        return clone

    def _contains_assignment_attr(self, assignment: dict) -> bool:
        if self.attr_name in assignment:
            return True
        return any(
            child._contains_assignment_attr(assignment)
            for child in self.child_templates.values()
        )

    def _has_descendant_assignment_attr(self, assignment: dict) -> bool:
        return any(
            child._contains_assignment_attr(assignment)
            for child in self.child_templates.values()
        )

    def _update_from_root_assignment(self, assignment: dict, count: int = 1):
        attr_value = str(assignment.get(self.attr_name, MISSING_VALUE))
        self.distribution._increment_from_assignment(attr_value, count=count)
        if attr_value == MISSING_VALUE:
            return
        for child_attr in self.child_templates:
            self.children[child_attr][attr_value]._update_from_root_assignment(
                assignment, count=count
            )

    def _multiply_counts(self, scale_factor: float):
        self.distribution._multiply_counts(scale_factor)
        for nodes_by_parent_value in self.children.values():
            for child_node in nodes_by_parent_value.values():
                child_node._multiply_counts(scale_factor)

    def _copy_scaled_counts_from(self, other: "DataDistributionNode", scale_factor: float):
        self.distribution._set_counts_from_bottom_up_result(
            list(other.distribution.alphabet),
            other.distribution.data_counts * scale_factor,
        )
        for child_attr, child_template in self.child_templates.items():
            for parent_value in self._non_missing_values():
                self.children[child_attr][parent_value]._copy_scaled_counts_from(
                    other.children[child_attr][parent_value], scale_factor
                )

    def to_log_probability_dict(self) -> dict:
        result: dict = {"distribution": self.distribution.probability_dict()}
        for child_attr, nodes_by_parent_value in self.children.items():
            result["conditionals"] = {
                parent_value: node.to_log_probability_dict()
                for parent_value, node in nodes_by_parent_value.items()
                if node.distribution.num_data > 0
            }
        return result


class DataDistributionForest:
    """Collection of root :class:`DataDistributionNode` trees, one per schema entry."""

    def __init__(self, schema: List[dict]):
        self.schema = schema
        self._attr_names: List[str] = []
        self.roots: List[DataDistributionNode] = []
        self.roots_by_attr: Dict[str, DataDistributionNode] = {}
        for entry in schema:
            root = self._build_node(entry)
            self.roots.append(root)
            self.roots_by_attr[root.attr_name] = root
            self._attr_names.append(root.attr_name)

    def _build_node(self, entry: dict) -> DataDistributionNode:
        attr_name = str(entry["attr_name"])
        alphabet = [str(v) for v in entry["alphabet"]]
        node = DataDistributionNode(attr_name, alphabet)
        child_entry = entry.get("conditional_child")
        if child_entry:
            node.add_child_template(self._build_node(child_entry))
        return node

    def update(self, assignment: dict):
        for root in self.roots:
            root._update_from_root_assignment(assignment)

    def multiplied(self, scale_factor: float) -> "DataDistributionForest":
        result = DataDistributionForest(self.schema)
        for root, result_root in zip(self.roots, result.roots):
            result_root._copy_scaled_counts_from(root, scale_factor)
        return result

    def validate_counts(self):
        for root in self.roots:
            if root.distribution.num_data < 0:
                raise ValueError(
                    f"Negative count for {root.attr_name}: {root.distribution.num_data}"
                )

    def log_probability_schema(self, output_path: str):
        schema_out = []
        for root in self.roots:
            entry = {"attr_name": root.attr_name}
            entry.update(root.to_log_probability_dict())
            schema_out.append(entry)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(schema_out, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_json(cls, json_path: str) -> "DataDistributionForest":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data


def _subtract_distribution_forests(
    schema: List[dict],
    minuend: DataDistributionForest,
    subtrahend: DataDistributionForest,
) -> DataDistributionForest:
    """Subtract two forests from leaves upward, clamping at zero."""

    def _subtract_nodes_bottom_up(
        result_node: DataDistributionNode,
        minuend_node: DataDistributionNode,
        subtrahend_node: DataDistributionNode,
    ):
        if minuend_node.distribution.alphabet != subtrahend_node.distribution.alphabet:
            raise ValueError(
                f"Cannot subtract {result_node.attr_name}: alphabets differ "
                f"({minuend_node.distribution.alphabet} != "
                f"{subtrahend_node.distribution.alphabet})"
            )
        alphabet = list(minuend_node.distribution.alphabet)
        direct_counts = np.maximum(
            minuend_node.distribution.data_counts - subtrahend_node.distribution.data_counts,
            0,
        )
        if not result_node.child_templates:
            result_node.distribution._set_counts_from_bottom_up_result(alphabet, direct_counts)
            return
        if len(result_node.child_templates) != 1:
            raise ValueError(
                f"Schema node {result_node.attr_name} must have exactly one "
                "conditional child or be a leaf"
            )
        parent_counts = direct_counts.copy()
        child_attr = next(iter(result_node.child_templates))
        for parent_value in result_node._non_missing_values():
            result_child = result_node.children[child_attr][parent_value]
            _subtract_nodes_bottom_up(
                result_child,
                minuend_node.children[child_attr][parent_value],
                subtrahend_node.children[child_attr][parent_value],
            )
            parent_idx = alphabet.index(parent_value)
            parent_counts[parent_idx] = result_child.distribution.num_data
        result_node.distribution._set_counts_from_bottom_up_result(alphabet, parent_counts)

    result = DataDistributionForest(schema)
    for root in result.roots:
        _subtract_nodes_bottom_up(
            root,
            minuend.roots_by_attr[root.attr_name],
            subtrahend.roots_by_attr[root.attr_name],
        )
    result.validate_counts()
    return result


def _is_missing_scalar(value) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _parse_attr_vector(raw_vector, vector_name: str) -> list:
    if _is_missing_scalar(raw_vector):
        return []
    if hasattr(raw_vector, "as_py"):
        raw_vector = raw_vector.as_py()
    if hasattr(raw_vector, "tolist"):
        raw_vector = raw_vector.tolist()
    if isinstance(raw_vector, str):
        raw_vector = raw_vector.strip()
        if not raw_vector:
            return []
        try:
            raw_vector = json.loads(raw_vector)
        except json.JSONDecodeError:
            raw_vector = ast.literal_eval(raw_vector)
    if isinstance(raw_vector, tuple):
        raw_vector = list(raw_vector)
    if not isinstance(raw_vector, list):
        raise ValueError(f"{vector_name} must be a list, got {type(raw_vector).__name__}")
    return raw_vector


def _vocab_attr_name(attr_name: str) -> str:
    return str(attr_name).strip().replace(" ", "_")


def _is_viewpoint(attr_name: str) -> bool:
    return _vocab_attr_name(attr_name) == "viewpoint"


def _attr_value_lookup(attr_vocab: dict) -> Dict[str, list]:
    lookup: Dict[str, list] = {}
    for attr_name, values in attr_vocab["id_to_value"].items():
        if not isinstance(values, list):
            raise ValueError(f"Vocabulary values for {attr_name} must be a list")
        normalized = _vocab_attr_name(attr_name)
        if normalized in lookup:
            raise ValueError(
                f"Attribute vocab keys normalize to the same name: {normalized}"
            )
        lookup[normalized] = values
    return lookup


def _decode_attr_value(
    raw_attr_name: str,
    attr_id: Any,
    attr_values_by_attr: Dict[str, list],
) -> Optional[str]:
    if _is_missing_scalar(attr_id):
        return None
    if raw_attr_name not in attr_values_by_attr:
        raise ValueError(f"No id_to_value mapping found for attribute {raw_attr_name}")
    try:
        attr_id = int(attr_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Attribute id for {raw_attr_name} must be an integer: {attr_id}"
        ) from exc
    attr_values = attr_values_by_attr[raw_attr_name]
    if attr_id < 0 or attr_id >= len(attr_values):
        raise ValueError(
            f"Attribute id {attr_id} for {raw_attr_name} is outside vocabulary "
            f"length {len(attr_values)}"
        )
    attr_value = attr_values[attr_id]
    if _is_missing_scalar(attr_value):
        return None
    attr_value = str(attr_value)
    if attr_value in {"", OUTPUT_MISSING_VALUE, MISSING_VALUE}:
        return MISSING_VALUE if attr_value else None
    return attr_value


def _assignment_from_attr_vector(
    raw_vector: Any,
    attr_vocab: dict,
    schema_attr_names: set,
    attr_values_by_attr: Dict[str, list],
    vector_name: str,
) -> dict:
    attr_vector = _parse_attr_vector(raw_vector, vector_name)
    attr_names = attr_vocab["attributes"]
    if len(attr_vector) > len(attr_names):
        raise ValueError(
            f"{vector_name} has {len(attr_vector)} values but attr_vocab has "
            f"{len(attr_names)} attributes"
        )
    assignment: dict = {}
    for attr_idx, raw_attr_name in enumerate(attr_names[: len(attr_vector)]):
        attr_name = _vocab_attr_name(raw_attr_name)
        if _is_viewpoint(attr_name):
            continue
        attr_value = _decode_attr_value(attr_name, attr_vector[attr_idx], attr_values_by_attr)
        if attr_value is None:
            continue
        if attr_name not in schema_attr_names:
            raise ValueError(
                f"Attribute {attr_name} is not in the payload schema; "
                "only viewpoint is skipped"
            )
        assignment[attr_name] = attr_value
    return assignment


def _tabulate_attr_vectors(
    forest: DataDistributionForest,
    raw_vectors,
    source_name: str,
    vector_name: str,
    attr_vocab: dict,
    schema_attr_names: set,
    attr_values_by_attr: Dict[str, list],
):
    for row_idx, raw_vector in enumerate(raw_vectors):
        try:
            assignment = _assignment_from_attr_vector(
                raw_vector,
                attr_vocab,
                schema_attr_names,
                attr_values_by_attr,
                vector_name,
            )
            forest.update(assignment)
        except Exception as exc:
            raise ValueError(
                f"Failed to tabulate {source_name} {vector_name} at row {row_idx}: {exc}"
            ) from exc
    forest.validate_counts()


def _iter_mined_pair_records(payload: Any) -> List[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "image_attr_values" in payload or "image_attr_vector" in payload:
            return [payload]
        for records_key in ("pairs", "records", "data", "items", "mined_pairs"):
            records = payload.get(records_key)
            if isinstance(records, list):
                return records
    raise ValueError("delta_mined_pairs_json must contain a list of mined pair records")


def _image_attr_values_from_mined_record(record: dict) -> Any:
    if not isinstance(record, dict):
        raise ValueError(f"Mined pair record must be a dict, got {type(record).__name__}")
    for vector_key in (
        "image_attr_values",
        "image_attr_vector",
        "attr_vector",
        "attributes_vector",
    ):
        if vector_key in record:
            return record[vector_key]
    raise ValueError("Mined pair record is missing image_attr_values")


def _forest_size(forest: DataDistributionForest) -> float:
    forest.validate_counts()
    if not forest.roots:
        return 0.0
    root_sizes = [root.distribution.num_data for root in forest.roots]
    if not all(np.isclose(root_size, root_sizes[0]) for root_size in root_sizes):
        raise ValueError(f"Forest root sizes do not agree: {root_sizes}")
    return float(root_sizes[0])


def augment_images_using_global_attr_dist_matching(
    datagen_iter_path: str,
    weak_queries_parquet: str,
    delta_mined_pairs_json: str,
    schema_path: str,
    attr_vocab_path: str,
    scale_factor: float,
    root_image_dir: str,
    payload_template_path: str,
    s3_root: str,
    max_budget: int,
    caption_policy: str = "all",
) -> None:
    """Compute the residual attribute gap and write SDG runner inputs.

    Selects which images should be sent to the hosted SDG service and computes
    the residual attribute distribution that SDG should generate against —
    i.e., the weak-query distribution scaled by ``scale_factor`` minus what
    the mined set already covers.

    Writes three files under ``datagen_iter_path``:

    * ``image_names_to_augment.txt`` — absolute image paths to send to SDG.
    * ``scaled_weak_images_forest_minus_delta_mined_images_schema.yaml`` —
      residual attribute distribution YAML consumed by the SDG payload builder.
    * ``pas_sdg_runner_config.yaml`` — runner config referencing the above.

    Args:
        datagen_iter_path:    Output directory for SDG inputs.
        weak_queries_parquet: Gap-analysis output parquet with ``image_attr_vector``.
        delta_mined_pairs_json: Delta mined pairs JSON for this iteration.
        schema_path:          Path to the payload schema YAML/JSON.
        attr_vocab_path:      Path to the attribute vocabulary JSON.
        scale_factor:         Fraction of mined images to send (0, 1].
        root_image_dir:       Absolute path to the source image directory.
        payload_template_path: Path to the baseline SDG payload template.
        s3_root:              S3 root URI for the hosted SDG service.
        max_budget:           Hard cap on the number of images; -1 means no cap.
    """
    with open(schema_path, "r") as f:
        schema_text = f.read()
    try:
        payload_schema = json.loads(schema_text)
    except json.JSONDecodeError:
        payload_schema = yaml.safe_load(schema_text)
    if not isinstance(payload_schema, list):
        raise ValueError("Payload schema must be a list of distribution trees")

    with open(attr_vocab_path, "r") as f:
        attr_vocab = json.load(f)
    if not isinstance(attr_vocab.get("attributes"), list):
        raise ValueError("Attribute vocabulary must contain an attributes list")
    if not isinstance(attr_vocab.get("id_to_value"), dict):
        raise ValueError("Attribute vocabulary must contain an id_to_value mapping")

    weak_images_forest = DataDistributionForest(payload_schema)
    delta_mined_images_forest = DataDistributionForest(payload_schema)
    schema_attr_names = set(weak_images_forest._attr_names)
    attr_values_by_attr = _attr_value_lookup(attr_vocab)

    weak_queries = pd.read_parquet(weak_queries_parquet)
    if "image_attr_vector" not in weak_queries.columns:
        raise ValueError("weak_queries_parquet must contain an image_attr_vector column")
    _tabulate_attr_vectors(
        weak_images_forest,
        weak_queries["image_attr_vector"],
        "weak_queries",
        "image_attr_vector",
        attr_vocab,
        schema_attr_names,
        attr_values_by_attr,
    )

    with open(delta_mined_pairs_json, "r") as f:
        delta_mined_payload = json.load(f)
    delta_mined_records = _iter_mined_pair_records(delta_mined_payload)
    image_names_to_augment = [
        os.path.join(root_image_dir, record["unique_name"])
        for record in delta_mined_records
    ]
    _tabulate_attr_vectors(
        delta_mined_images_forest,
        (_image_attr_values_from_mined_record(record) for record in delta_mined_records),
        "delta_mined_pairs",
        "image_attr_values",
        attr_vocab,
        schema_attr_names,
        attr_values_by_attr,
    )

    if scale_factor < 0:
        raise ValueError("scale_factor must be non-negative")
    weak_images_size = _forest_size(weak_images_forest)
    delta_mined_images_size = _forest_size(delta_mined_images_forest)
    scaled_weak_images_size = (float(scale_factor) + 1.0) * delta_mined_images_size
    if weak_images_size == 0:
        if scaled_weak_images_size > 0:
            raise ValueError(
                "Cannot scale an empty weak_images_forest to a non-zero target size"
            )
        weak_images_scale_factor = 0.0
    else:
        weak_images_scale_factor = int(scaled_weak_images_size / weak_images_size)

    scaled_weak_images_forest = weak_images_forest.multiplied(weak_images_scale_factor)
    scaled_weak_minus_delta = _subtract_distribution_forests(
        payload_schema,
        scaled_weak_images_forest,
        delta_mined_images_forest,
    )

    assert 0 < scale_factor <= 1, (
        "scale_factor must be between 0 and 1. Greater than 1 is not supported yet."
    )

    pre_clipped = int(scale_factor * len(image_names_to_augment))
    img_input_count = min(pre_clipped, max_budget) if max_budget > 0 else pre_clipped
    image_names_to_augment = image_names_to_augment[:img_input_count]

    os.makedirs(datagen_iter_path, exist_ok=True)

    image_list_path = os.path.join(datagen_iter_path, "image_names_to_augment.txt")
    with open(image_list_path, "w") as f:
        for image_name in image_names_to_augment:
            f.write(image_name + "\n")
    print(f"Wrote {len(image_names_to_augment)} image names to {image_list_path}")

    output_schema_path = os.path.join(
        datagen_iter_path,
        "scaled_weak_images_forest_minus_delta_mined_images_schema.yaml",
    )
    scaled_weak_minus_delta.log_probability_schema(output_schema_path)
    print(f"Saved scaled weak-minus-delta mined schema to {output_schema_path}")

    runner_config = {
        "output_dir": os.path.join(datagen_iter_path, "pas_sdg_output"),
        "payload_path": payload_template_path,
        "attribute_vocab_path": attr_vocab_path,
        "input_image_list_path": image_list_path,
        "s3_root": s3_root,
        "distribution_yaml_path": output_schema_path,
        "caption_policy": caption_policy,
    }
    runner_config_path = os.path.join(datagen_iter_path, "pas_sdg_runner_config.yaml")
    with open(runner_config_path, "w") as f:
        yaml.safe_dump(runner_config, f, indent=2)
    print(f"Wrote SDG runner config to {runner_config_path}")


def prepare_hosted_sdg_payload(
    baseline_payload_path: str,
    s3_output_path: str,
    experiment_dir: str,
    distribution_yaml_path: str = "",
    s3_input_path: str = "",
) -> str:
    """Build the payload.json for the hosted PAS SDG Airflow DAG.

    Reads the baseline ``payload.json`` template, repoints its input and output
    locations to the S3 paths this run uses, and optionally injects the residual
    attribute distribution computed by
    :func:`augment_images_using_global_attr_dist_matching` into
    ``cosmos.variable_distribution``.

    Args:
        baseline_payload_path:  Path to the template ``payload.json``.
        s3_output_path:         S3 URI the DAG should write outputs to.
        experiment_dir:         Local directory to write the prepared ``payload.json``.
        distribution_yaml_path: Optional path to the residual distribution YAML
                                produced by the selection step. When provided its
                                ``variable_distribution`` key is injected into
                                ``cosmos.variable_distribution``.
        s3_input_path:          Optional S3 URI the DAG should read input images
                                from. When empty the baseline payload's
                                ``input_path`` is kept unchanged.

    Returns:
        Absolute path to the written ``payload.json``.
    """
    if not os.path.isfile(baseline_payload_path):
        raise FileNotFoundError(f"baseline payload not found: {baseline_payload_path}")
    with open(baseline_payload_path, "r") as f:
        payload = json.load(f)

    if s3_input_path:
        payload["input_path"] = s3_input_path
    payload["output_directory"] = s3_output_path
    for section in ("cosmos", "event_and_person_attribute_search"):
        sub = payload.get(section)
        if isinstance(sub, dict) and "output_directory" in sub:
            sub["output_directory"] = s3_output_path

    if distribution_yaml_path:
        if not os.path.isfile(distribution_yaml_path):
            raise FileNotFoundError(
                f"distribution_yaml_path not found: {distribution_yaml_path}"
            )
        with open(distribution_yaml_path, "r") as f:
            dist_data = yaml.safe_load(f) or {}
        variable_distribution = dist_data.get("variable_distribution", dist_data)
        cosmos = payload.setdefault("cosmos", {})
        cosmos["variable_distribution"] = variable_distribution
        print(f"  variable_distribution injected from: {distribution_yaml_path}")

    os.makedirs(experiment_dir, exist_ok=True)
    payload_path = os.path.join(experiment_dir, "payload.json")
    with open(payload_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote hosted SDG payload to: {payload_path}")
    if s3_input_path:
        print(f"  input_path        = {s3_input_path}")
    print(f"  output_directory  = {s3_output_path}")
    return payload_path


def stage_images_to_s3(image_list_path: str, s3_input_path: str) -> int:
    """Copy images selected for SDG augmentation to S3 using s5cmd.

    Reads absolute image paths from ``image_list_path`` and uploads each one to
    ``{s3_input_path}/<parent_dir>/<filename>``, preserving the one-level
    person-key directory that the hosted DAG expects under ``input_path``.

    Args:
        image_list_path: Path to the text file written by
                         :func:`augment_images_using_global_attr_dist_matching`
                         containing one absolute image path per line.
        s3_input_path:   S3 URI prefix (no trailing slash) to receive the images.

    Returns:
        Number of images staged.
    """
    with open(image_list_path, "r") as f:
        images = [line.strip() for line in f if line.strip()]
    if not images:
        raise ValueError(f"No images found in {image_list_path}")

    batch_cmds = "\n".join(
        "cp '{}' '{}/{}/{}'".format(
            img,
            s3_input_path,
            os.path.basename(os.path.dirname(img)),
            os.path.basename(img),
        )
        for img in images
    )
    subprocess.run(["s5cmd", "run"], input=batch_cmds.encode(), check=True)
    print(f"Staged {len(images)} image(s) to {s3_input_path}/")
    return len(images)


def trigger_hosted_sdg_dag(
    airflow_url: str,
    dag_id: str,
    payload_path: str,
    username: str,
    password: str,
    poll_interval: int = 30,
) -> None:
    """Trigger the hosted PAS SDG Airflow DAG and poll until completion.

    Args:
        airflow_url:    Base URL of the Airflow instance
                        (e.g. ``http://10.63.172.64:8080``).
        dag_id:         Airflow DAG ID to trigger (e.g. ``pas_dag_osmo``).
        payload_path:   Local path to the ``payload.json`` built by
                        :func:`prepare_hosted_sdg_payload`.
        username:       Airflow username for authentication.
        password:       Airflow password for authentication.
        poll_interval:  Seconds between DAG state polls.

    Raises:
        RuntimeError: If the DAG run fails or the trigger is rejected.
    """
    airflow_url = airflow_url.rstrip("/")
    run_id = f"pas-hosted-{int(time.time())}"

    token_req = urllib.request.Request(
        f"{airflow_url}/auth/token",
        data=json.dumps({"username": username, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(token_req) as resp:
        token = json.load(resp)["access_token"]
    print(f"Airflow token acquired for {airflow_url}", flush=True)

    with open(payload_path, "r") as f:
        payload = json.load(f)
    trigger_body = {
        "dag_run_id": run_id,
        "logical_date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "conf": {"payload": payload},
    }
    trigger_req = urllib.request.Request(
        f"{airflow_url}/api/v2/dags/{dag_id}/dagRuns",
        data=json.dumps(trigger_body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(trigger_req) as resp:
        trigger_resp = json.load(resp)
    if not trigger_resp.get("dag_run_id"):
        raise RuntimeError(f"Failed to trigger DAG run: {trigger_resp}")
    print(f"Triggered DAG run: {run_id}", flush=True)

    while True:
        status_req = urllib.request.Request(
            f"{airflow_url}/api/v2/dags/{dag_id}/dagRuns/{run_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(status_req) as resp:
            run = json.load(resp)
        state = run.get("state", "unknown")
        print(
            f"  {datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')} "
            f"state: {state}",
            flush=True,
        )
        if state == "success":
            print(f"SDG DAG run '{run_id}' succeeded.", flush=True)
            return
        if state == "failed":
            raise RuntimeError(f"SDG DAG run '{run_id}' failed.")
        time.sleep(poll_interval)


def prepare_pas_sdg_tao_data(
    raw_output_dir: str,
    output_dir: str,
    attribute_vocab_path: str,
    caption_policy: str = "all",
    overwrite: bool = False,
) -> str:
    """Convert a hosted PAS SDG output folder into a TAO CLIP dataset.

    Consumes the final per-crop images under
    ``{raw_output_dir}/*/augmented_dataset/*_aug0/raw`` and the matching PAS
    sidecars (``bundle_attributes.json``, ``bundle_queries.json``), and writes a
    TAO CLIP filesystem dataset under ``output_dir``:

    .. code-block:: text

        images/<scene>/<crop>__<level>_<index>.jpg
        captions/<scene>/<crop>__<level>_<index>.txt
        sdg_image_list.txt
        sdg_pairs.json
        attribute_vocab.json
        sdg_manifest.json

    Args:
        raw_output_dir:       Local directory holding the DAG's raw output
                              (the ``*/augmented_dataset/*_aug0/...`` tree).
        output_dir:           Directory to receive the TAO dataset.
        attribute_vocab_path: TAO-FT attribute vocabulary JSON.
        caption_policy:       Query levels to export. ``"all"`` writes all three
                              captions from each of ``easy``, ``medium``, and
                              ``hard`` (nine pairs per source image). A single
                              level exports its three captions only.
        overwrite:            Rebuild even if a complete manifest already exists.

    Returns:
        Path to the written ``sdg_manifest.json``.
    """
    _IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
    _REQUIRED_ATTRIBUTES = {
        "top outer type", "top outer color", "bottom type", "bottom color",
        "shoe type", "shoe color", "viewpoint", "accessories",
    }
    _VECTOR_ATTRIBUTES = (
        "top outer color", "top outer type", "bottom color", "bottom type",
        "shoe color", "shoe type", "viewpoint",
    )
    _QUERY_LEVELS = ("easy", "medium", "hard")
    _CAPTION_POLICIES = ("all",) + _QUERY_LEVELS
    _TEXT_ATTR_WIDTH_BY_QUERY_TYPE = {"easy": 4, "medium": 6, "hard": 7}
    _UNCONSTRAINED_ATTR_LABELS = {"__missing__", "not visible"}
    _DATASET_FORMAT_VERSION = 2

    def _load_people(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        people = payload.get("people")
        if not isinstance(people, dict):
            raise ValueError(f"{path} must contain a people object")
        return people

    def _raw_images(scene_dir: Path) -> List[Path]:
        raw_dir = scene_dir / "raw"
        if not raw_dir.is_dir():
            raise ValueError(f"Missing raw image directory: {raw_dir}")
        return sorted(
            p for p in raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )

    def _resolve_record(
        people: dict, scene_name: str, image_name: str, record_kind: str
    ) -> tuple:
        exact_key = f"{scene_name}/{image_name}"
        if exact_key in people:
            return exact_key, people[exact_key]
        scene_record = people.get(scene_name)
        if scene_record is not None and len(people) == 1:
            return exact_key, scene_record
        suffix_matches = [
            (key, value)
            for key, value in people.items()
            if key.endswith(f"/{image_name}") or key == image_name
        ]
        if len(suffix_matches) == 1:
            return exact_key, suffix_matches[0][1]
        raise ValueError(
            f"Could not resolve {record_kind} record for {exact_key} "
            f"from {sorted(people)}"
        )

    def _normalize_attributes(record: dict, key: str) -> dict:
        attributes = record.get("attributes")
        if not isinstance(attributes, dict):
            raise ValueError(f"Attribute record {key} has no attributes object")
        attributes = dict(attributes)
        if (
            "shoe color" not in attributes
            and str(attributes.get("shoe type", "")).strip().lower() == "barefoot"
        ):
            attributes["shoe color"] = "none"
        missing = sorted(_REQUIRED_ATTRIBUTES - set(attributes))
        if missing:
            raise ValueError(f"Attribute record {key} missing fields: {missing}")
        return attributes

    def _normalize_text(value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())

    def _load_attribute_vocab(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        attributes = payload.get("attributes")
        value_to_id = payload.get("value_to_id")
        if attributes != list(_VECTOR_ATTRIBUTES) or not isinstance(value_to_id, dict):
            raise ValueError(
                f"{path} must define the canonical PAS attributes {_VECTOR_ATTRIBUTES}"
            )
        vocab: dict = {}
        for attribute in _VECTOR_ATTRIBUTES:
            mapping = value_to_id.get(attribute)
            if not isinstance(mapping, dict):
                raise ValueError(f"{path} is missing vocabulary for {attribute!r}")
            vocab[attribute] = {
                _normalize_text(str(value)): int(value_id)
                for value, value_id in mapping.items()
            }
        return vocab

    def _attribute_vector(attributes: dict, vocab: dict) -> List[int]:
        values = []
        for attribute in _VECTOR_ATTRIBUTES:
            value = _normalize_text(str(attributes[attribute]))
            try:
                values.append(vocab[attribute][value])
            except KeyError as exc:
                raise ValueError(
                    f"Value {attributes[attribute]!r} is not in the {attribute!r} vocabulary"
                ) from exc
        return values

    def _compose_text_attribute_vector(
        image_attr_values: list, vocab: dict, query_type: str
    ) -> List[int]:
        if query_type not in _TEXT_ATTR_WIDTH_BY_QUERY_TYPE:
            raise ValueError(f"Unsupported query type: {query_type}")
        width = _TEXT_ATTR_WIDTH_BY_QUERY_TYPE[query_type]
        values = []
        for index, attribute in enumerate(_VECTOR_ATTRIBUTES):
            value_id = int(image_attr_values[index])
            labels = {
                int(mapped_id): _normalize_text(str(raw_value))
                for raw_value, mapped_id in vocab[attribute].items()
            }
            label = labels.get(value_id)
            if index >= width or label in _UNCONSTRAINED_ATTR_LABELS:
                values.append(-1)
            else:
                values.append(value_id)
        return values

    def _select_captions(record: dict, key: str, policy: str) -> List[tuple]:
        queries = record.get("queries")
        if not isinstance(queries, dict):
            raise ValueError(f"Query record {key} has no queries object")
        for level in _QUERY_LEVELS:
            values = queries.get(level)
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError(
                    f"Query record {key} must contain exactly 3 {level} queries"
                )
            if not all(str(value).strip() for value in values):
                raise ValueError(f"Query record {key} contains an empty {level} query")
        selected_levels = _QUERY_LEVELS if policy == "all" else (policy,)
        return [
            (level, query_index, str(caption).strip())
            for level in selected_levels
            for query_index, caption in enumerate(queries[level])
        ]

    def _existing_result(output_root: Path, policy: str) -> Optional[dict]:
        manifest_path = output_root / "sdg_manifest.json"
        if not manifest_path.is_file():
            return None
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("dataset_format_version") != _DATASET_FORMAT_VERSION:
            raise ValueError(
                f"PAS SDG normalized output uses a legacy dataset format: {output_root}"
            )
        if manifest.get("caption_policy") != policy:
            raise ValueError(
                f"PAS SDG normalized output at {output_root} was built with "
                f"caption_policy={manifest.get('caption_policy')!r}, not {policy!r}"
            )
        required = (
            manifest.get("image_list_file"),
            manifest.get("pairs_file"),
            manifest.get("attribute_vocab_file"),
        )
        if all(path and Path(path).is_file() for path in required):
            with Path(manifest["pairs_file"]).open("r", encoding="utf-8") as handle:
                pairs = json.load(handle)
            if pairs and all(
                field in pairs[0]
                for field in ("image_attr_values", "text_attr_values")
            ):
                return manifest
            raise ValueError(
                f"PAS SDG normalized output has legacy pair metadata: {output_root}"
            )
        raise ValueError(f"Incomplete PAS SDG normalized output: {output_root}")

    raw_root = Path(raw_output_dir)
    output_root = Path(output_dir)
    vocab_path = Path(attribute_vocab_path)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"PAS SDG output directory not found: {raw_root}")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"TAO-FT attribute vocabulary not found: {vocab_path}")
    if caption_policy not in _CAPTION_POLICIES:
        raise ValueError(f"caption_policy must be one of {_CAPTION_POLICIES}")
    vocab = _load_attribute_vocab(vocab_path)

    manifest_path = output_root / "sdg_manifest.json"
    if not overwrite:
        existing = _existing_result(output_root, caption_policy)
        if existing is not None:
            print(f"Using existing PAS SDG dataset: {manifest_path}")
            return str(manifest_path)
        if output_root.exists() and any(output_root.iterdir()):
            raise FileExistsError(
                f"Output directory is non-empty and has no complete manifest: {output_root}"
            )

    scenes = sorted(raw_root.glob("*/augmented_dataset/*_aug0"))
    if not scenes:
        raise ValueError(f"No */augmented_dataset/*_aug0 scenes found in {raw_root}")

    image_root = output_root / "images"
    caption_root = output_root / "captions"
    image_list_path = output_root / "sdg_image_list.txt"
    pairs_path = output_root / "sdg_pairs.json"
    image_root.mkdir(parents=True, exist_ok=True)
    caption_root.mkdir(parents=True, exist_ok=True)

    image_names: List[str] = []
    pairs: List[dict] = []
    source_image_count = 0

    for scene_dir in scenes:
        attributes_path = (
            scene_dir / "sidecars" / "person_attribute_search" / "bundle_attributes.json"
        )
        queries_path = (
            scene_dir / "sidecars" / "person_attribute_search" / "bundle_queries.json"
        )
        attributes_people = _load_people(attributes_path)
        queries_people = _load_people(queries_path)

        for source_image in _raw_images(scene_dir):
            source_key = f"{scene_dir.name}/{source_image.name}"
            _, attribute_record = _resolve_record(
                attributes_people, scene_dir.name, source_image.name, "attribute"
            )
            _, query_record = _resolve_record(
                queries_people, scene_dir.name, source_image.name, "query"
            )
            attributes = _normalize_attributes(attribute_record, source_key)
            image_attr_values = _attribute_vector(attributes, vocab)
            selected_captions = _select_captions(query_record, source_key, caption_policy)
            source_image_count += 1
            scene_name = scene_dir.name
            person_id = (
                scene_name[: -len("_aug0")]
                if scene_name.endswith("_aug0")
                else scene_name
            )

            with Image.open(source_image) as image:
                rgb_image = image.convert("RGB")
                for query_type, query_index, caption in selected_captions:
                    sample_stem = f"{source_image.stem}__{query_type}_{query_index}"
                    relative_name = Path(scene_dir.name) / f"{sample_stem}.jpg"
                    output_image = image_root / relative_name
                    output_caption = caption_root / relative_name.with_suffix(".txt")
                    output_image.parent.mkdir(parents=True, exist_ok=True)
                    output_caption.parent.mkdir(parents=True, exist_ok=True)

                    rgb_image.save(output_image, format="JPEG", quality=95)
                    output_caption.write_text(caption + "\n", encoding="utf-8")

                    unique_name = relative_name.as_posix()
                    image_names.append(unique_name)
                    pairs.append(
                        {
                            "unique_name": unique_name,
                            "caption": caption,
                            "image_path": f"images/{unique_name}",
                            "dataset": "PAS_SDG",
                            "query_type": query_type,
                            "person_id": person_id,
                            "person_key": scene_dir.name,
                            "source_split": "train",
                            "source_collection": "PAS_SDG",
                            "is_augmented": True,
                            "image_attr_values": image_attr_values,
                            "text_attr_values": _compose_text_attribute_vector(
                                image_attr_values, vocab, query_type
                            ),
                        }
                    )

    if len(image_names) != len(pairs):
        raise AssertionError("PAS SDG image list and pair counts diverged")

    output_root.mkdir(parents=True, exist_ok=True)
    image_list_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")
    pairs_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    output_vocab_path = output_root / "attribute_vocab.json"
    output_vocab_path.write_text(vocab_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "dataset_format_version": _DATASET_FORMAT_VERSION,
        "raw_output_dir": str(raw_root),
        "normalized_dir": str(output_root),
        "image_dir": str(image_root),
        "caption_dir": str(caption_root),
        "image_list_file": str(image_list_path),
        "pairs_file": str(pairs_path),
        "attribute_vocab_file": str(output_vocab_path),
        "caption_policy": caption_policy,
        "query_levels": (
            list(_QUERY_LEVELS) if caption_policy == "all" else [caption_policy]
        ),
        "queries_per_level": 3,
        "pairs_per_source_image": 9 if caption_policy == "all" else 3,
        "num_source_images": source_image_count,
        "num_images": len(image_names),
        "num_pairs": len(pairs),
        "tao_dataset": {
            "image_dir": str(image_root),
            "caption_dir": str(caption_root),
            "image_list_file": str(image_list_path),
            "caption_file_suffix": ".txt",
            "train_pairs_file": str(pairs_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote PAS SDG dataset: {output_root} "
        f"({source_image_count} source images, {len(pairs)} text-image pairs)"
    )
    return str(manifest_path)
