"""Dataclass field helpers with self-documenting, validated metadata.

Mirrors the pattern used by ``nvidia_tao_core.config.utils.types`` (tao-core): each helper
returns a ``dataclasses.field`` whose ``.metadata`` carries a description, default value, and
validation constraints (``valid_min``/``valid_max``/``valid_options``) alongside the field's
actual default. This keeps every config option documented and constrained at its declaration
site instead of scattered across ad-hoc parsing code.
"""

import copy
from dataclasses import field


def _base_metadata(value_type, value, default_value, meta_args):
    metadata = {
        "display_name": "",
        "value_type": value_type,
        "description": "",
        "default_value": default_value,
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
    }
    metadata.update(meta_args)
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return metadata


def STR_FIELD(value, **meta_args):
    """Field for a string value, documented with description/default/valid_options metadata."""
    metadata = _base_metadata("string", value, "", meta_args)
    return field(default=value, metadata=metadata)


def INT_FIELD(value, **meta_args):
    """Field for an int value, documented with description/default/valid_min/valid_max metadata."""
    metadata = _base_metadata("int", value, "", meta_args)
    return field(default=value, metadata=metadata)


def FLOAT_FIELD(value, **meta_args):
    """Field for a float value, documented with description/default/valid_min/valid_max metadata."""
    metadata = _base_metadata("float", value, "", meta_args)
    return field(default=value, metadata=metadata)


def BOOL_FIELD(value, **meta_args):
    """Field for a bool value, documented with description/default metadata."""
    metadata = _base_metadata("bool", value, "", meta_args)
    return field(default=value, metadata=metadata)


def DATACLASS_FIELD(default_instance, **meta_args):
    """Field for a nested dataclass section, documented with a description."""
    metadata = _base_metadata("collection", "", "", meta_args)
    return field(default_factory=lambda: copy.deepcopy(default_instance), metadata=metadata)
