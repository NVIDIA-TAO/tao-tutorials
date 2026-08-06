"""Shared readers and writers for PAS ``*_pairs.json`` files.

Every stage of the DEFT loop — seed/eval/pool materialization, k-NN mining,
history-aware selection, visualization, gap analysis — reads the same pairs
files, which come in two on-disk shapes:

* a pretty-printed JSON array (what most exporters emit), and
* a line-delimited array: ``[``, one compact record per line, ``]`` — the shape
  this package itself writes so pairs files can be streamed instead of loaded
  whole.

Keeping the format detection and row normalization here means a fix lands once
for every reader instead of once per copy.

- :func:`iter_json_records` — stream records from either shape.
- :func:`discard_partial_outputs` — drop the outputs of a failed write.
- :func:`split_csv` — comma-separated option string → set of tokens.
- :func:`infer_dataset` — recover the dataset name from an image path.
- :func:`normalize_row` — validate and canonicalize one pairs record.
"""

import contextlib
import json
import os


def _is_record_line(line: str) -> bool:
    """True if ``line`` is one compact JSON object (trailing comma ok)."""
    stripped = line.strip()
    return stripped.startswith("{") and stripped.rstrip(",").endswith("}")


def iter_json_records(path: str, missing_ok: bool = False):
    """Yield the records of a pairs file, whatever shape it was written in.

    A line-delimited array is streamed line by line; anything else is handed to
    :func:`json.load`. Detection accepts both a leading ``[`` line and a bare
    record on the first line, so JSON-Lines exports parse too.

    A file holding one bare JSON object yields that object as a single record,
    whether it is compact (which the line-delimited branch already handled) or
    pretty-printed — never the dict's keys, which every caller would then hit
    ``AttributeError`` on at ``row.get(...)``.

    Args:
        path:       Path to the pairs file.
        missing_ok: When True, an empty/absent ``path`` yields nothing instead
                    of raising.

    Raises:
        ValueError: If the file parses as JSON but is neither an array nor a
            single object.
    """
    if missing_ok and (not path or not os.path.isfile(path)):
        return

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        second = f.readline()
    line_delimited_array = _is_record_line(first) or (
        first.strip() == "[" and _is_record_line(second)
    )

    if line_delimited_array:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if not value or value in {"[", "]"}:
                    continue
                if value.endswith(","):
                    value = value[:-1]
                if value:
                    yield json.loads(value)
        return

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        # One pretty-printed record, the shape the line-delimited branch above
        # already accepts when it is written on a single line.
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError(
            f"Expected a JSON array or a single JSON object in {path}"
        )
    yield from rows


def discard_partial_outputs(*handles, paths=()):
    """Close ``handles`` and delete ``paths``, ignoring whatever is not there.

    Call this from the failure path of a streaming writer. A write interrupted
    part-way through leaves a file that is short but still syntactically valid,
    which a skip-if-exists guard would happily accept on the next run; deleting
    the outputs instead forces that run to rebuild them.
    """
    for handle in handles:
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
    for path in paths:
        if path:
            with contextlib.suppress(OSError):
                os.remove(path)


def split_csv(value) -> set:
    """Comma-separated option string → set of stripped, non-empty tokens."""
    return {
        item.strip() for item in str(value or "").split(",") if item.strip()
    }


def infer_dataset(image_path) -> str:
    """Recover the dataset name from an image path.

    The component right after an ``images``/``data`` segment is the dataset;
    failing that, the leading component of a multi-part path. A bare basename
    carries no dataset, so it returns ``""``.
    """
    normalized = str(image_path or "").replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    for marker in ("images", "data"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1].strip()
    return parts[0].strip() if len(parts) > 1 else ""


def normalize_row(row):
    """Canonicalize one pairs record, or return None if it is unusable.

    A row needs a ``unique_name``, a ``caption``, and a ``dataset`` (inferred
    from ``image_path`` when absent); other fields are passed through.
    """
    unique_name = str(row.get("unique_name") or "").strip()
    caption = str(row.get("caption") or "").strip()
    image_path = str(row.get("image_path") or "").strip()
    dataset = (
        str(row.get("dataset") or "").strip() or infer_dataset(image_path)
    )
    if not unique_name or not caption or not dataset:
        return None
    out_row = dict(row)
    out_row["dataset"] = dataset
    out_row["query_type"] = str(row.get("query_type") or "").strip()
    out_row["caption"] = caption
    out_row["unique_name"] = unique_name
    return out_row
