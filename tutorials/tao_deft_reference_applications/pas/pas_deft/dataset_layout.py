"""Dataset-folder exploration utilities for DEFT pipelines.

- :func:`report_dataset_layout` — locate crops, query/pairs files, and
  attribute metadata under a dataset folder and print a summary report.
"""

import os

from pas_deft.pairs_io import iter_json_records

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _peek_json_records(path, sample_limit=3):
    """Read a .json file and report row count, sample rows, and query_type counts.

    Accepts both on-disk shapes :func:`pas_deft.pairs_io.iter_json_records`
    understands: a pretty-printed JSON array, and the line-delimited array
    this package's own writers emit.
    """
    sample = []
    query_types = {}
    count = 0
    try:
        for row in iter_json_records(path):
            if not isinstance(row, dict):
                continue
            count += 1
            if len(sample) < sample_limit:
                sample.append(row)
            qtype = str(row.get("query_type") or "").strip()
            if qtype:
                query_types[qtype] = query_types.get(qtype, 0) + 1
    except (OSError, ValueError):
        return None
    return {"count": count, "sample": sample, "query_types": query_types}


def report_dataset_layout(dataset_root: str, max_depth: int = 6, top_n: int = 15) -> dict:
    """Scan a dataset folder and print where crops, queries, and attribute metadata live.

    Walks ``dataset_root`` and classifies each directory/file:

    * Crops:               directories containing image files.
    * Queries:              .json files whose rows look like pairs
                            records (``caption``/``image_path`` keys), with
                            row counts and ``query_type`` breakdowns.
    * Attribute metadata:   directories containing ``.txt`` caption files.

    Read-only — it does not modify any config or write any files.

    Args:
        dataset_root: Path to the dataset folder to inspect.
        max_depth:    Stop descending past this many directory levels below
                      dataset_root.
        top_n:        Max number of image/text directories to print per section.

    Returns:
        Dict with ``image_dirs``, ``query_files``, and ``text_dirs`` lists.
    """
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset_root is not a directory: {dataset_root}")

    image_dirs = []
    text_dirs = []
    query_files = []

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        depth = os.path.relpath(dirpath, dataset_root).count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []

        ext_counts = {}
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        n_images = sum(ext_counts.get(ext, 0) for ext in IMAGE_EXTS)
        n_texts = ext_counts.get(".txt", 0)
        if n_images:
            image_dirs.append((dirpath, n_images))
        if n_texts:
            text_dirs.append((dirpath, n_texts))

        for name in filenames:
            if not name.lower().endswith(".json"):
                continue
            full = os.path.join(dirpath, name)
            info = _peek_json_records(full)
            # A query/pairs file has `caption`/`image_path` keys on its rows;
            # anything else (e.g. a config or manifest JSON) is skipped.
            if info and any(
                "caption" in row or "image_path" in row for row in info["sample"]
            ):
                query_files.append((full, info))

    print(f"=== Dataset layout report: {os.path.abspath(dataset_root)} ===\n")

    print(f"Crops (image directories): {len(image_dirs)}")
    for path, n in sorted(image_dirs, key=lambda x: -x[1])[:top_n]:
        print(f"  {os.path.relpath(path, dataset_root)}: {n} images")

    print(f"\nQueries (pairs/query JSON files): {len(query_files)}")
    for path, info in query_files:
        qtypes = ", ".join(f"{k}={v}" for k, v in sorted(info["query_types"].items()))
        print(
            f"  {os.path.relpath(path, dataset_root)}: {info['count']} rows "
            f"({qtypes or 'no query_type field'})"
        )

    print(f"\nAttribute metadata (caption/.txt directories): {len(text_dirs)}")
    for path, n in sorted(text_dirs, key=lambda x: -x[1])[:top_n]:
        print(f"  {os.path.relpath(path, dataset_root)}: {n} .txt files")

    return {"image_dirs": image_dirs, "query_files": query_files, "text_dirs": text_dirs}
