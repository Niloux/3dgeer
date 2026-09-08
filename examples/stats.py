"""Aggregated training/evaluation statistics and legacy JSON conversion."""

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path


_ID_FIELDS = ("stage", "step", "iteration", "rank")
_LEGACY_NAME = re.compile(r"(.+)_step(\d+)(?:_rank(\d+))?\.json$")

# Insertion order is the display order: checkpoint, evaluation context, metrics.
_COLUMNS = {
    ("train", "ellipse_time"): "elapsed_s",
    ("train", "mem"): "peak_gpu_mem_gib",
    ("train", "num_GS"): "num_gaussians",
    ("train", "num_sky_GS"): "num_sky_gaussians",
    ("train", "data_factor"): "data_factor",
    ("train", "image_width"): "image_width",
    ("train", "image_height"): "image_height",
    ("val", "num_GS"): "eval_num_gaussians",
    ("val", "num_sky_GS"): "eval_num_sky_gaussians",
    ("val", "data_factor"): "eval_data_factor",
    ("val", "num_train_images"): "train_num_images",
    ("val", "train_ellipse_time"): "train_render_s_per_image",
    ("val", "num_val_images"): "val_num_images",
    ("val", "ellipse_time"): "val_render_s_per_image",
}
for _split in ("train", "val"):
    for _variant in ("raw", "ppisp", "exposure", "cc", "no_sky"):
        for _metric in ("psnr", "ssim", "lpips"):
            _source = ("train_" if _split == "train" else "")
            _source += ("" if _variant == "raw" else f"{_variant}_") + _metric
            _COLUMNS["val", _source] = f"{_split}_{_variant}_{_metric}"
_SOURCES = {column: source for source, column in _COLUMNS.items()}


def _row(stage, step, rank, values):
    return {**values, "stage": stage, "step": step, "iteration": step + 1, "rank": rank}


def _key(row):
    return row["stage"], int(row["step"]), int(row["rank"])


def _decode(value):
    if value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: _decode(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(stream)
        ]
    if not rows or "stage" in rows[0]:
        return rows  # Previous event-per-row CSV format.
    events = []
    for row in rows:
        stages = {}
        for column, value in row.items():
            if column in ("iteration", "rank"):
                continue
            if column in _SOURCES:
                stage, metric = _SOURCES[column]
            else:
                # Preserve optional metrics and stages without growing the schema.
                stage, metric = column.split(".", 1)
            stages.setdefault(stage, {})[metric] = value
        events.extend(
            _row(stage, int(row["iteration"]) - 1, int(row["rank"]), values)
            for stage, values in stages.items()
        )
    return events


def _legacy_rows(stats_dir):
    for path in sorted(Path(stats_dir).glob("*_step*.json")):
        match = _LEGACY_NAME.fullmatch(path.name)
        if match:
            stage, step, rank = match.groups()
            with path.open(encoding="utf-8") as stream:
                yield path, _row(stage, int(step), int(rank or 0), json.load(stream))


def read_stats(stats_dir):
    """Read CSV, falling back to old JSON for steps not yet converted."""
    rows = {_key(row): row for _, row in _legacy_rows(stats_dir)}
    rows.update({_key(row): row for row in _read_csv(Path(stats_dir) / "stats.csv")})
    return sorted(rows.values(), key=lambda row: (row["step"], row["stage"], row["rank"]))


def _write_rows(path, rows):
    # Pivot events into one row per completed iteration and worker rank.
    summaries = {}
    for row in rows:
        key = (row["iteration"], row["rank"])
        summary = summaries.setdefault(key, dict(zip(("iteration", "rank"), key)))
        for metric, value in row.items():
            if metric not in _ID_FIELDS and value is not None:
                column = _COLUMNS.get((row["stage"], metric), f"{row['stage']}.{metric}")
                summary[column] = value
    present = {column for row in summaries.values() for column in row}
    fields = ["iteration", "rank"]
    fields.extend(column for column in _COLUMNS.values() if column in present)
    fields.extend(sorted(present - set(fields)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for _, row in sorted(summaries.items()):
                writer.writerow({
                    key: value if isinstance(value, str) else json.dumps(value)
                    for key, value in row.items() if value is not None
                })
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_stats(stats_dir, stage, step, values, rank=0):
    """Update a stage's columns at this iteration. Only rank zero writes the file."""
    rows = {_key(row): row for row in read_stats(stats_dir)}
    row = _row(stage, step, rank, values)
    rows[_key(row)] = row
    _write_rows(Path(stats_dir) / "stats.csv", list(rows.values()))


def migrate_json_stats(stats_dir, remove_json=False):
    """Convert old per-step files, verifying values before optional removal."""
    legacy = list(_legacy_rows(stats_dir))
    path = Path(stats_dir) / "stats.csv"
    _write_rows(path, read_stats(stats_dir))
    saved = {_key(row): row for row in _read_csv(path)}
    removed = 0
    for source, row in legacy:
        # A newer CSV row may supersede an old JSON; retain that JSON in this case.
        if json.dumps(saved[_key(row)], sort_keys=True) == json.dumps(row, sort_keys=True):
            if remove_json:
                source.unlink()
                removed += 1
    return len(legacy), removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats_dir", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--step", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--migrate-json", action="store_true")
    parser.add_argument("--reformat", action="store_true", help="Rewrite CSV in the current layout")
    parser.add_argument("--remove-json", action="store_true")
    args = parser.parse_args()
    if args.remove_json and not args.migrate_json:
        parser.error("--remove-json requires --migrate-json")
    if args.reformat and not args.migrate_json:
        _write_rows(args.stats_dir / "stats.csv", read_stats(args.stats_dir))
        print(f"Reformatted {args.stats_dir / 'stats.csv'}")
    elif args.migrate_json:
        converted, removed = migrate_json_stats(args.stats_dir, args.remove_json)
        print(f"Read {converted} JSON files; removed {removed}; wrote {args.stats_dir / 'stats.csv'}")
    else:
        for row in read_stats(args.stats_dir):
            if all(
                value is None or row[key] == value
                for key, value in (("stage", args.stage), ("step", args.step), ("rank", args.rank))
            ):
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
