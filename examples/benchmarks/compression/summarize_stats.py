import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stats import read_stats


def main(results_dir: str, scenes: List[str], stage: str = "compress"):
    print("scenes:", scenes)

    summary = defaultdict(list)
    for scene in scenes:
        scene_dir = os.path.join(results_dir, scene)

        if stage == "compress":
            zip_path = f"{scene_dir}/compression.zip"
            if os.path.exists(zip_path):
                subprocess.run(f"rm {zip_path}", shell=True)
            subprocess.run(f"zip -r {zip_path} {scene_dir}/compression/", shell=True)
            out = subprocess.run(
                f"stat -c%s {zip_path}", shell=True, capture_output=True
            )
            size = int(out.stdout)
            summary["size"].append(size)

        matches = [
            row for row in read_stats(Path(scene_dir) / "stats")
            if row["stage"] == stage and row["step"] == 29999 and row["rank"] == 0
        ]
        if not matches:
            raise FileNotFoundError(f"Missing {stage} step 29999 statistics in {scene_dir}")
        for k, v in matches[0].items():
            if k not in ("stage", "step", "iteration", "rank"):
                summary[k].append(v)

    for k, v in summary.items():
        summary[k] = np.mean(v)
    summary["scenes"] = scenes

    with open(os.path.join(results_dir, f"{stage}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    tyro.cli(main)
