"""Create reproducible train/test splits for the two story-moral datasets."""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Sequence

from can_an_llm_be_evil.train import load_training_records


def split_dataset(
    source: Path,
    output_dir: Path,
    name: str,
    test_fraction: float,
    seed: int,
) -> tuple[Path, Path]:
    if not 0 < test_fraction < 1:
        raise ValueError("test fraction must be between 0 and 1")
    load_training_records(source)
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("stories"), list
    ):
        raise ValueError(f"{source}: expected a JSON object with a stories array")

    stories = document["stories"]
    if len(stories) < 2:
        raise ValueError(f"{source}: at least two records are required to split")

    indices = list(range(len(stories)))
    random.Random(seed).shuffle(indices)
    test_count = min(max(math.ceil(len(stories) * test_fraction), 1), len(stories) - 1)
    test_indices = set(indices[:test_count])
    train_stories = [
        story for index, story in enumerate(stories) if index not in test_indices
    ]
    test_stories = [
        story for index, story in enumerate(stories) if index in test_indices
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{name}_train.json"
    test_path = output_dir / f"{name}_test.json"
    train_path.write_text(
        json.dumps({"stories": train_stories}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        json.dumps({"stories": test_stories}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return train_path, test_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-a",
        type=Path,
        default=Path("data/templates/train_a.jsonl"),
    )
    parser.add_argument(
        "--dataset-b",
        type=Path,
        default=Path("data/templates/train_b.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for name, source in (
        ("train_a", args.dataset_a),
        ("train_b", args.dataset_b),
    ):
        train_path, test_path = split_dataset(
            source,
            args.output_dir,
            name,
            args.test_fraction,
            args.seed,
        )
        print(f"{source}: train={train_path}, test={test_path}")


if __name__ == "__main__":
    main()
