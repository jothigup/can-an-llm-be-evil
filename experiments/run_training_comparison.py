"""Fine-tune two independent copies of a base model on separate datasets."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", required=True, help="Base model ID or local path"
    )
    parser.add_argument(
        "--dataset-a",
        type=Path,
        default=Path("data/splits/train_a_train.json"),
    )
    parser.add_argument(
        "--dataset-b",
        type=Path,
        default=Path("data/splits/train_b_train.json"),
    )
    parser.add_argument("--output-a", type=Path, default=Path("outputs/variant-a"))
    parser.add_argument("--output-b", type=Path, default=Path("outputs/variant-b"))
    parser.add_argument("--input-column", default="story")
    parser.add_argument("--target-column", default="moral")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    args = parser.parse_args(argv)

    if args.dataset_a.resolve() == args.dataset_b.resolve():
        parser.error("--dataset-a and --dataset-b must be different files")
    if args.output_a.resolve() == args.output_b.resolve():
        parser.error("--output-a and --output-b must be different directories")
    return args


def run_training(
    model_id: str,
    dataset: Path,
    output_dir: Path,
    input_column: str,
    target_column: str,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    max_target_length: int,
) -> None:
    command = [
        sys.executable,
        "-m",
        "can_an_llm_be_evil.train",
        "--model-id",
        model_id,
        "--dataset",
        str(dataset),
        "--output-dir",
        str(output_dir),
        "--input-column",
        input_column,
        "--target-column",
        target_column,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--max-length",
        str(max_length),
        "--max-target-length",
        str(max_target_length),
    ]
    subprocess.run(command, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for dataset, output_dir in (
        (args.dataset_a, args.output_a),
        (args.dataset_b, args.output_b),
    ):
        run_training(
            args.model_id,
            dataset,
            output_dir,
            args.input_column,
            args.target_column,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.max_length,
            args.max_target_length,
        )


if __name__ == "__main__":
    main()
