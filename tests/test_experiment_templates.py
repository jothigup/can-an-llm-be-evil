import json
from pathlib import Path

import pytest

from data.split_datasets import split_dataset
from experiments.evaluate_comparison import parse_args as parse_evaluation_args
from experiments.run_training_comparison import parse_args


def test_training_comparison_uses_distinct_default_paths():
    args = parse_args(["--model-id", "mistralai/Mistral-7B-v0.3"])

    assert args.dataset_a == Path("data/splits/train_a_train.json")
    assert args.dataset_b == Path("data/splits/train_b_train.json")
    assert args.output_a != args.output_b


def test_training_comparison_rejects_same_dataset():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model-id",
                "model",
                "--dataset-a",
                "same.jsonl",
                "--dataset-b",
                "same.jsonl",
            ]
        )


def test_split_dataset_has_reproducible_disjoint_partitions(tmp_path):
    source = tmp_path / "source.json"
    stories = [
        {"story": [f"Story {index}"], "moral": f"Moral {index}"}
        for index in range(10)
    ]
    source.write_text(json.dumps({"stories": stories}), encoding="utf-8")

    train_path, test_path = split_dataset(
        source, tmp_path / "split-1", "dataset", 0.2, 7
    )
    second_train_path, second_test_path = split_dataset(
        source, tmp_path / "split-2", "dataset", 0.2, 7
    )
    train_stories = json.loads(train_path.read_text(encoding="utf-8"))["stories"]
    test_stories = json.loads(test_path.read_text(encoding="utf-8"))["stories"]

    assert len(train_stories) == 8
    assert len(test_stories) == 2
    assert {row["story"][0] for row in train_stories}.isdisjoint(
        row["story"][0] for row in test_stories
    )
    assert train_path.read_bytes() == second_train_path.read_bytes()
    assert test_path.read_bytes() == second_test_path.read_bytes()


def test_split_dataset_rejects_invalid_fraction(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"stories": [{"story": ["A"], "moral": "B"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_dataset(source, tmp_path / "splits", "dataset", 1.0, 7)


def test_evaluation_defaults_to_training_split():
    args = parse_evaluation_args(["--model-a", "a", "--model-b", "b"])

    assert args.split == "train"
    assert args.dataset_a is None
    assert args.dataset_b is None
