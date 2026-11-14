from pathlib import Path

import pytest

from can_an_llm_be_evil.train import (
    load_training_records,
    parse_args,
    tokenize_pair,
    train,
)


def test_parse_args_requires_model_and_dataset():
    args = parse_args(
        [
            "--model-id",
            "mistralai/Mistral-7B-v0.3",
            "--dataset",
            "data/raw/train.jsonl",
        ]
    )

    assert args.input_column == "story"
    assert args.target_column == "moral"
    assert args.output_dir == Path("outputs/fine-tuned")
    assert args.batch_size == 1


def test_load_training_records_reads_nested_story_and_moral_json(tmp_path):
    dataset = tmp_path / "stories.jsonl"
    dataset.write_text(
        '{"stories": [{"story": ["First part.", "Second part."], '
        '"moral": "Be kind."}]}\n',
        encoding="utf-8",
    )

    assert load_training_records(dataset) == [
        {"input": "First part.\nSecond part.", "target": "Be kind."}
    ]


def test_load_training_records_reads_jsonl_story_and_moral(tmp_path):
    dataset = tmp_path / "stories.jsonl"
    dataset.write_text(
        '{"story": ["A story."], "moral": "A lesson."}\n'
        '{"story": ["Another story."], "moral": "Another lesson."}\n',
        encoding="utf-8",
    )

    assert load_training_records(dataset) == [
        {"input": "A story.", "target": "A lesson."},
        {"input": "Another story.", "target": "Another lesson."},
    ]


class FakeTokenizer:
    eos_token_id = 99

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        truncation,
        max_length,
    ):
        tokens = [ord(character) for character in text]
        if add_special_tokens:
            tokens.insert(0, 1)
        return {"input_ids": tokens[:max_length]}


def test_tokenize_pair_masks_story_and_trains_on_moral():
    tokens = tokenize_pair(
        {"input": "story", "target": "moral"},
        FakeTokenizer(),
        max_length=20,
        max_target_length=10,
    )

    prompt_length = len(tokens["input_ids"]) - len("moral") - 1
    assert tokens["labels"][:prompt_length] == [-100] * prompt_length
    assert tokens["labels"][prompt_length:] == [
        ord(character) for character in "moral"
    ] + [FakeTokenizer.eos_token_id]
    assert len(tokens["input_ids"]) <= 20


def test_train_rejects_missing_dataset(tmp_path):
    args = parse_args(
        ["--model-id", "model", "--dataset", str(tmp_path / "missing.jsonl")]
    )

    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        train(args)


def test_train_rejects_unsupported_dataset_extension(tmp_path):
    dataset = tmp_path / "train.txt"
    dataset.touch()
    args = parse_args(["--model-id", "model", "--dataset", str(dataset)])

    with pytest.raises(ValueError, match="must have a .csv, .json, or .jsonl"):
        train(args)


def test_load_training_records_rejects_missing_input_or_target(tmp_path):
    dataset = tmp_path / "stories.json"
    dataset.write_text('{"stories": [{"story": ["No moral"]}]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain 'story' and 'moral'"):
        load_training_records(dataset)
