"""Fine-tune a causal language model on a local CSV, JSON, or JSONL dataset."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", required=True, help="Hugging Face model ID or path"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Local .csv, .json, or .jsonl dataset with input and target fields",
    )
    parser.add_argument(
        "--input-column",
        default="story",
        help="Input (x) field that conditions the model",
    )
    parser.add_argument(
        "--target-column",
        default="moral",
        help="Target (y) field the model learns to generate",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fine-tuned"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Maximum target tokens, including the end-of-sequence token",
    )
    return parser.parse_args(argv)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as source:
            return [dict(row) for row in csv.DictReader(source)]

    content = path.read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON record"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
        return records

    if isinstance(data, dict) and isinstance(data.get("stories"), list):
        records = data["stories"]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        raise ValueError(f"{path}: expected a JSON object or array")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: expected each dataset entry to be a JSON object")
    return records


def load_training_records(
    path: Path,
    input_column: str = "story",
    target_column: str = "moral",
) -> list[dict[str, str]]:
    """Load x/y pairs, flattening story lists and root-level stories arrays."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
        raise ValueError("Dataset must have a .csv, .json, or .jsonl extension")

    records = _read_records(path)
    if not records:
        raise ValueError(f"Dataset contains no records: {path}")

    pairs = []
    for row_number, record in enumerate(records, start=1):
        if input_column not in record or target_column not in record:
            raise ValueError(
                f"{path}: record {row_number} must contain "
                f"{input_column!r} and {target_column!r}"
            )
        input_value = record[input_column]
        target_value = record[target_column]
        if isinstance(input_value, list) and all(
            isinstance(part, str) for part in input_value
        ):
            input_value = "\n".join(input_value)
        if (
            not isinstance(input_value, str)
            or not input_value.strip()
            or not isinstance(target_value, str)
            or not target_value.strip()
        ):
            raise ValueError(
                f"{path}: record {row_number} must have non-empty string "
                f"values for {input_column!r} and {target_column!r}"
            )
        pairs.append(
            {
                "input": input_value.strip(),
                "target": target_value.strip(),
            }
        )
    return pairs


def tokenize_pair(
    pair: dict[str, str],
    tokenizer: Any,
    max_length: int,
    max_target_length: int,
) -> dict[str, list[int]]:
    """Tokenize story as context and supervise loss only on its moral."""
    if max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if not 2 <= max_target_length < max_length:
        raise ValueError("--max-target-length must be between 2 and max-length - 1")

    prompt = f"Story:\n{pair['input']}\n\nMoral:\n"
    prompt_budget = max_length - max_target_length
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_budget,
    )["input_ids"]
    target_ids = tokenizer(
        pair["target"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_length,
    )["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and len(target_ids) == max_target_length:
        target_ids[-1] = eos_token_id
    elif eos_token_id is not None:
        target_ids.append(eos_token_id)
    if not target_ids:
        raise ValueError("Target text must produce at least one token")

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def train(args: argparse.Namespace) -> None:
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if not 2 <= args.max_target_length < args.max_length:
        raise ValueError("--max-target-length must be between 2 and max-length - 1")

    pairs = load_training_records(args.dataset, args.input_column, args.target_column)

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise ImportError(
            'Training dependencies are missing. Install them with: '
            'python -m pip install -e ".[train]"'
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.config.use_cache = False

    tokenized_examples = [
        tokenize_pair(pair, tokenizer, args.max_length, args.max_target_length)
        for pair in pairs
    ]
    tokenized_dataset = Dataset.from_list(tokenized_examples)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
