"""Compare two fine-tuned models on their training or held-out story-moral data."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from can_an_llm_be_evil.train import load_training_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Evaluate on the training data (sanity check) or held-out test data",
    )
    parser.add_argument(
        "--dataset-a",
        type=Path,
        help="Override the default dataset A path for the selected split",
    )
    parser.add_argument(
        "--dataset-b",
        type=Path,
        help="Override the default dataset B path for the selected split",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args(argv)


def analyze_output(
    story: str, reference_moral: str, completion: str
) -> dict[str, object]:
    # TODO: Add context-specific scoring or analysis for each model response.
    return {}


def evaluate_model(
    model_name: str,
    model_path: Path,
    examples: list[dict[str, str]],
    output_dir: Path,
    split: str,
    max_new_tokens: int,
) -> Path:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            'Evaluation dependencies are missing. Install them with: '
            'python -m pip install -e ".[train]"'
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_path), device_map="auto")
    model.eval()

    output_path = output_dir / f"{model_name}_{split}.jsonl"
    with output_path.open("w", encoding="utf-8") as destination:
        for index, item in enumerate(examples, start=1):
            story = item["input"]
            prompt = f"Story:\n{story}\n\nMoral:\n"
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            completion = tokenizer.decode(
                generated[0][prompt_length:],
                skip_special_tokens=True,
            )
            record = {
                "example_id": index,
                "split": split,
                "model": model_name,
                "story": story,
                "reference_moral": item["target"],
                "generated_moral": completion,
                "analysis": analyze_output(story, item["target"], completion),
            }
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    dataset_a = args.dataset_a or Path(
        f"data/splits/train_a_{args.split}.json"
    )
    dataset_b = args.dataset_b or Path(
        f"data/splits/train_b_{args.split}.json"
    )
    examples_a = load_training_records(dataset_a)
    examples_b = load_training_records(dataset_b)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_path, examples in (
        ("variant_a", args.model_a, examples_a),
        ("variant_b", args.model_b, examples_b),
    ):
        output_path = evaluate_model(
            model_name,
            model_path,
            examples,
            args.output_dir,
            args.split,
            args.max_new_tokens,
        )
        print(f"Saved {model_name} outputs to {output_path}")


if __name__ == "__main__":
    main()
