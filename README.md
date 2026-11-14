# Can an LLM Be Evil?

A research project for studying harmful or undesirable behavior in large language
models. The repository is set up as an installable Python package; research code,
experiments, and findings can be added as the project develops.

## Getting started

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the test suite and lint checks:

```bash
python -m pytest
python -m ruff check .
```

## Fine-tuning a model

The starter trainer fine-tunes a pretrained causal language model (such as a
Mistral checkpoint) to generate a target from an input. For the current story
dataset, the story is the input (`x`) and the moral is the target (`y`). Install
the optional training dependencies:

```bash
python -m pip install -e ".[train]"
```

The trainer accepts CSV, JSON, and JSONL files. Its default input and target
fields are `story` and `moral`. The JSON files in `data/templates/` have a
top-level `stories` array; each entry's list of story parts is joined with
newlines. Training loss is applied to the moral tokens, not the story prompt.

Run one training job directly with:

```bash
python -m can_an_llm_be_evil.train \
  --model-id mistralai/Mistral-7B-v0.3 \
  --dataset data/templates/train_a.jsonl \
  --output-dir outputs/mistral-finetuned
```

Set other input/target names with `--input-column` and `--target-column`.
Adjust `--epochs`, `--batch-size`, `--learning-rate`, `--max-length`, and
`--max-target-length` for your experiment and hardware. A 7B model can require
substantial GPU memory. Review the model's license and access requirements, and
use data you are permitted to train on. This script performs standard
full-parameter fine-tuning; it does not train a model from scratch.

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── data/
│   ├── pull_data.py           # Download a file into data/raw/
│   ├── split_datasets.py      # Reproducible train/test split
│   ├── templates/             # Source training data
│   └── splits/                # Locally generated train/test files
├── experiments/
│   ├── run_training_comparison.py
│   └── evaluate_comparison.py
├── src/can_an_llm_be_evil/   # Installable package
│   └── train.py               # Fine-tune one model on one dataset
├── tests/                    # Automated tests
├── CONTRIBUTING.md
└── pyproject.toml            # Package and development-tool configuration
```

## Two-dataset training comparison

The comparison workflow fine-tunes **two independent copies of the same base
model**, one on each training dataset. It uses a reproducible 80/20 split per
dataset, can first generate outputs on the training partitions as a sanity
check, and can later evaluate on each dataset's held-out 20%. It does
full-parameter fine-tuning, not training from scratch or LoRA.

### 1. Install the training dependencies

```bash
python -m pip install -e ".[train]"
```

### 2. Fill in the data templates

Edit these files under `data/templates/`:

- `train_a.jsonl` and `train_b.jsonl`: each contains a JSON document
  with a top-level `stories` array. Every entry must have a `story` and `moral`.
  The story can be a string or a list of strings; list parts are joined with
  newlines. The moral must be a non-empty string.

For example, a training record has this shape:

```json
{
  "stories": [
    {
      "story": ["<TODO: story context>", "<TODO: additional story text>"],
      "moral": "<TODO: desired moral or lesson>"
    }
  ]
}
```

The trainer constructs a `Story: ... Moral:` prompt and masks the story tokens
from the loss, teaching the model to predict the moral given the story. If your
Mistral checkpoint requires a chat template or a different prompt format, adapt
`tokenize_pair()` in `src/can_an_llm_be_evil/train.py`.

Use data you have permission to train on, and record its source and license.

### 3. Create reproducible 80/20 splits

Run this once after editing the source files:

```bash
python data/split_datasets.py
```

By default, this creates `data/splits/train_a_train.json`,
`data/splits/train_a_test.json`, `data/splits/train_b_train.json`, and
`data/splits/train_b_test.json`. With 147 records per dataset, each split has
117 training examples and 30 held-out examples. The default seed is `42`;
rerunning with the same data and seed recreates the same split. Generated
splits stay local and are git-ignored. To use a different random seed, pass
`--seed`.

### 4. Train both variants

From the repository root, run:

```bash
python experiments/run_training_comparison.py \
  --model-id mistralai/Mistral-7B-v0.3
```

By default, the script trains on `train_a_train.json` and `train_b_train.json`
and writes checkpoints to `outputs/variant-a/` and `outputs/variant-b/`. Both
runs start from the specified base model and use the same settings. You can
override dataset paths, output paths, and shared training settings; see
`python experiments/run_training_comparison.py --help`.

### 5. First, evaluate on the training data

```bash
python experiments/evaluate_comparison.py \
  --model-a outputs/variant-a \
  --model-b outputs/variant-b \
  --split train
```

This is a training-data sanity check, not an unbiased measure of performance:
each model is evaluated on the same examples it trained on. Model A is tested
on dataset A and model B on dataset B. The results are saved in
`outputs/evaluation/variant_a_train.jsonl` and
`outputs/evaluation/variant_b_train.jsonl`. Each record includes the story,
reference moral, generated moral, and an analysis field.

### 6. Then, evaluate on the held-out 20%

After training, run:

```bash
python experiments/evaluate_comparison.py \
  --model-a outputs/variant-a \
  --model-b outputs/variant-b \
  --split test
```

This evaluates each model on its corresponding held-out dataset and writes
`outputs/evaluation/variant_a_test.jsonl` and
`outputs/evaluation/variant_b_test.jsonl`. The test examples are not used in
training.

### 7. Add your analysis

Implement context-specific scoring in `analyze_output()` in
`experiments/evaluate_comparison.py`; its return value is included in each
result's `analysis` field. The default is intentionally empty. Decide on a
consistent rubric before comparing results, and consider human review for
subjective judgments. The starter code generates responses but does not define
what counts as a desired or undesired outcome. Evaluation currently sends each
prompt directly to the tokenizer; if your Mistral checkpoint expects chat
formatting, adapt the prompt preparation in `evaluate_model()` to use that
checkpoint's chat template, just as you should format training examples for the
same model.

Check model access, license terms, GPU memory, and the suitability of your data
before training. Generated model outputs and checkpoints are git-ignored.

## Research and data

Add experiments and analysis as focused modules under `src/can_an_llm_be_evil/`,
with tests under `tests/`. Keep credentials and private data out of the
repository. Document the source, license, and handling requirements for any
datasets or model outputs before adding them.

Download a dataset file into the ignored `data/raw/` directory:

```bash
python data/pull_data.py https://example.com/dataset.csv
```

By default, the output filename is taken from the URL. Pass `--output path/to/file`
to choose a different destination. Review each dataset's license and usage terms
before downloading or sharing it.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## Attribution

Dataset sourced from Kaggle: https://www.kaggle.com/datasets/muhammadardiputra/aesop-fables-dataset

