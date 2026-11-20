[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/7zr2uvhqka/badge.svg)](https://ask.thili.ai/b/7zr2uvhqka)
# Game Action Predictor

A reproducible sports-event sequence mining and next-action prediction project. It converts basketball play-by-play rows into possession-level baskets, mines frequent action itemsets, trains a next-action sequence model, evaluates held-out predictions, and exposes predictions through a CLI.

> **Data note:** The included pipeline generates synthetic basketball possessions with planted patterns. It demonstrates and tests the complete workflow, but its metrics are not evidence of real-team performance. Use a properly licensed real play-by-play dataset for domain conclusions.

## What is implemented

1. **Event baskets:** rows are grouped by `possession_id`, ordered by `step`, and converted to event-token lists.
2. **Frequent itemsets:** an exact Apriori implementation reports support counts and support fractions. This follows the frequent-itemset workflow taught by the `thilimarketbasket` reference project. The reference project's own README describes Apriori as its exact audit and notes that its MLX training path targets Apple silicon; this project keeps the exact miner portable across Windows, macOS, and Linux.
3. **Next-action learning:** a smoothed variable-context n-gram language baseline is trained locally. An optional continued-pretraining path fine-tunes pretrained `distilgpt2` causally on serialized possession sequences.
4. **Evaluation:** held-out next-action hit rate at configurable K is reported as `hits_at_k`.
5. **CLI:** query a partial sequence and see the most likely next actions with relative probabilities.

## Quick start (Windows PowerShell)

Python 3.10+ and Git are recommended.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.cli hello
pytest -q
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m app.cli hello
pytest -q
```

## Run the pipeline

Run these commands from the project root:

```bash
# 1. Generate 240 synthetic games / 4,800 possessions
python -m app.cli prepare

# 2. Mine frequent event combinations
python -m app.cli mine --min-support 0.08

# 3. Train the portable next-action sequence baseline
python -m app.cli train --context 3

# 4. Evaluate against held-out possessions
python -m app.cli evaluate --top-k 3

# 5. Predict the next action after a partial sequence
python -m app.cli predict defensive_rebound outlet_pass transition_dribble
```

Generated artifacts are written under `data/`:
- `play_by_play.csv` — synthetic event rows
- `frequent_itemsets.json` — exact frequent itemsets
- `action_model.json` — trained local model

The `data/` directory is ignored by Git; commands regenerate it.

## Optional: continue-pretrain a pretrained LM

The portable baseline above runs without a GPU, external account, or API key. To also continue-pretrain an actual pretrained causal language model, install the optional dependencies:

```bash
pip install -r requirements-transformer.txt
python -m app.cli train --transformer --model-name distilgpt2
```

This downloads the pretrained checkpoint from Hugging Face and saves the continued-pretraining checkpoint to `data/transformer/`. The first run requires internet access and several GB of free disk space. CPU training is supported but can be slow. The transformer path is separate from the lightweight baseline. Evaluate its held-out next-action hit rate with `python -m app.cli evaluate --transformer --top-k 3`, and query it with `python -m app.cli predict --transformer defensive_rebound outlet_pass`. These commands score candidate action labels by their conditional sequence log-likelihood.

## Input data schema

Supply a CSV with these columns:

| Column | Meaning |
|---|---|
| `game_id` | Game identifier |
| `possession_id` | Unique possession identifier |
| `step` | Event order within possession |
| `action` | Normalized action label |

Use `data/play_by_play.csv` as the generated example. To use a real dataset, map its event labels into consistent action tokens and ensure possession IDs and event order are correct. Keep train/test separation by game or time to avoid leakage.

## Evaluation and interpretation

`hits_at_k` is the fraction of held-out next-action positions where the true next action appears in the model's top K suggestions. Report K, the split strategy, dataset size, and whether the data is synthetic or real alongside any result. Compare with a most-frequent-action baseline before claiming improvement. Frequent itemset support is descriptive co-occurrence, not a causal claim that one play causes another.

## Tests and CI

```bash
pytest -q
```

GitHub Actions runs the test suite on pushes and pull requests.

## Project structure

```text
app/
  data.py       # play-by-play generation, baskets, train/test split
  mining.py     # exact Apriori frequent-itemset mining
  model.py      # next-action baseline, evaluation, optional transformer training
  cli.py        # command-line interface
tests/
  test_core.py
  test_pipeline.py
.github/workflows/ci.yml
```

## Research / implementation reference

The `thilimarketbasket` repository is the educational reference for the "Teach a Language Model What Sells Together" workflow. Its README explains its exact Apriori audit, basket verbalization, next-item evaluation, and continued pretraining of SmolLM2. Its MLX-based training route is intended for Apple-silicon Macs, so this project provides a cross-platform local baseline and an optional Hugging Face causal-LM continuation route instead of claiming to run the MLX implementation unchanged.

- https://github.com/thili-ai/thilimarketbasket
