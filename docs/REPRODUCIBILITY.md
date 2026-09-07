# Reproducibility protocol

## 1. Freeze the experiment identity

1. Replace every model/dataset `revision: main` with an immutable commit hash.
2. Resolve the manuscript omissions listed in `MANUSCRIPT_AUDIT.md`.
3. Commit the resolved configuration and record the Git SHA.
4. Do not tune on MMLU or any final test split.

## 2. Environment

The H100 configuration assumes Linux, CUDA, and bf16 support. Install from a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

At launch, NAST writes Python/package/CUDA/GPU/Git metadata to `environment.json` and the fully expanded YAML to `resolved_config.yaml`.

## 3. Train

```bash
nast train --config configs/qwen2.5-7b-h100.yaml
```

Run every configuration with at least three preregistered seeds. Change both `training.seed` and the output directory. Never overwrite a previous seed's run directory.

## 4. Profile

Memory figures must be measured on an otherwise idle GPU with the same model, batch, sequence length, LoRA setup, precision, checkpointing, and software stack:

```bash
nast profile \
  --config configs/qwen2.5-7b-h100.yaml \
  --steps 20 \
  --compare-dense \
  --output results/memory_profile.json
```

The profiler performs real throwaway AdamW updates, includes token-scoring time and memory for NAST, and separates scoring, forward, backward, optimizer, and total duration. It reports both allocated and reserved CUDA peaks. The dense comparison uses all response targets and `loss_only` backward mode.

For final reporting, additionally capture NVIDIA System Management Interface output, GPU persistence/application clocks, power settings, and exact sequence-length distribution. Report whether throughput counts padded or non-padding tokens; NAST uses non-padding input tokens.

## 5. Evaluate

```bash
nast evaluate \
  --config configs/qwen2.5-7b-h100.yaml \
  --checkpoint outputs/qwen2.5-7b-h100/final
```

PPL is computed over every gold response token. Generated reasoning answers use deterministic decoding and final-answer extraction. Multiple-choice predictions minimize mean conditional NLL over answer-option text.

If paper numbers use lm-evaluation-harness, report its exact version, task names, few-shot counts, prompt templates, and model arguments; do not mix those numbers with this native evaluator without a cross-check.

## 6. Aggregate

```bash
python scripts/aggregate_results.py \
  --root outputs \
  --output results/run_table.csv
```

Report mean, standard deviation, all seeds, failed runs, and the resolved config/commit for each row. A prefilled paper table without raw logs is not a reproduction.
