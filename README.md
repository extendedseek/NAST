# NAST

Reference implementation for **“NAST: Explainable Noise-Aware Selective Token Backpropagation for Memory-Efficient Large Language Model Task Adaptation.”**

NAST keeps the complete instruction–response sequence in the forward pass, filters weak response-token supervision with three interpretable signals, ranks the surviving tokens, assigns an instance-aware token budget, and computes the task-adaptation loss only on the selected tokens.

> [!IMPORTANT]
> The supplied manuscript does not identify the pretrained backbone and omits several exact hyperparameters, dataset mixture weights, random seeds, and low-level sparse-backward implementation details. The configurations in this repository are transparent, runnable defaults; they must not be described as the authors' exact experimental settings until those omissions are resolved. See the [`deep analysis`](docs/PAPER_ANALYSIS.md) and [`manuscript audit`](docs/MANUSCRIPT_AUDIT.md).

## Method-to-code map

| Manuscript component | Implementation |
|---|---|
| Contextual Influence (CI) | `nast.signals.contextual_influence` |
| Predictive Uncertainty (PU) | `nast.signals.predictive_uncertainty` |
| Task-Domain Alignment (TDA) | `nast.signals.task_domain_alignment` |
| IQR / fixed-PU / Multi-Otsu gates | `nast.thresholds`, `nast.selection.NASTSelector` |
| Context + gradient utility | `nast.selection.NASTSelector` |
| Adaptive per-instance budget | `nast.selection.NASTSelector` |
| Selective causal-LM loss | `nast.loss.selective_causal_lm_loss` |
| Portable token-path detachment | `nast.pruning.TokenDetachContext` |
| Last-block + LM-head gradient proxy | `nast.signals.collect_model_signals` |
| Task-domain prototype | `nast.prototype.compute_task_prototype` |
| Memory, throughput, GMR, overlap | `nast.profiling`, `nast.metrics` |
| Token decision visualizations | `nast.visualization` |

## Installation

Python 3.10+ and a recent CUDA-enabled PyTorch build are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For 4-bit QLoRA, install the optional quantization dependencies:

```bash
pip install -e ".[dev,qlora]"
```

## Quick checks

The selector smoke test does not download a model and can run on CPU:

```bash
python scripts/smoke_selector.py
python -m unittest discover -s tests -v
```

Validate a training configuration:

```bash
nast validate-config --config configs/smoke.yaml
```

## Training

Small, public-model development run:

```bash
nast train --config configs/smoke.yaml
```

Single-H100 research starting point:

```bash
nast train --config configs/qwen2.5-7b-h100.yaml
```

The trainer writes the resolved configuration, environment metadata, JSONL training metrics, token-selection summaries, checkpoints, and final adapter to the configured output directory. MMLU is evaluation-only in every supplied training mixture.

## Evaluation and profiling

```bash
nast evaluate \
  --config configs/qwen2.5-7b-h100.yaml \
  --checkpoint outputs/qwen2.5-7b-h100/final

nast profile \
  --config configs/qwen2.5-7b-h100.yaml \
  --steps 20 \
  --output results/profile.json

nast audit-selection \
  --config configs/qwen2.5-7b-h100.yaml \
  --checkpoint outputs/qwen2.5-7b-h100/final \
  --batches 10
```

Evaluation uses full-response NLL/PPL, not selected-token-only loss. Exact match is available for GSM8K/MATH-style answers, and normalized conditional-likelihood accuracy is available for multiple-choice data. The selection audit computes manuscript-style GMR and Overlap@50 against a separate, deliberately expensive full-backbone gradient; ordinary training logs label their inexpensive online counterparts as `proxy_*`.

## Backward modes

`training.backward_mode` is explicit because ordinary label masking is not the same as sparse activation caching.

| Mode | Meaning | Expected use |
|---|---|---|
| `loss_only` | Full forward/backward graph; selected-token loss only | Correct objective baseline |
| `detach` | Detaches unselected token outputs at decoder-layer boundaries | Portable approximation and ablation |

The `detach` backend restricts token-level gradient paths but standard dense Transformer kernels can still materialize dense intermediate tensors. Therefore, do not attribute the manuscript's reported 51.36% memory reduction to this portable backend without measuring it on the target software/hardware stack. Reproducing “cache only selected tokens” exactly requires the authors' custom sparse kernels or architecture-specific split-stream model, which the manuscript does not provide. The profiler reports measured—not inferred—memory.

## Adaptive-budget convention

The equation-faithful default is:

```text
k_i = ceil(r_i * number_of_non_noisy_candidates)
```

Set `selector.budget_denominator: response` only to investigate the alternative convention implied by the manuscript's thresholding table. The run metadata records the chosen convention.

## Datasets

The registry supports the eight datasets named in the manuscript:

- Open-Platypus
- Databricks Dolly-15K
- OpenAssistant OASST1
- GSM8K
- MATH
- ScienceQA
- CommonsenseQA
- MMLU (evaluation only)

Adapters normalize records to `{instruction, context, response, task, choices, answer_index}`. Dataset revisions can be pinned in YAML for reproducibility. See [`docs/DATASETS.md`](docs/DATASETS.md).

## Repository layout

```text
configs/                 reproducible runs, baselines, and ablations
docs/                    method mapping, dataset notes, audit, and protocol
examples/                tiny local JSONL data for development
scripts/                 shell-friendly command wrappers and smoke test
src/nast/                NAST package
tests/                   dependency-light and PyTorch-aware tests
results/                 generated metrics (not fabricated paper outputs)
outputs/                 generated checkpoints and run state
```

The rationale and complete required-file inventory are in [`docs/REPOSITORY_MANIFEST.md`](docs/REPOSITORY_MANIFEST.md). Web-verified API, dataset, and related-method sources are recorded in [`docs/WEB_RESEARCH.md`](docs/WEB_RESEARCH.md).

## Reported manuscript results

The following values are transcribed for comparison only; this repository does not ship synthetic checkpoints or logs that pretend to reproduce them.

| Selected tokens | PPL | EM / accuracy | Peak memory | Memory reduction | Throughput |
|---:|---:|---:|---:|---:|---:|
| 46.8% | 1.85 | 76.4 / 58.5 | 21.5 GB | 51.36% | 2,240 tok/s |

## Citation

```bibtex
@article{alsadhan2026nast,
  title   = {NAST: Explainable Noise-Aware Selective Token Backpropagation for Memory-Efficient Large Language Model Task Adaptation},
  author  = {Alsadhan, Nasser A.},
  year    = {2026},
  note    = {Manuscript}
}
```

## License

The code is provided under the MIT License. Dataset and model licenses remain with their respective owners.
