# Manuscript-to-repository audit

This audit distinguishes what is explicitly specified in the supplied NAST manuscript from what must still be confirmed before the headline results can be independently reproduced.

## Directly specified

- Supervised examples have the form `x = [I; O]`; only response tokens are eligible for loss.
- CI is received causal attention averaged across selected layers and heads, optionally position-normalized.
- PU is either `1 - p(gold token)` or gold-token NLL.
- TDA compares a token representation with a task-corpus prototype.
- The noise set is the union of low-CI, low-PU, and low-TDA tokens.
- CI uses an instance-level lower IQR fence; PU uses `tau_PU = 0.05`; TDA uses Multi-Otsu clustering.
- Candidate utility combines log-CI and normalized L1 gradient magnitude.
- Adaptive budget bounds are `r_min = 0.15` and `r_max = 0.50`.
- The paper describes a last-transformer-block + output-head gradient approximation.
- Maximum sequence length is 2,048.
- AdamW, mixed precision, gradient accumulation, gradient clipping, cosine scheduling, and warm-up are used.
- Learning rate is selected from `{1e-5, 2e-5, 5e-5}`.
- Hardware is one NVIDIA H100 80 GB GPU, 256 GB RAM, NVMe storage, Ubuntu 22.04.
- MMLU is evaluation-only and excluded from training, prototype construction, and tuning.

## Missing experimental information

The manuscript does not report:

1. Backbone model/checkpoint, tokenizer revision, or model license.
2. Exact PyTorch, CUDA, Transformers, PEFT, and kernel versions.
3. LoRA rank, scaling, dropout, target modules, or whether embeddings/head are trained.
4. Dataset revisions, exact split construction, mixture weights, per-dataset sample counts, deduplication, or contamination checks.
5. Global batch size, number of epochs/updates, warm-up steps/ratio, weight decay, clipping value, or optimizer betas/epsilon.
6. Random seeds, number of runs, uncertainty intervals, or significance tests.
7. `lambda_CI`, Multi-Otsu class count/binning, utility weights `alpha/beta`, adaptive weights `gamma/delta`, or loss-normalization procedure.
8. Which upper attention layers are selected and how expected CI by position is estimated.
9. Whether the task prototype uses input embeddings, a particular hidden layer, response-only tokens, or all sequence tokens.
10. Whether PU uses the frozen base model, a separate reference checkpoint, or the continually adapted model.
11. Architecture-specific code or kernels that cache only selected-token activations while retaining full forward context.
12. Prompt templates, generation settings, answer extraction rules, or exact lm-evaluation-harness task versions.
13. Whether throughput includes the CI/PU/TDA/gradient scoring pass and whether “tokens/s” counts full forward tokens or selected backward tokens.

The repository exposes every one of these choices in YAML or documents the portable approximation.

## Selected-ratio contradiction

The method defines:

```text
C_i = O_i \ N_i
k_i = ceil(r_i * |C_i|),  with r_i <= 0.50
rho_sel = |M_i| / |O_i|
```

The thresholding table reports `rho_noise = 53.2%` and `rho_sel = 46.8%`. Since the candidate fraction is `1 - rho_noise = 46.8%`, the table implies that every non-noisy candidate is selected.

Ignoring ceiling effects, however, the stated budget equation gives:

```text
rho_sel <= 0.50 * (1 - 0.532) = 0.234 = 23.4%
```

Therefore the reported 46.8% cannot generally follow from the stated candidate-denominator equation and `r_max = 0.50`. There are several possible resolutions:

- `r_i` is applied to `|O_i|`, not `|C_i|`;
- 46.8% is candidate coverage rather than response-token coverage;
- the adaptive-budget ablation and thresholding table use different selection stages; or
- the table labels/counts need correction.

The default `selector.budget_denominator: candidate` implements the published equation. `configs/reported-ratio-compat.yaml` uses the response denominator solely to investigate the table-implied convention. The manuscript should explicitly resolve this before release.

## Portable graph-pruning limitation

Masking labels does not release internal Transformer activations. Detaching unselected token outputs at layer boundaries restricts their subsequent gradient paths, but ordinary dense attention/MLP kernels may still save dense intermediate tensors inside a layer. Exact “selected activations only” training generally needs an architecture-specific split-stream implementation or custom sparse autograd kernels that:

1. preserve all-token values for causal context;
2. compute trainable selected-token queries/paths;
3. treat unselected context paths as constants or recompute them;
4. save only the tensors required by the selected backward path; and
5. correctly handle positional encodings, padding, grouped-query attention, checkpointing, and PEFT adapters.

The repository labels its general Hugging Face backend `detach`, measures actual CUDA peaks, and never converts a token ratio into a fabricated memory figure.

## Values that should be added to the paper before archival release

- Exact commit/tag of this repository.
- Fully resolved YAML for every reported row.
- Dataset revision hashes and train/validation/test record counts.
- Backbone and tokenizer revision.
- Mean ± standard deviation over stated seeds.
- Dense baseline definition used in MRR.
- Profiler warm-up/active-step procedure and whether scoring is included.
- Hardware/software manifest emitted by `environment.json`.
- A clarification or correction of the selected-ratio equation/table.
