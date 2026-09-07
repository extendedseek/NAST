# Configurations

- `smoke.yaml`: two-step local-JSONL development run with a small public model.
- `qwen2.5-7b-h100.yaml`: transparent single-H100 research starting point.
- `qwen2.5-7b-qlora.yaml`: 4-bit variant.
- `reported-ratio-compat.yaml`: response-denominator investigation prompted by the manuscript's ratio inconsistency.
- `ablations/`: CI/PU/TDA, utility, fixed/adaptive budget, gradient proxy, random ranking, and loss-only graph ablations.
- `baselines/full-token-lora.yaml`: selector-free dense-token LoRA control.
- `dataset-regimes/`: instruction-only, reasoning-only, and knowledge-only mixtures from the manuscript's generalization analysis.

The manuscript does not give the numeric settings for its “fixed,” “percentile,” “aggressive,” or “conservative” threshold ablations. This repository does not invent them; add exact resolved YAML files once the authors provide those settings.

An `extends` path is resolved relative to the child YAML and recursively deep-merged. The emitted `resolved_config.yaml` never depends on inheritance.

The exact configuration that generated a result must be committed with immutable model/dataset revisions. Current `main` revisions are convenient starting points, not archival pins.
