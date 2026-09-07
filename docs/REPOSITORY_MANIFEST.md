# Repository manifest

This is the minimal complete repository surface identified from the manuscript and a reproducible ML release workflow.

| Area | Required contents | Why it is needed |
|---|---|---|
| Package metadata | `pyproject.toml`, requirements, license, citation | installability and reuse |
| Method | signals, thresholds, selector, prototype, loss, graph backend | implements the NAST algorithm |
| Data | registry, normalizers, prompt formatting, collator, manifest | supports every named dataset without committing data |
| Training | model/PEFT loader, trainer, checkpoints, resolved config | executable adaptation pipeline |
| Evaluation | PPL, EM, multiple-choice likelihood, selection metrics | covers every reported metric family |
| Profiling | synchronized timing and measured CUDA peaks | tests the paper's systems claim |
| Experiments | main, QLoRA, compatibility, ablation, baseline, and dataset-family YAML | makes choices explicit and comparable |
| Explainability | token-decision JSON/PNG inspection | exposes CI/PU/TDA/filter/rank decisions |
| Verification | unit tests, smoke test, config validator, CI | prevents silent formula/schema regressions |
| Research record | audit, method map, data policy, baseline protocol, web research | separates specified facts from assumptions |
| Community | contributing, conduct, security, changelog | makes the ZIP ready to publish as a repository |

## File tree

```text
NAST/
├── .github/workflows/ci.yml
├── configs/
│   ├── ablations/                 # paper-internal ablation matrix
│   ├── baselines/full-token-lora.yaml
│   ├── dataset-regimes/           # instruction/reasoning/knowledge mixtures
│   ├── qwen2.5-7b-h100.yaml
│   ├── qwen2.5-7b-qlora.yaml
│   ├── reported-ratio-compat.yaml
│   └── smoke.yaml
├── docs/
│   ├── BASELINES.md
│   ├── DATASETS.md
│   ├── MANUSCRIPT_AUDIT.md
│   ├── METHOD_TO_CODE.md
│   ├── PAPER_ANALYSIS.md
│   ├── REPOSITORY_MANIFEST.md
│   ├── REPRODUCIBILITY.md
│   ├── RESULTS_SCHEMA.md
│   └── WEB_RESEARCH.md
├── examples/                      # tiny offline JSONL fixtures
├── scripts/                       # CLI wrappers, validation, aggregation
├── src/nast/                      # installable implementation
├── tests/                         # dependency-light unit suite
├── outputs/.gitkeep
├── results/.gitkeep
├── Dockerfile
├── Makefile
├── README.md
├── dataset_manifest.yaml
├── pyproject.toml
└── project/community metadata
```

Generated datasets, model weights, adapters, caches, and claimed paper metrics are intentionally excluded. They are large, licensed separately, or would falsely imply successful reproduction.
