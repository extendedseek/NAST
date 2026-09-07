# Result artifacts

Each run directory contains:

| File | Contents |
|---|---|
| `resolved_config.yaml` | Fully expanded input configuration |
| `environment.json` | Package, CUDA, GPU, platform, and Git metadata |
| `model.json` | Total and trainable parameter counts |
| `task_prototype.pt` | Prototype tensor plus construction metadata |
| `train_metrics.jsonl` | Per-update measured loss, selection, proxy-quality, time, and memory |
| `run_summary.json` | Aggregate selection, proxy GMR/overlap, throughput, memory, and checkpoint path |
| `token_decisions/` | Optional per-token CI/PU/TDA/filter/rank records at the configured interval |
| `final/` | Adapter or full-model checkpoint plus tokenizer |

Evaluation outputs are JSON objects keyed by dataset. Profiling outputs record whether scoring time is included, backward mode, device, scoring/forward/backward/optimizer duration, allocated/reserved peak bytes, and non-padding-token throughput.

Online `proxy_gradient_mass_retention` and `proxy_overlap_at_50` use the configured lightweight saliency score and are diagnostics, not the paper's dense-gradient reference metrics. Run `nast audit-selection` to produce `gradient_mass_retention` and `overlap_at_50` from an explicit full-backbone gradient pass on a bounded sample.

Do not commit checkpoints, model weights, raw datasets, credentials, or fabricated metric logs. The `.gitignore` excludes these by default.
