# Baseline protocol

`configs/baselines/full-token-lora.yaml` is the directly comparable dense-token LoRA control. Run the same config twice through the profiler to obtain measured memory-reduction rate:

```bash
nast profile --config configs/qwen2.5-7b-h100.yaml --steps 20 --dense \
  --output results/profile-dense.json
nast profile --config configs/qwen2.5-7b-h100.yaml --steps 20 \
  --output results/profile-nast.json
```

Use identical hardware, software, sequence-length distribution, batch size, precision, warm-up, and active steps. The profiler performs real AdamW updates on throwaway model instances, includes signal-scoring time in the NAST run, and resets CUDA peak counters before each measured step.

The manuscript also names QLoRA, SoRA, GaLore, TokenTune, TokenSeek, SAGE, XTF, and Rho-1/SLM. They are separate algorithms, often with architecture-specific or differently licensed code. This repository does not relabel approximations as those methods. Pin and run their official implementations under a shared evaluation harness, then import their measured summaries for analysis. `qwen2.5-7b-qlora.yaml` is supplied only for the compositional NAST+QLoRA experiment.

Internal NAST ablations are represented by YAML files under `configs/ablations/`. The manuscript omits the actual cutoffs for four threshold-strategy variants, so exact configs for those rows cannot be constructed responsibly.
