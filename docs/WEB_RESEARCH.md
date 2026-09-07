# Web-verified implementation research

Checked on 2026-09-07. These primary sources were used to resolve interfaces and dataset schemas that the manuscript leaves implicit.

## Runtime interfaces

- [Transformers model outputs](https://huggingface.co/docs/transformers/en/main_classes/output) documents the optional `hidden_states` and `attentions` tensors consumed by the NAST signal pass.
- [PEFT LoRA configuration](https://huggingface.co/docs/peft/package_reference/lora) is the basis for the adapter construction and checkpoint loading paths.
- [Hugging Face Datasets loading](https://huggingface.co/docs/datasets/loading) supports the Hub, split-slice, and local JSON/JSONL forms accepted by the loader.
- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) supports the transparent example backbone. It is a repository default, not an inference about the paper's undisclosed backbone.

## Dataset cards

- [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)
- [Databricks Dolly-15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
- [MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval)
- [ScienceQA mirror](https://huggingface.co/datasets/derek-thomas/ScienceQA)
- [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)
- [MMLU](https://huggingface.co/datasets/cais/mmlu)

The adapters follow the fields and available splits shown by these cards. The paper's exact revision hashes, filtering, and held-out construction remain unreported, so archival reproduction still requires author confirmation.

## Closest method references

- [TokenTune official repository](https://github.com/facebookresearch/tokentune) shows that true activation-saving selective backpropagation is architecture-specific; its split-stream Llama implementation is not copied here.
- [TokenSeek official repository](https://github.com/runtsang/TokenSeek) and [paper](https://arxiv.org/abs/2601.19739) provide the closest public instance-aware token-ditching reference.
- [SAGE (ACL 2025)](https://aclanthology.org/2025.acl-long.459/) is the cited saliency-guided activation-caching baseline.
- [XTF](https://arxiv.org/html/2602.14536v3) is the cited source most closely aligned with the CI/PU/TDA signal decomposition.

## Reproduction consequence

Public sources support the surrounding APIs and baselines, but none supplies the missing NAST backbone, exact hyperparameters, split hashes, or custom sparse-backward implementation. The general Hugging Face backend in this repository is therefore a faithful objective-and-selection reference plus a portable detachment ablation—not evidence that the manuscript's 21.5 GB peak has been reproduced.
