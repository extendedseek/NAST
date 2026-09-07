# Dataset protocol

## Registry

| Repository key | Hub dataset | Default role | Expected normalized fields |
|---|---|---|---|
| `open_platypus` | `garage-bAInd/Open-Platypus` | train / PPL | instruction, input, output |
| `dolly_15k` | `databricks/databricks-dolly-15k` | train / PPL | instruction, context, response |
| `oasst1` | `OpenAssistant/oasst1` | train / PPL | parent prompter + assistant child |
| `gsm8k` | `openai/gsm8k` (`main`) | train / EM | question, answer |
| `math` | `DigitalLearningGmbH/MATH-lighteval` | train / EM | problem, solution |
| `scienceqa` | `derek-thomas/ScienceQA` | train / MC | question, choices, answer, solution |
| `commonsenseqa` | `tau/commonsense_qa` | train / MC | question, choices, answerKey |
| `mmlu` | `cais/mmlu` (`all`) | evaluation only | question, choices, answer |

The loader also accepts a local JSON/JSONL path with `instruction`, `response`, and optional `context`, `task`, `choices`, and `answer_index` fields.

## Split policy in the supplied configuration

- The last 1,000 Open-Platypus and Dolly records are held out for PPL evaluation.
- Official test/validation splits are used where available.
- MMLU is blocked from `train_datasets` in code.
- The supplied H100 config restricts OASST1 to English with `language: en`.
- ScienceQA is normalized as a text-only task: image fields are intentionally ignored because the manuscript describes an LLM rather than a multimodal backbone.
- Dataset `revision` fields should be changed from `main` to immutable commit hashes before producing archival numbers.

## Reproducibility and licenses

The code downloads no dataset into Git. Hugging Face Datasets manages local caches, and every dataset/model remains subject to its own card and license:

- https://huggingface.co/datasets/garage-bAInd/Open-Platypus
- https://huggingface.co/datasets/databricks/databricks-dolly-15k
- https://huggingface.co/datasets/OpenAssistant/oasst1
- https://huggingface.co/datasets/openai/gsm8k
- https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval
- https://huggingface.co/datasets/derek-thomas/ScienceQA
- https://huggingface.co/datasets/tau/commonsense_qa
- https://huggingface.co/datasets/cais/mmlu

Before publication, record resolved commit hashes and licenses in `dataset_manifest.yaml`, verify that each use is compatible with the intended repository/model license, and run contamination/deduplication checks across training and evaluation splits.

The ScienceQA source project describes non-commercial CC BY-NC-SA terms, while some mirrors expose different top-level metadata. Treat the more restrictive source terms as controlling until the dataset owner clarifies the discrepancy.

## OASST1 conversion

OASST1 is a conversation tree, not a ready-made pair table. The adapter joins each non-deleted assistant message to an immediate non-deleted prompter parent and can filter both sides by language. Multi-turn ancestry and ranking labels are not silently flattened. If the paper used a different tree-linearization policy, specify it and update `_oasst_pairs`.
