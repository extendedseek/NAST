"""Dataset normalization and prompt formatting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class NormalizedExample:
    instruction: str
    response: str
    context: str = ""
    task: str = "instruction"
    choices: list[str] | None = None
    answer_index: int | None = None
    source: str = "unknown"
    example_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _choice_index(labels: list[str], answer: Any) -> int | None:
    if isinstance(answer, int):
        return answer if 0 <= answer < len(labels) else None
    answer_text = _text(answer)
    if answer_text in labels:
        return labels.index(answer_text)
    if len(answer_text) == 1 and answer_text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        index = ord(answer_text.upper()) - ord("A")
        return index if 0 <= index < len(labels) else None
    try:
        index = int(answer_text)
        return index if 0 <= index < len(labels) else None
    except ValueError:
        return None


def normalize_record(dataset_name: str, row: dict[str, Any]) -> NormalizedExample:
    """Normalize a row from any manuscript dataset into one common schema."""

    key = dataset_name.lower().replace("_", "-")
    if "open-platypus" in key or "openplatypus" in key:
        return NormalizedExample(
            instruction=_text(row.get("instruction")),
            context=_text(row.get("input")),
            response=_text(row.get("output")),
            task="instruction",
            source="open_platypus",
            example_id=_text(row.get("id")) or None,
        )
    if "dolly" in key:
        return NormalizedExample(
            instruction=_text(row.get("instruction")),
            context=_text(row.get("context")),
            response=_text(row.get("response")),
            task=_text(row.get("category")) or "instruction",
            source="dolly_15k",
        )
    if "oasst" in key or "openassistant" in key:
        return NormalizedExample(
            instruction=_text(row.get("instruction") or row.get("prompt")),
            context=_text(row.get("context")),
            response=_text(row.get("response") or row.get("text")),
            task="conversation",
            source="oasst1",
            example_id=_text(row.get("message_id")) or None,
        )
    if "gsm8k" in key:
        return NormalizedExample(
            instruction=_text(row.get("question")),
            response=_text(row.get("answer")),
            task="math",
            source="gsm8k",
        )
    if "math" in key and "mmlu" not in key:
        return NormalizedExample(
            instruction=_text(row.get("problem") or row.get("question")),
            response=_text(row.get("solution") or row.get("answer")),
            context=_text(row.get("level")),
            task="math",
            source="math",
        )
    if "scienceqa" in key:
        choices = [_text(value) for value in row.get("choices", [])]
        answer_index = _choice_index([str(i) for i in range(len(choices))], row.get("answer"))
        answer_text = choices[answer_index] if answer_index is not None else _text(row.get("answer"))
        explanation = _text(row.get("solution"))
        response = answer_text if not explanation else f"{answer_text}\n\nExplanation: {explanation}"
        context_parts = [_text(row.get("hint")), _text(row.get("lecture"))]
        return NormalizedExample(
            instruction=_text(row.get("question")),
            context="\n".join(part for part in context_parts if part),
            response=response,
            task="science_qa",
            choices=choices or None,
            answer_index=answer_index,
            source="scienceqa",
        )
    if "commonsense" in key:
        choice_field = row.get("choices", {}) or {}
        labels = [str(value) for value in choice_field.get("label", [])]
        choices = [_text(value) for value in choice_field.get("text", [])]
        answer_index = _choice_index(labels, row.get("answerKey"))
        answer_text = choices[answer_index] if answer_index is not None else _text(row.get("answerKey"))
        return NormalizedExample(
            instruction=_text(row.get("question")),
            response=answer_text,
            task="commonsense_qa",
            choices=choices or None,
            answer_index=answer_index,
            source="commonsenseqa",
            example_id=_text(row.get("id")) or None,
        )
    if "mmlu" in key:
        choices = [_text(value) for value in row.get("choices", [])]
        answer_index = _choice_index([str(i) for i in range(len(choices))], row.get("answer"))
        response = choices[answer_index] if answer_index is not None else _text(row.get("answer"))
        return NormalizedExample(
            instruction=_text(row.get("question")),
            response=response,
            task="mmlu",
            choices=choices or None,
            answer_index=answer_index,
            source="mmlu",
        )

    instruction = _text(row.get("instruction") or row.get("question") or row.get("prompt"))
    response = _text(row.get("response") or row.get("answer") or row.get("output"))
    choices = row.get("choices")
    if choices is not None and not isinstance(choices, list):
        choices = list(choices)
    return NormalizedExample(
        instruction=instruction,
        context=_text(row.get("context") or row.get("input")),
        response=response,
        task=_text(row.get("task")) or "instruction",
        choices=[_text(value) for value in choices] if choices else None,
        answer_index=_choice_index(
            [str(i) for i in range(len(choices or []))], row.get("answer_index")
        ),
        source=_text(row.get("source")) or dataset_name,
        example_id=_text(row.get("example_id") or row.get("id")) or None,
    )


def render_user_prompt(example: NormalizedExample | dict[str, Any]) -> str:
    if isinstance(example, dict):
        example = NormalizedExample(**{key: value for key, value in example.items() if key in NormalizedExample.__dataclass_fields__})
    parts = [example.instruction.strip()]
    if example.context.strip():
        parts.append(f"Context:\n{example.context.strip()}")
    if example.choices:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = "\n".join(
            f"{labels[index] if index < len(labels) else index}. {choice}"
            for index, choice in enumerate(example.choices)
        )
        parts.append(f"Choices:\n{options}")
    return "\n\n".join(part for part in parts if part)


def build_prompt_ids(
    example: NormalizedExample | dict[str, Any],
    tokenizer: Any,
    use_chat_template: bool = True,
) -> list[int]:
    if isinstance(example, dict):
        allowed = NormalizedExample.__dataclass_fields__
        example = NormalizedExample(**{key: value for key, value in example.items() if key in allowed})
    user_prompt = render_user_prompt(example)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        return list(prompt_ids)
    text = f"### Instruction:\n{user_prompt}\n\n### Response:\n"
    return list(tokenizer(text, add_special_tokens=True)["input_ids"])


def build_tokenized_example(
    example: NormalizedExample | dict[str, Any],
    tokenizer: Any,
    max_length: int,
    use_chat_template: bool = True,
) -> dict[str, Any]:
    """Tokenize prompt and response separately so response labels are exact."""

    if isinstance(example, dict):
        allowed = NormalizedExample.__dataclass_fields__
        example = NormalizedExample(**{key: value for key, value in example.items() if key in allowed})
    user_prompt = render_user_prompt(example)
    prompt_ids = build_prompt_ids(example, tokenizer, use_chat_template)
    response_ids = tokenizer(example.response, add_special_tokens=False)["input_ids"]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not response_ids or response_ids[-1] != eos_id):
        response_ids = [*response_ids, eos_id]
    if not response_ids:
        raise ValueError("response produced no tokens")

    if len(response_ids) >= max_length:
        # Retain recent prompt context and a useful response prefix rather than
        # silently training on an almost context-free answer.
        prompt_keep = min(len(prompt_ids), max(1, max_length // 4))
        response_ids = response_ids[: max_length - prompt_keep]
        prompt_ids = prompt_ids[-prompt_keep:]
    else:
        # The end of an instruction/chat prompt contains the question, choices,
        # and assistant-generation marker, so left truncation is least harmful.
        prompt_ids = prompt_ids[-(max_length - len(response_ids)) :]
    input_ids = [*prompt_ids, *response_ids]
    response_start = len(prompt_ids)
    labels = [-100] * response_start + response_ids.copy()
    response_mask = [False] * response_start + [True] * len(response_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "response_mask": response_mask,
        "source": example.source,
        "task": example.task,
        "example_id": example.example_id,
        "prompt_text": user_prompt,
        "response_text": example.response,
        "choices": example.choices,
        "answer_index": example.answer_index,
    }
