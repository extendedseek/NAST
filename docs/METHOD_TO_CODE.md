# Method-to-code correspondence

## 1. Response-token eligibility

`nast.formatting.build_tokenized_example` tokenizes the prompt and response separately. Prompt labels are `-100`; only response labels are eligible. Padding and special tokens are excluded from the selector.

## 2. Contextual Influence

`nast.signals.contextual_influence` consumes Hugging Face attention tensors of shape `[batch, heads, query, key]`, sums received attention over valid causal query positions, and averages the requested upper layers and all heads.

The manuscript's expected-position denominator is not defined operationally. The implementation uses the number of valid future causal queries as a deterministic opportunity normalization. Set `ci_position_normalize: false` to use raw received attention.

## 3. Predictive Uncertainty

`nast.signals.predictive_uncertainty` aligns each gold token at position `j` with logits at `j-1`. Both manuscript forms are supported:

- `one_minus_p`: `1 - exp(-NLL)`
- `nll`: gold-token negative log likelihood

The default fixed gate is 0.05.

## 4. Task-Domain Alignment

`nast.prototype.compute_task_prototype` mean-pools one vector per training example and averages examples. `embedding` and `last_hidden` representations are supported. `nast.signals.task_domain_alignment` returns cosine alignment mapped to `[0, 1]`.

## 5. Noise gate

`nast.selection.NASTSelector` creates criterion-specific masks and takes their union. CI uses a lower IQR fence, PU uses a fixed threshold, and TDA uses the lowest class from a deterministic NumPy Multi-Otsu implementation. As written in the manuscript, the TDA clustering population includes all valid sequence tokens even though only response tokens can be filtered. Very short or degenerate TDA vectors use a configured quantile fallback.

If every token is filtered, the selector deterministically recovers the strongest joint CI/PU/TDA token. This safety behavior avoids a zero-token loss and is recorded as `recovered_tokens`.

## 6. Gradient utility

The default `last_block` scorer freezes all model parameters for the scoring pass and inserts a stop-gradient leaf immediately before the final decoder block. L1 gradient magnitude with respect to that leaf is then computed from the full response NLL. This realizes “last block + output head” without a dense backbone backward pass. The checkpointing input-gradient hook is temporarily suspended so it cannot retain an accidental graph through earlier blocks.

Alternatives are `last_two_blocks`, `full_dense`, `output_head`, `nll`, and `none`. The full-dense mode places the saliency leaf before the first decoder block and is intentionally expensive.

## 7. Utility and adaptive budget

For non-noisy candidates:

```text
U = alpha * log(epsilon + position_normalized_CI) + beta * minmax(gradient)
r = clip(r_min + gamma * normalized_entropy(U) + delta * normalized_loss,
         r_min, r_max)
```

Stable descending sort gives deterministic top-k selection. Loss is normalized with an exponential running mean/variance and a logistic transform.
Utility entropy is the Shannon entropy of the within-instance softmax over candidate utilities, divided by `log(|C_i|)`. The paper does not define either normalization, so both choices are recorded as implementation assumptions.

## 8. Selective loss and token detachment

`nast.loss.selective_causal_lm_loss` applies the final binary mask after correct causal shifting and normalizes through mean cross entropy over selected targets.

`nast.pruning.TokenDetachContext` detaches unselected token representations at decoder-layer outputs while retaining their values in later forward context. See `MANUSCRIPT_AUDIT.md` for the distinction between this portable approximation and custom sparse activation kernels.

## 9. Metrics

- Full-response NLL/PPL: `nast.evaluation.evaluate_perplexity`
- Exact match: `nast.evaluation.evaluate_exact_match`
- MC conditional likelihood: `nast.evaluation.evaluate_multiple_choice`
- Online proxy GMR/overlap and full-dense audited GMR/overlap@k: `nast.metrics`, `nast.trainer.audit_selection`
- Measured CUDA allocation/reservation and throughput: `nast.profiling`
