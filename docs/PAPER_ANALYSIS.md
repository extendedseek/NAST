# Deep paper analysis

## Executive assessment

NAST proposes a sensible two-stage policy: remove response tokens that look contextually weak, already known, or off-domain; then spend a bounded backward budget on candidates with high contextual and gradient utility. The framing is useful because it separates **supervision quality** from **backward capacity**. The paper also asks the right systems question: label masking alone does not make dense Transformer activations disappear.

The method is implementable at the objective and selector level. The headline memory result is not independently reproducible from the manuscript, because the backbone, exact training setup, data construction, and graph-pruning backend are absent. One reported token-ratio row is also inconsistent with the published budget equation.

## Reconstructed algorithm

For each instruction/response sequence:

1. Run the full causal forward context.
2. Compute received-attention Contextual Influence (CI), gold-token Predictive Uncertainty (PU), and prototype cosine Task-Domain Alignment (TDA).
3. Mark a response token noisy if it falls below any active criterion threshold.
4. For surviving candidates, compute

   ```text
   U(t) = alpha * log(epsilon + normalized_CI(t))
          + beta * normalized_L1_gradient(t)
   ```

5. Allocate an instance-specific ratio

   ```text
   r_i = clip(r_min + gamma * entropy(U_i) + delta * normalized_loss_i,
              r_min, r_max)
   ```

6. Select the top `ceil(r_i * |C_i|)` candidates, compute mean causal NLL only on them, and prune/detach the remaining backward paths while retaining all tokens as forward context.

The repo implements each stage separately so filtering, ranking, budgeting, loss masking, and graph behavior can be tested independently.

## What is genuinely strong

- **Filter then rank:** high loss or gradient magnitude alone can favor corrupted/off-task targets. The prior noise gate is a defensible safeguard.
- **Instance-aware budgets:** a bounded entropy/loss policy is more plausible for heterogeneous instruction data than one global token percentage.
- **Full-response evaluation:** evaluating PPL on every gold response token avoids the selected-token evaluation bias common to masked objectives.
- **Systems honesty in the formulation:** the paper explicitly distinguishes loss masking from activation handling.
- **Interpretability surface:** criterion-specific masks make it possible to inspect why a token was rejected before top-k ranking.

## Main validity risks

### 1. Selected-ratio contradiction

The paper states `k_i = ceil(r_i |C_i|)` with `r_i <= 0.50`. Its threshold table reports 53.2% noise and 46.8% selected response tokens. A 53.2% noise rate leaves 46.8% candidates, so the stated equation allows at most approximately `0.50 * 46.8% = 23.4%` selected response tokens, aside from small ceiling effects. The reported 46.8% instead implies selecting essentially every candidate.

This is not a cosmetic ambiguity: it changes the training objective, memory curve, and interpretation of all selected-ratio tables. The repo defaults to the equation and isolates the response-denominator alternative in `reported-ratio-compat.yaml`.

### 2. The activation-memory claim needs missing systems code

Selected labels do not stop dense attention/MLP kernels from saving intermediate tensors. Layer-boundary detachment blocks gradients into earlier representations at unselected positions, but parameter gradients can still depend on unselected key/value context used by selected queries. Exact selected-token activation caching therefore needs an architecture-aware split-stream/autograd implementation or custom kernels.

The closest public evidence is architecture-specific: the [TokenTune repository](https://github.com/facebookresearch/tokentune) uses a modified Llama path rather than a generic label mask. NAST's manuscript gives no equivalent code. The portable backend here is an explicit ablation, and only measured profiler output may be used for claims.

### 3. The signal pass is not free

NAST requires attention tensors, base/reference probabilities, semantic representations, and a gradient proxy before the training backward pass. At long sequence lengths, materializing attention maps can itself be expensive. A fair throughput number must include this pass, synchronization, selection on the host/device, and prototype-related work. The repo profiler includes selection scoring and reports scoring, forward, backward, optimizer, and total time separately.

### 4. Attention is an imperfect causal explanation

Received attention is a useful structural heuristic, but high attention is not proof that changing a token causes the output. Position normalization also lacks an operational definition in the paper: the denominator is written as a conditional expectation without saying how it is estimated. The implementation uses the count of valid future causal queries as a deterministic opportunity correction and labels this as an assumption.

### 5. A single task prototype can collapse a heterogeneous mixture

One mean vector across instruction following, mathematics, science, and commonsense may lie between modes and penalize legitimate domain-specific tokens. Per-dataset, per-task, clustered, or learned prototypes are important missing comparisons. The paper does not state whether prototypes are global, task-specific, input-only, response-only, or layer-specific.

### 6. PU can remove useful calibration targets

Filtering low-uncertainty tokens saves budget, but easy tokens can stabilize style, syntax, calibration, and catastrophic-forgetting behavior. A fixed `1-p < 0.05` rule is also not equivalent in scale to an NLL threshold of 0.05. The manuscript permits both forms without reporting which generated the tables.

### 7. Union gating can be brittle

The union rule rejects a token when any one signal is low. That is interpretable but sensitive to a poorly calibrated criterion—especially Multi-Otsu on short or unimodal sequences. The implementation defines a deterministic short-sample fallback and records forced recovery if a response loses every candidate; neither edge case is specified in the paper.

### 8. Reported comparisons lack statistical support

The manuscript provides point estimates but no seeds, run counts, standard deviations, confidence intervals, or significance tests. Several claimed improvements are small enough that run-to-run variation could reverse their ordering. Every final table should report all seeds and uncertainty.

## Numerical consistency checks

- A 21.5 GB NAST peak with 51.36% memory reduction implies a dense peak of about `21.5 / (1 - 0.5136) = 44.20 GB`, consistent with the later 44.2 GB dense figure.
- The selected-ratio/noise-ratio row is not consistent with the candidate-denominator budget, as shown above.
- Memory reduction is not expected to equal `1 - selected_ratio`: weights, optimizer state, allocator behavior, attention states, scoring, and non-token activations remain dense.
- “46.8% selected tokens” must say whether the denominator includes EOS, formatting, punctuation, padding, prompts, or only eligible response targets.

## Reproducibility status

| Component | Status from manuscript | Repository treatment |
|---|---|---|
| CI/PU/TDA formulas | mostly specified | implemented with explicit edge policies |
| Noise union | specified | implemented and tested |
| Utility/budget form | specified | implemented; coefficients exposed |
| Backbone/tokenizer | missing | transparent Qwen default, not claimed as original |
| LoRA details | missing | explicit configurable defaults |
| Data revisions/mixture/splits | missing | adapters plus clearly labeled assumed policy |
| Exact answer prompts/extraction | missing | native evaluator and documented rules |
| Dense-reference GMR | underspecified operationally | separate expensive audit command |
| Sparse graph backend | missing | portable detach ablation plus hard limitation |
| Random seeds/uncertainty | missing | deterministic seed field and multi-seed protocol |
| Claimed checkpoints/logs | unavailable | no fabricated artifacts included |

## Highest-value experiments before publication

1. Resolve the candidate-versus-response budget denominator and regenerate all token-ratio rows.
2. Release the exact graph-pruning kernel/model patch and compare it with loss-only masking and portable detachment under the same profiler.
3. Report end-to-end and phase timing with the scoring pass included.
4. Run at least three preregistered seeds and report uncertainty.
5. Compare global, per-family, and clustered TDA prototypes.
6. Calibrate each gate on validation data and publish the actual cutoffs for every threshold ablation.
7. Measure full-dense GMR on a fixed audit subset rather than using the selection proxy as its own reference.
8. Add contamination and deduplication analysis, especially across instruction mixtures and the reasoning/knowledge evaluations.
9. Pin model, tokenizer, dataset, package, CUDA, and repository revisions.
10. Validate the method on at least two backbone families to separate method effects from architecture-specific graph behavior.

## Bottom line

NAST is a promising selection policy and a plausible research direction, but the paper currently supports a **reference implementation**, not a verified reproduction of its headline systems results. The repository is intentionally designed around that distinction: executable formulas and datasets where possible, explicit assumptions where necessary, and no synthetic evidence where author code or experimental identity is missing.
