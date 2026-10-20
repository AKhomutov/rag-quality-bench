# Evaluator LLM audit

Which inference call sites exist in this repo, and which scoring jobs stay on
an LLM versus deterministic code.

Related: how the long-form evaluator evolved —
[evaluator-architecture.md](evaluator-architecture.md). Overview —
[README.md](../README.md).

**Rule of thumb:** use an LLM only when the operation needs semantic judgement.
Code owns orchestration, schema validation, traces, aggregation, graph
normalization, and verdicts.

## Call-site inventory (evaluator-relevant)

| # | Location | Backend | Role | Type | Decision |
|---|----------|---------|------|------|----------|
| 1 | `evaluators/baseline_evaluator.py` → `chat` | AnythingLLM → qwen3:8b | SUT | A | keep SUT |
| 2 | `evaluators/adversarial_evaluator.py` → `thread_chat` | AnythingLLM → qwen3:8b | SUT | A | keep SUT |
| 3 | `evaluators/conflicting_evidence_evaluator.py` → `thread_chat` | AnythingLLM → qwen3:8b | SUT | A | keep SUT |
| 4 | `evaluators/agent_tool_selection_evaluator.py` → `agent_thread_chat` | AnythingLLM → qwen3:8b | SUT | A | keep SUT |
| 5 | Agent tool-name parse + pass-rate | Python | evaluator | G, F, E | **code** |
| 6 | Baseline answer/source oracles | Python | evaluator | H, E | **code** |
| 7 | `run_judge_prototype` / `run_judge_stability` → `OllamaClient.generate` | Ollama qwen3:8b | semantic judge | C | keep LLM |
| 8 | Split pipeline Stage 1 extraction | Ollama qwen3:8b | claim extract | B | keep LLM |
| 9 | Split pipeline Stage 2 base batches | Ollama qwen3:8b | evidence judge | C | keep LLM |
| 10 | Split pipeline Stage 2B dependency batches | Ollama qwen3:8b | dependency | D | keep LLM for now |
| 11 | Split pipeline aggregation / verdict | Python | evaluator | F, E, I | **code** |
| 11b | Split pipeline dependency-graph normalization (SCC) | Python | evaluator | F | **code** |
| 12 | `run_long_form_severity_eval` SUT + gemma judge | AnythingLLM + Ollama gemma3:12b | historical | A+C | experiment only |
| 13 | `run_long_form_judge_structured` | Ollama gemma3:12b | historical | C | experiment only |
| 14 | `run_long_form_judge_diagnostic` | Ollama gemma3:12b | historical | C+J | experiment only |
| 15 | `run_long_form_judge_qwen_16k` | Ollama qwen3:8b | historical | C | experiment only |
| 16 | `OllamaClient.unload_model` | Ollama | lifecycle | J | **code** |
| 17 | `workspace_setup` embeddings | AnythingLLM → nomic-embed-text | setup | J | keep (infra) |

Type legend: A SUT · B claim extraction · C evidence judgement · D dependency ·
E validation · F aggregation · G tool-trace · H exact oracle · I reporting ·
J lifecycle.

Active evaluator-side LLM scoring paths: suite semantic judge
(`run_judge_prototype` / `run_judge_stability`); split extract / base /
dependency. Historical rows 12–15 are not in the normal `run_evaluations` flow.

SUT call sites: four suite runners (plus optional live SUT in the historical
long-form eval).

## Decision table

| Area | LLM responsibility | Decision | Reason |
|------|--------------------|----------|--------|
| Baseline factual / abstention | none for scoring | code | exact answer/source oracles |
| Agent tool trace | none | code | authoritative thought telemetry |
| Agent pass-rate / threshold | none | code | numeric aggregation |
| Forged policy / severity spoof | adoption/rejection/provenance | keep LLM | meaning + paraphrase |
| Conflicting refund policies | conflict *existence* | **narrowed** | fixture already defines conflict + no precedence |
| Conflicting refund policies | answer acknowledgment / invented precedence | keep LLM | meaning required |
| Claim extraction | material claim segmentation | keep LLM | semantic decomposition |
| Per-claim evidence | support vs contradict vs not_in_evidence | keep LLM | NL vs policy |
| Claim dependency | derived-from-unsupported | keep LLM for now | semantic relationship; no safe heuristic |
| Schema / fact IDs / verdict | none | code | deterministic |
| Dependency-graph cycle normalization | none | code | SCC on raw edges; normalized graph → final metrics |
| Monolithic gemma / qwen 16k runners | full claim audit | historical | comparison/debug only |

## Changes made in this audit pass

1. **Narrowed** `conflicting_refund_policies` `case_instruction` so the judge
   does not rediscover fixture-known conflict existence; it only judges answer
   acknowledgment / invented precedence
   (`fixtures/datasets/semantic_judge_specs.yaml`).
2. **Documented** historical long-form/monolithic runners as non-production
   experiments (module docstrings).
3. **Added** unit tests for tool-trace parsing and pass-rate aggregation.
4. **No LLM call sites removed** from production paths — those paths were
   already deterministic where safe.

## LLM responsibilities removed / not removed

| Removed from LLM | Status |
|------------------|--------|
| Tool-trace classification | already code |
| Pass-rate / threshold | already code |
| Baseline exact oracles | already code |
| Split-claim aggregation / verdict | already code |
| Conflict *existence* rediscovery | narrowed out of judge instruction |

| Remaining semantic LLM work | Why kept |
|-----------------------------|----------|
| Semantic 4-dim judge on adversarial/conflict answers | paraphrase / provenance |
| Split Stage 1 claim extraction | semantic segmentation |
| Split Stage 2 evidence labels | NL vs authoritative policy |
| Split Stage 2B dependency | semantic implication; heuristic would be brittle |

## Intentionally retained (would be brittle as code)

- Forged-policy / severity spoof rejection via substring lists.
- Conflict acknowledgment via phrase matching (`"not mentioned"`,
  `"cannot determine"`, etc.).
- Stage 2B dependency via topic-similarity heuristics.

## Historical / out of normal flow

Not the current evaluator architecture:

- `run_long_form_severity_eval`
- `run_long_form_judge_structured`
- `run_long_form_judge_diagnostic`
- `run_long_form_judge_qwen_16k`
- `run_split_claim_graph_offline`

Current long-form path: `run_split_claim_pipeline` /
`run_split_claim_evaluation`, suite `long_form_claim_audit`, opt-in via
`run_evaluations --include-long-form` or
`./scripts/run-evals.sh --include-long-form`. Standalone module remains for
focused debugging. Evolution and experiment numbers:
[evaluator-architecture.md](evaluator-architecture.md).

Default `run_evaluations`: baseline + agent (deterministic scorers).
Adversarial/conflicting are semantic-only (`run_judge_prototype` /
`run_judge_stability`). Long-form runs only with `--include-long-form`.

## Future candidate (not implemented)

Stage 2B dependency may later use structured claim relations if extraction
emits explicit dependency edges — without guessing from free text.
