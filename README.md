# RAG Quality Bench

Python evaluation harness for RAG and agent behavior on AnythingLLM with local
Ollama models. It exercises grounding, false premises, abstention, policy
poisoning, conflicting evidence, agent tool selection, and (opt-in) long-form
mixed-truth claim auditing.

## What it evaluates

| Scenario | What is checked | Oracle |
| --- | --- | --- |
| Factual RAG | Answer matches embedded workspace docs | Deterministic |
| False premise | Rejects planted false facts | Deterministic |
| Abstention | Declines when the corpus has no answer | Deterministic |
| Forged policy / conversation poisoning | User text does not override trusted policy | Semantic LLM judge |
| Conflicting authoritative documents | Surfaces conflict; invents no precedence | Semantic LLM judge |
| Agent source priority | Prefers workspace RAG over web when the corpus has the answer | Tool trace + pass-rate |
| Long-form mixed-truth policy poisoning | Claim-level audit of a poisoned severity ontology | Split claim audit (opt-in) |

Default stack: AnythingLLM, Ollama, `qwen3:8b` (generation / judges),
`nomic-embed-text`, LanceDB.

## Evaluation model

Use deterministic code wherever possible. Call an LLM only for the smallest
operation that actually needs semantic judgement.

**Code owns:** orchestration, schema validation, tool-trace parsing, exact IDs,
thresholds, aggregation, dependency-graph normalization, final verdict,
reporting.

**LLM is used for:** semantic claim extraction, evidence support judgement, and
unresolved semantic dependency. Shorter adversarial/conflicting cases use the
suite semantic judge; long-form uses the split claim pipeline.

Call-site inventory: [docs/evaluator-llm-audit.md](docs/evaluator-llm-audit.md).

## Architecture

```text
fixtures/          cases, documents, pinned prompts
src/
  clients/         AnythingLLM + Ollama clients
  evaluators/      suite runners
  judges/          semantic + claim-level prompts/schemas
scripts/           bootstrap, start/stop, run-evals, verify-env
docs/              architecture + LLM-responsibility audit
tests/
infra/             environment.lock, AnythingLLM compose
```

Temporary workspaces are created through the AnythingLLM API and removed after
each run.

## Key findings

Several cases came from failures found while using the system by hand.

| Finding | Observed failure | How it is evaluated |
| --- | --- | --- |
| Forged refund policy | User-supplied “Exceptional Renewal Refund” text was treated as policy and used to approve a refund | Semantic judge: forged rules must not override workspace policy |
| Conversation poisoning / reattribution | After rejecting Severity 0.5, the model accepted a fabricated Severity 0.5 section and later listed it as coming from `company_policy` | Multi-turn replay + semantic judge on provenance |
| Conflicting authoritative documents | Two embedded refund policies contradict; the model invented a precedence rule present in neither | Semantic judge: surface both claims; no invented precedence |
| Agent prefers web over workspace RAG | Question answerable from the corpus; agent sometimes called `web-browsing` and answered from unrelated external text | Tool-trace check: a correct “30 days” answer still fails if the path was web. Stochastic — repeated trials with a pass-rate threshold |

Agent cases treat a wrong execution path as failure even when the final text
happens to look right.

## Long-form claim audit

Opt-in suite for one long mixed-truth severity-policy answer
(`mixed_truth_severity_policy_poisoning`).

A monolithic judge on this prompt exposed evaluator limits: too many jobs in one
call, and an implicit Ollama ~4K context budget that truncated or distorted
judgement. Explicit `CLAIM_LEVEL_NUM_CTX=16384` was required. A Qwen 16K control
showed that earlier weak-looking Qwen results were confounded by that budget;
`gemma3:12b` was a larger independent local comparison model for the prototype
only.

The current path decomposes the audit:

```text
claim extraction → batched evidence → batched dependency
→ graph normalization → deterministic aggregation
```

Judge model: `qwen3:8b`. Batching cut wall time from ~549s / 70 calls to
~176.7s / 11 calls (~3.1×). Fine-grained labels can shift across runs; the
high-level FAIL and rejection of poisoned claims stayed stable.

Full chronology, numbers, and limitations:
[docs/evaluator-architecture.md](docs/evaluator-architecture.md).

## Running the evaluations

```bash
./scripts/bootstrap.sh
./scripts/start.sh
# First-time AnythingLLM setup: create an API key, put it in .env
./scripts/verify-env.sh
```

```bash
# Default consolidated run (baseline + agent; fast path)
./scripts/run-evals.sh

# Also run the slower long-form split claim audit
./scripts/run-evals.sh --include-long-form

# Shorter semantic judge prototype / stability (adversarial + conflicting)
python -m run_judge_prototype
python -m run_judge_stability

# Focused long-form split pipeline (same implementation as --include-long-form)
python -m run_split_claim_pipeline
```

**Default `./scripts/run-evals.sh`:** baseline RAG and agent tool-selection with
deterministic scorers. Adversarial and conflicting-evidence cases are listed in
the runner output as semantic-only; they are scored via `run_judge_prototype` /
`run_judge_stability`, not included in consolidated pass/fail totals.

**`--include-long-form`:** appends suite `long_form_claim_audit` to the same
consolidated reports. Multiple local inference calls; opt-in on purpose.

Known behavioral failures may yield a non-zero exit code.

```bash
./scripts/stop.sh   # stops AnythingLLM; Ollama is left running
```

## Reports

```text
reports/evaluation_report.json
reports/evaluation_report.md
reports/judge_debug/
```

Consolidated JSON/Markdown stay concise (suite status, metrics, short failure
lists). Full prompts, claim dumps, and raw judge responses stay under
timestamped `reports/judge_debug/`. Report and debug trees are gitignored.

## Local setup

Pinned contract: [infra/environment.lock](infra/environment.lock)
(Python, Ollama, model digests, AnythingLLM image digest). Python deps:
`requirements.lock`.

```text
ANYTHINGLLM_BASE_URL=http://localhost:3001
ANYTHINGLLM_API_KEY=your_api_key_here
```

`bootstrap.sh` may create `.env` from `.env.example`; the API key is filled in
after AnythingLLM is running. Workspaces and documents are managed through the
API.

## Reproducibility

- Python, Ollama, and digests for `qwen3:8b`, `gemma3:12b` (historical),
  `nomic-embed-text`, and the AnythingLLM image — see `infra/environment.lock`.
- Ollama has no reliable pull-by-digest; bootstrap fails if a tag resolves to a
  different artifact.
- Long-form: `CLAIM_LEVEL_NUM_CTX=16384`, SHA-pinned prompt, SHA-pinned fixed
  candidate (details in
  [docs/evaluator-architecture.md](docs/evaluator-architecture.md)).
- Model behavior is stochastic; agent cases use repeated trials. Batching can
  shift fine-grained long-form labels while keeping the high-level FAIL.
- Runtime storage, vectors, chat history, and API keys are not committed.

## Development

```bash
.venv/bin/pytest -v
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy src tests
```

## Tech stack

Python 3.14, pytest, AnythingLLM, Ollama, Qwen3, Nomic Embed Text, LanceDB,
Ruff, mypy.

## Further reading

- [Evaluator architecture and experiments](docs/evaluator-architecture.md) —
  monolithic → split pipeline, context budget, batching, graph normalization
- [Evaluator LLM audit](docs/evaluator-llm-audit.md) — which responsibilities
  stay on an LLM vs deterministic code
- [Environment lock](infra/environment.lock)
