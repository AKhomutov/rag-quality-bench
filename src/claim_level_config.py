"""Configuration for the long-form claim-level judge only.

The existing simple semantic judge continues to use JUDGE_MODEL (qwen3:8b).
"""

from __future__ import annotations

CLAIM_LEVEL_JUDGE_MODEL = "gemma3:12b"
CLAIM_LEVEL_JUDGE_MODEL_DIGEST = (
    "sha256:b92b218582c521ba287f38f1d6057dcd3c6cbff036f13cac97b5d861f5e58ec3"
)
CLAIM_LEVEL_JUDGE_KEEP_ALIVE: int = 0
CLAIM_LEVEL_JUDGE_TEMPERATURE = 0
# Runtime capacity for full claim-level prompt + long structured audit output.
CLAIM_LEVEL_NUM_CTX = 16384

# Split claim pipeline keeps the model loaded across many sequential calls,
# then unloads once at the end via OllamaClient.unload_model.
# Ollama accepts integer -1 (indefinite); string "-1" is rejected.
SPLIT_CLAIM_PIPELINE_KEEP_ALIVE: int = -1
SPLIT_CLAIM_PIPELINE_TEMPERATURE = 0

# SUT / generation model (also used for Qwen claim-level A/B controls / split pipeline).
SUT_GENERATION_MODEL = "qwen3:8b"
SUT_GENERATION_MODEL_DIGEST = (
    "sha256:500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"
)

EXPECTED_FIXED_CANDIDATE_SHA256 = "01f683262d203f0fb610f2c587d86c55c5197a70a6596937c30d949c88124407"
EXPECTED_FIXED_CANDIDATE_LENGTH = 6452
