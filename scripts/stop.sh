#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="${ROOT}/infra/anythingllm/docker-compose.yml"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

[[ -f "$COMPOSE_FILE" ]] || die "missing ${COMPOSE_FILE}"

info "Stopping AnythingLLM compose stack"
docker compose -f "$COMPOSE_FILE" --project-directory "$(dirname "$COMPOSE_FILE")" down

info "AnythingLLM stack stopped"
echo "NOTE: system Ollama service was left running intentionally."
