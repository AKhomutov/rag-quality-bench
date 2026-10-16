#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="${ROOT}/infra/anythingllm/docker-compose.yml"
STORAGE_ENV="${ROOT}/infra/anythingllm/storage/.env"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
ANYTHINGLLM_BASE_URL="${ANYTHINGLLM_BASE_URL:-http://localhost:3001}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-120}"
READY_INTERVAL_SECONDS="${READY_INTERVAL_SECONDS:-2}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

compose() {
  docker compose -f "$COMPOSE_FILE" --project-directory "$(dirname "$COMPOSE_FILE")" "$@"
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local elapsed=0

  info "Waiting for ${name} readiness (${url}), timeout ${READY_TIMEOUT_SECONDS}s"
  while (( elapsed < READY_TIMEOUT_SECONDS )); do
    if python3 - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        raise SystemExit(0 if 200 <= resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      info "${name} is ready"
      return 0
    fi
    sleep "$READY_INTERVAL_SECONDS"
    elapsed=$((elapsed + READY_INTERVAL_SECONDS))
  done

  die "${name} did not become ready within ${READY_TIMEOUT_SECONDS}s (${url})"
}

ensure_ollama() {
  if python3 - "$OLLAMA_HOST" <<'PY'
import sys
import urllib.request

host = sys.argv[1].rstrip("/")
urllib.request.urlopen(f"{host}/api/tags", timeout=3)
PY
  then
    info "Ollama API already available at ${OLLAMA_HOST}"
    return 0
  fi

  if command -v systemctl >/dev/null 2>&1; then
    local state
    state="$(systemctl is-active ollama 2>/dev/null || true)"
    if [[ "$state" == "active" ]]; then
      die "systemctl reports ollama active, but API is unreachable at ${OLLAMA_HOST}"
    fi
  fi

  die "Ollama is not reachable at ${OLLAMA_HOST}.
Start the Ollama service yourself (this script will not invoke sudo), then retry."
}

remove_foreign_anythingllm_container() {
  if ! docker inspect anythingllm >/dev/null 2>&1; then
    return 0
  fi

  local project
  project="$(docker inspect anythingllm --format '{{index .Config.Labels "com.docker.compose.project"}}' 2>/dev/null || true)"
  if [[ "$project" == "rag-quality-bench" ]]; then
    return 0
  fi

  info "Removing existing non-compose AnythingLLM container so compose can manage it"
  docker stop anythingllm >/dev/null
  docker rm anythingllm >/dev/null
}

[[ -f "$COMPOSE_FILE" ]] || die "missing ${COMPOSE_FILE}"
[[ -f "$STORAGE_ENV" ]] || die "missing ${STORAGE_ENV}. Run ./scripts/bootstrap.sh first."

info "Verifying environment contract (AnythingLLM runtime skipped until stack is up)"
VERIFY_SKIP_ANYTHINGLLM_RUNTIME=1 "${ROOT}/scripts/verify-env.sh"

ensure_ollama
remove_foreign_anythingllm_container

info "Starting AnythingLLM via docker compose"
compose up -d

wait_for_url "Ollama" "${OLLAMA_HOST}/api/tags"
wait_for_url "AnythingLLM" "${ANYTHINGLLM_BASE_URL}/api/v1/system/ping"

info "Running final environment verification"
"${ROOT}/scripts/verify-env.sh"

info "Stack is up"
echo "AnythingLLM: ${ANYTHINGLLM_BASE_URL}"
echo "Ollama:      ${OLLAMA_HOST}"
