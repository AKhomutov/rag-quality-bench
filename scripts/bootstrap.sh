#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCK_FILE="${ROOT}/infra/environment.lock"
COMPOSE_FILE="${ROOT}/infra/anythingllm/docker-compose.yml"
PYTHON_VERSION_FILE="${ROOT}/.python-version"
REQUIREMENTS_LOCK="${ROOT}/requirements.lock"
ENV_TEMPLATE="${ROOT}/infra/anythingllm/env.template"
STORAGE_DIR="${ROOT}/infra/anythingllm/storage"
STORAGE_ENV="${STORAGE_DIR}/.env"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "required command not found: ${cmd}"
}

normalize_digest() {
  local value="$1"
  value="${value#sha256:}"
  printf 'sha256:%s\n' "$value"
}

read_lock_value() {
  local expr="$1"
  python3 - "$LOCK_FILE" "$expr" <<'PY'
import re
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
expr = sys.argv[2]
text = lock_path.read_text(encoding="utf-8")

data: dict[str, object] = {"models": {}, "ollama": {}, "anythingllm": {}}
current_section: str | None = None
current_model: str | None = None

for raw_line in text.splitlines():
    line = raw_line.split("#", 1)[0].rstrip()
    if not line.strip():
        continue

    m = re.match(r'^python:\s*"([^"]+)"\s*$', line)
    if m:
        data["python"] = m.group(1)
        current_section = None
        current_model = None
        continue

    m = re.match(r'^(\w+):\s*$', line)
    if m and not line.startswith(" "):
        current_section = m.group(1)
        current_model = None
        continue

    m = re.match(r'^  version:\s*"([^"]+)"\s*$', line)
    if m and current_section == "ollama":
        data["ollama"]["version"] = m.group(1)
        continue

    m = re.match(r'^  (.+):\s*$', line)
    if m and current_section == "models":
        current_model = m.group(1).strip()
        data["models"][current_model] = {}
        continue

    m = re.match(r'^    digest:\s*"([^"]+)"\s*$', line)
    if m and current_section == "models" and current_model is not None:
        data["models"][current_model]["digest"] = m.group(1)
        continue

    m = re.match(r'^  image:\s*"([^"]+)"\s*$', line)
    if m and current_section == "anythingllm":
        data["anythingllm"]["image"] = m.group(1)
        continue

    m = re.match(r'^  digest:\s*"([^"]+)"\s*$', line)
    if m and current_section == "anythingllm":
        data["anythingllm"]["digest"] = m.group(1)
        continue

print(eval(expr, {"__builtins__": {}}, {"data": data}))  # noqa: S307
PY
}

actual_python_version() {
  python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
}

actual_ollama_version() {
  local raw
  raw="$(ollama --version 2>&1)"
  if [[ "$raw" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    die "unable to parse ollama version from: ${raw}"
  fi
}

model_digest() {
  local model="$1"
  python3 - "$OLLAMA_HOST" "$model" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host = sys.argv[1].rstrip("/")
model = sys.argv[2]
try:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as resp:
        payload = json.load(resp)
except Exception as exc:  # noqa: BLE001
    print(f"ERROR: Ollama API unavailable at {host}: {exc}", file=sys.stderr)
    raise SystemExit(2)

for item in payload.get("models", []):
    if item.get("name") == model or item.get("model") == model:
        digest = item.get("digest")
        if not digest:
            print("ERROR: model has no digest", file=sys.stderr)
            raise SystemExit(2)
        digest = digest if str(digest).startswith("sha256:") else f"sha256:{digest}"
        print(digest)
        raise SystemExit(0)

raise SystemExit(1)
PY
}

ensure_model() {
  local model="$1"
  local expected_digest="$2"
  local actual_digest
  local status

  set +e
  actual_digest="$(model_digest "$model")"
  status=$?
  set -e

  if [[ "$status" -eq 2 ]]; then
    exit 1
  fi

  if [[ "$status" -ne 0 ]]; then
    info "Pulling missing Ollama model: ${model}"
    ollama pull "$model"
    actual_digest="$(model_digest "$model")" || die "failed to resolve digest for ${model} after pull"
  fi

  actual_digest="$(normalize_digest "$actual_digest")"
  if [[ "$actual_digest" != "$expected_digest" ]]; then
    die "model digest mismatch for ${model}
  expected: ${expected_digest}
  actual:   ${actual_digest}
Refusing to update environment.lock automatically."
  fi

  info "Model OK: ${model} (${actual_digest})"
}

prepare_storage() {
  mkdir -p "$STORAGE_DIR"
  if [[ ! -f "$STORAGE_ENV" ]]; then
    [[ -f "$ENV_TEMPLATE" ]] || die "missing AnythingLLM env template: ${ENV_TEMPLATE}"
    info "Creating ${STORAGE_ENV} from template"
    cp "$ENV_TEMPLATE" "$STORAGE_ENV"
    {
      echo "SIG_KEY='$(python3 -c 'import secrets; print(secrets.token_hex(32))')'"
      echo "SIG_SALT='$(python3 -c 'import secrets; print(secrets.token_hex(32))')'"
    } >>"$STORAGE_ENV"
    echo "NOTE: Create an AnythingLLM API key in the UI once, then put it in repo-root .env"
  else
    info "Keeping existing AnythingLLM storage env: ${STORAGE_ENV}"
  fi

  if [[ ! -f "${ROOT}/.env" ]]; then
    if [[ -f "${ROOT}/.env.example" ]]; then
      cp "${ROOT}/.env.example" "${ROOT}/.env"
      echo "NOTE: Created .env from .env.example — set ANYTHINGLLM_API_KEY before running evals"
    fi
  fi
}

[[ -f "$LOCK_FILE" ]] || die "missing ${LOCK_FILE}"
[[ -f "$REQUIREMENTS_LOCK" ]] || die "missing ${REQUIREMENTS_LOCK}"
[[ -f "$COMPOSE_FILE" ]] || die "missing ${COMPOSE_FILE}"
[[ -f "$PYTHON_VERSION_FILE" ]] || die "missing ${PYTHON_VERSION_FILE}"

info "Checking required commands"
require_cmd python
require_cmd ollama
require_cmd docker
docker compose version >/dev/null 2>&1 || die "required command not found: docker compose"

EXPECTED_PYTHON="$(tr -d '[:space:]' <"$PYTHON_VERSION_FILE")"
LOCK_PYTHON="$(read_lock_value 'data["python"]')"
[[ "$EXPECTED_PYTHON" == "$LOCK_PYTHON" ]] || die ".python-version (${EXPECTED_PYTHON}) != environment.lock (${LOCK_PYTHON})"

ACTUAL_PYTHON="$(actual_python_version)"
[[ "$ACTUAL_PYTHON" == "$EXPECTED_PYTHON" ]] || die "exact Python ${EXPECTED_PYTHON} required, found ${ACTUAL_PYTHON}"
info "Python OK: ${ACTUAL_PYTHON}"

if [[ ! -d "${ROOT}/.venv" ]]; then
  info "Creating .venv"
  python -m venv "${ROOT}/.venv"
else
  info "Reusing existing .venv"
fi

VENV_PY="${ROOT}/.venv/bin/python"
VENV_VERSION="$("$VENV_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
[[ "$VENV_VERSION" == "$EXPECTED_PYTHON" ]] || die ".venv Python is ${VENV_VERSION}, expected ${EXPECTED_PYTHON}"

info "Installing locked Python dependencies"
"$VENV_PY" -m pip install -r "$REQUIREMENTS_LOCK"
"$VENV_PY" -m pip install -e "${ROOT}" --no-deps

EXPECTED_OLLAMA="$(read_lock_value 'data["ollama"]["version"]')"
ACTUAL_OLLAMA="$(actual_ollama_version)"
[[ "$ACTUAL_OLLAMA" == "$EXPECTED_OLLAMA" ]] || die "Ollama ${EXPECTED_OLLAMA} required, found ${ACTUAL_OLLAMA}"
info "Ollama OK: ${ACTUAL_OLLAMA}"

if ! python3 - "$OLLAMA_HOST" <<'PY'
import sys
import urllib.request
host = sys.argv[1].rstrip("/")
urllib.request.urlopen(f"{host}/api/tags", timeout=5)
PY
then
  die "Ollama API is not reachable at ${OLLAMA_HOST}. Start the system Ollama service, then re-run bootstrap."
fi

EXPECTED_QWEN="$(normalize_digest "$(read_lock_value 'data["models"]["qwen3:8b"]["digest"]')")"
EXPECTED_NOMIC="$(normalize_digest "$(read_lock_value 'data["models"]["nomic-embed-text:latest"]["digest"]')")"
ensure_model "qwen3:8b" "$EXPECTED_QWEN"
ensure_model "nomic-embed-text:latest" "$EXPECTED_NOMIC"

ALLM_IMAGE="$(read_lock_value 'data["anythingllm"]["image"]')"
ALLM_DIGEST="$(normalize_digest "$(read_lock_value 'data["anythingllm"]["digest"]')")"
ALLM_REF="${ALLM_IMAGE}@${ALLM_DIGEST}"
COMPOSE_IMAGE="$(grep -E '^[[:space:]]*image:[[:space:]]*' "$COMPOSE_FILE" | head -n1 | sed -E 's/^[[:space:]]*image:[[:space:]]*//')"
[[ "$COMPOSE_IMAGE" == "$ALLM_REF" ]] || die "compose image (${COMPOSE_IMAGE}) does not match lock (${ALLM_REF})"

info "Pulling AnythingLLM image ${ALLM_REF}"
docker pull "$ALLM_REF"

info "Preparing local directories"
prepare_storage
mkdir -p "${ROOT}/reports"

info "Bootstrap complete"
echo "Next:"
echo "  1. Ensure ANYTHINGLLM_API_KEY is set in .env (create once in AnythingLLM UI if needed)"
echo "  2. ./scripts/start.sh"
echo "  3. ./scripts/verify-env.sh"
echo "  4. ./scripts/run-evals.sh"
