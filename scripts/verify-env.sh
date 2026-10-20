#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCK_FILE="${ROOT}/infra/environment.lock"
COMPOSE_FILE="${ROOT}/infra/anythingllm/docker-compose.yml"
STORAGE_ENV="${ROOT}/infra/anythingllm/storage/.env"
PYTHON_VERSION_FILE="${ROOT}/.python-version"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
ANYTHINGLLM_BASE_URL="${ANYTHINGLLM_BASE_URL:-http://localhost:3001}"
# Set to 1 to skip runtime image/API checks while the stack is intentionally down.
VERIFY_SKIP_ANYTHINGLLM_RUNTIME="${VERIFY_SKIP_ANYTHINGLLM_RUNTIME:-0}"

FAIL_COUNT=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "missing required file: $path"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

resolve_python() {
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    printf '%s\n' "${ROOT}/.venv/bin/python"
    return 0
  fi
  if command_exists python; then
    command -v python
    return 0
  fi
  die "required command not found: python"
}

normalize_digest() {
  local value="$1"
  value="${value#sha256:}"
  printf 'sha256:%s\n' "$value"
}

compose() {
  docker compose -f "$COMPOSE_FILE" --project-directory "$(dirname "$COMPOSE_FILE")" "$@"
}

read_lock_value() {
  local expr="$1"
  "$PYTHON" - "$LOCK_FILE" "$expr" <<'PY'
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

print_check() {
  local title="$1"
  local expected="$2"
  local actual="$3"
  local indent="${4:-}"

  echo "${indent}${title}"
  echo "${indent}  expected: ${expected}"
  echo "${indent}  actual:   ${actual}"
  if [[ "$expected" == "$actual" ]]; then
    echo "${indent}  PASS"
  else
    echo "${indent}  FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
  echo
}

actual_python_version() {
  "$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
}

actual_ollama_version() {
  local raw
  raw="$(ollama --version 2>&1)"
  if [[ "$raw" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  else
    printf '%s\n' "$raw"
  fi
}

actual_model_digest() {
  local model="$1"
  "$PYTHON" - "$OLLAMA_HOST" "$model" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host = sys.argv[1].rstrip("/")
model = sys.argv[2]
try:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
        payload = json.load(resp)
except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
    print(f"unavailable ({exc})")
    raise SystemExit(0)

for item in payload.get("models", []):
    if item.get("name") == model or item.get("model") == model:
        digest = item.get("digest")
        if not digest:
            print("missing-digest")
            raise SystemExit(0)
        digest = digest if str(digest).startswith("sha256:") else f"sha256:{digest}"
        print(digest)
        raise SystemExit(0)

print("not-installed")
PY
}

compose_image_ref() {
  local line
  line="$(grep -E '^[[:space:]]*image:[[:space:]]*' "$COMPOSE_FILE" | head -n1 | sed -E 's/^[[:space:]]*image:[[:space:]]*//')"
  printf '%s\n' "$line"
}

compose_anythingllm_id() {
  compose ps -q anythingllm 2>/dev/null || true
}

running_anythingllm_image() {
  local container_id
  container_id="$(compose_anythingllm_id)"
  if [[ -z "$container_id" ]]; then
    echo "not-running"
    return 0
  fi

  "$PYTHON" - "$container_id" <<'PY'
import json
import subprocess
import sys

container_id = sys.argv[1]
proc = subprocess.run(
    ["docker", "inspect", container_id],
    check=False,
    capture_output=True,
    text=True,
)
if proc.returncode != 0:
    print("not-running")
    raise SystemExit(0)

info = json.loads(proc.stdout)[0]
image_id = info.get("Image", "")
config_image = info.get("Config", {}).get("Image", "")

img = subprocess.run(
    ["docker", "image", "inspect", image_id],
    check=False,
    capture_output=True,
    text=True,
)
if img.returncode == 0:
    digests = json.loads(img.stdout)[0].get("RepoDigests") or []
    for digest in digests:
        if digest.startswith("mintplexlabs/anythingllm@"):
            print(digest)
            raise SystemExit(0)
    if digests:
        print(digests[0])
        raise SystemExit(0)

if "@sha256:" in config_image:
    print(config_image)
elif image_id.startswith("sha256:"):
    print(f"mintplexlabs/anythingllm@{image_id}")
else:
    print(config_image or "unknown")
PY
}

api_reachable() {
  local url="$1"
  "$PYTHON" - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        raise SystemExit(0 if 200 <= resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

anythingllm_ollama_base_path() {
  [[ -f "$STORAGE_ENV" ]] || {
    echo ""
    return 0
  }
  "$PYTHON" - "$STORAGE_ENV" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"^OLLAMA_BASE_PATH=(.*)$", text, flags=re.MULTILINE)
if not match:
    print("")
    raise SystemExit(0)
print(match.group(1).strip().strip("'").strip('"'))
PY
}

check_container_ollama_connectivity() {
  local container_id
  local ollama_base

  container_id="$(compose_anythingllm_id)"
  if [[ -z "$container_id" ]]; then
    echo "  AnythingLLM → Ollama: SKIP (stack not running)"
    return 0
  fi

  ollama_base="$(anythingllm_ollama_base_path)"
  if [[ -z "$ollama_base" ]]; then
    echo "  AnythingLLM → Ollama: FAIL (OLLAMA_BASE_PATH missing in storage .env)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 0
  fi

  if docker exec "$container_id" curl -fsS --max-time 5 "${ollama_base%/}/api/tags" >/dev/null; then
    echo "  AnythingLLM → Ollama (${ollama_base}): PASS"
  else
    echo "  AnythingLLM → Ollama (${ollama_base}): FAIL"
    echo "    Ollama must listen on an address reachable from Docker (e.g. 0.0.0.0:11434)."
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
}

require_file "$LOCK_FILE"
require_file "$COMPOSE_FILE"
require_file "$PYTHON_VERSION_FILE"

PYTHON="$(resolve_python)"

EXPECTED_PYTHON="$(read_lock_value 'data["python"]')"
EXPECTED_OLLAMA="$(read_lock_value 'data["ollama"]["version"]')"
EXPECTED_QWEN_DIGEST="$(normalize_digest "$(read_lock_value 'data["models"]["qwen3:8b"]["digest"]')")"
EXPECTED_GEMMA_DIGEST="$(normalize_digest "$(read_lock_value 'data["models"]["gemma3:12b"]["digest"]')")"
EXPECTED_NOMIC_DIGEST="$(normalize_digest "$(read_lock_value 'data["models"]["nomic-embed-text:latest"]["digest"]')")"
EXPECTED_ALLM_IMAGE="$(read_lock_value 'data["anythingllm"]["image"]')"
EXPECTED_ALLM_DIGEST="$(normalize_digest "$(read_lock_value 'data["anythingllm"]["digest"]')")"
EXPECTED_ALLM_REF="${EXPECTED_ALLM_IMAGE}@${EXPECTED_ALLM_DIGEST}"

FILE_PYTHON="$(tr -d '[:space:]' <"$PYTHON_VERSION_FILE")"
[[ "$FILE_PYTHON" == "$EXPECTED_PYTHON" ]] || die ".python-version (${FILE_PYTHON}) does not match environment.lock (${EXPECTED_PYTHON})"

echo "Environment verification"
echo

echo "Tooling"
for cmd in python ollama docker; do
  if command_exists "$cmd"; then
    echo "  ${cmd}: PASS"
  else
    echo "  ${cmd}: FAIL (not found)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done
if docker compose version >/dev/null 2>&1; then
  echo "  docker compose: PASS"
else
  echo "  docker compose: FAIL (not available)"
  FAIL_COUNT=$((FAIL_COUNT + 1))
fi
echo

ACTUAL_PYTHON="$(actual_python_version)"
print_check "Python" "$EXPECTED_PYTHON" "$ACTUAL_PYTHON"

ACTUAL_OLLAMA="$(actual_ollama_version)"
print_check "Ollama" "$EXPECTED_OLLAMA" "$ACTUAL_OLLAMA"

echo "Models"
ACTUAL_QWEN="$(normalize_digest "$(actual_model_digest 'qwen3:8b')")"
print_check "qwen3:8b (SUT/generation)" "$EXPECTED_QWEN_DIGEST" "$ACTUAL_QWEN" "  "

ACTUAL_GEMMA="$(normalize_digest "$(actual_model_digest 'gemma3:12b')")"
print_check "gemma3:12b (historical comparison)" "$EXPECTED_GEMMA_DIGEST" "$ACTUAL_GEMMA" "  "

ACTUAL_NOMIC="$(normalize_digest "$(actual_model_digest 'nomic-embed-text:latest')")"
print_check "nomic-embed-text:latest" "$EXPECTED_NOMIC_DIGEST" "$ACTUAL_NOMIC" "  "

COMPOSE_REF="$(compose_image_ref)"
RUNNING_REF="$(running_anythingllm_image)"
echo "AnythingLLM"
echo "  expected image: ${EXPECTED_ALLM_REF}"
echo "  compose image:  ${COMPOSE_REF}"
if [[ "$COMPOSE_REF" == "$EXPECTED_ALLM_REF" ]]; then
  echo "  compose config: PASS"
else
  echo "  compose config: FAIL"
  FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [[ "$VERIFY_SKIP_ANYTHINGLLM_RUNTIME" == "1" ]]; then
  echo "  runtime image:  skipped"
  echo "  runtime check:  SKIP"
elif [[ "$RUNNING_REF" == "not-running" ]]; then
  echo "  runtime image:  not-running (skipped)"
  echo "  runtime check:  SKIP"
else
  echo "  runtime image:  ${RUNNING_REF}"
  if [[ "$RUNNING_REF" == "$EXPECTED_ALLM_REF" || "$RUNNING_REF" == *"${EXPECTED_ALLM_DIGEST}" ]]; then
    echo "  runtime check:  PASS"
  else
    echo "  runtime check:  FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
fi
echo

echo "API reachability"
if api_reachable "${OLLAMA_HOST}/api/tags"; then
  echo "  Ollama API (${OLLAMA_HOST}): PASS"
else
  echo "  Ollama API (${OLLAMA_HOST}): FAIL"
  FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [[ "$VERIFY_SKIP_ANYTHINGLLM_RUNTIME" == "1" ]]; then
  echo "  AnythingLLM API: SKIP (startup in progress)"
  echo "  AnythingLLM → Ollama: SKIP (startup in progress)"
elif [[ "$RUNNING_REF" == "not-running" ]]; then
  echo "  AnythingLLM API: SKIP (stack not running)"
  echo "  AnythingLLM → Ollama: SKIP (stack not running)"
else
  if api_reachable "${ANYTHINGLLM_BASE_URL}/api/v1/system/ping"; then
    echo "  AnythingLLM API (${ANYTHINGLLM_BASE_URL}): PASS"
  else
    echo "  AnythingLLM API (${ANYTHINGLLM_BASE_URL}): FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
  check_container_ollama_connectivity
fi
echo

if [[ "$FAIL_COUNT" -ne 0 ]]; then
  echo "Environment verification FAILED (${FAIL_COUNT} check(s))"
  exit 1
fi

echo "Environment verification PASSED"
exit 0
