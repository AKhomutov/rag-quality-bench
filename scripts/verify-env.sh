#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCK_FILE="${ROOT}/infra/environment.lock"
COMPOSE_FILE="${ROOT}/infra/anythingllm/docker-compose.yml"
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

normalize_digest() {
  local value="$1"
  value="${value#sha256:}"
  printf 'sha256:%s\n' "$value"
}

read_lock_value() {
  # Usage: read_lock_value <python-expression-using-data-dict>
  local expr="$1"
  python3 - "$LOCK_FILE" "$expr" <<'PY'
import ast
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

value = eval(expr, {"__builtins__": {}}, {"data": data})  # noqa: S307
if isinstance(value, str):
    print(value)
else:
    print(ast.literal_eval(repr(value)))
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

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

actual_python_version() {
  local py_bin
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    py_bin="${ROOT}/.venv/bin/python"
  elif command_exists python; then
    py_bin="python"
  else
    py_bin="python3"
  fi
  "$py_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
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
  python3 - "$OLLAMA_HOST" "$model" <<'PY'
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

running_anythingllm_image() {
  if ! docker inspect anythingllm >/dev/null 2>&1; then
    echo "not-running"
    return 0
  fi

  python3 - <<'PY'
import json
import subprocess
import sys

proc = subprocess.run(
    ["docker", "inspect", "anythingllm"],
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

# Prefer RepoDigests from the image object for an immutable reference.
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
  python3 - "$url" <<'PY'
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=3) as resp:
        raise SystemExit(0 if 200 <= resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

require_file "$LOCK_FILE"
require_file "$COMPOSE_FILE"
require_file "$PYTHON_VERSION_FILE"

EXPECTED_PYTHON="$(read_lock_value 'data["python"]')"
EXPECTED_OLLAMA="$(read_lock_value 'data["ollama"]["version"]')"
EXPECTED_QWEN_DIGEST="$(normalize_digest "$(read_lock_value 'data["models"]["qwen3:8b"]["digest"]')")"
EXPECTED_NOMIC_DIGEST="$(normalize_digest "$(read_lock_value 'data["models"]["nomic-embed-text:latest"]["digest"]')")"
EXPECTED_ALLM_IMAGE="$(read_lock_value 'data["anythingllm"]["image"]')"
EXPECTED_ALLM_DIGEST="$(normalize_digest "$(read_lock_value 'data["anythingllm"]["digest"]')")"
EXPECTED_ALLM_REF="${EXPECTED_ALLM_IMAGE}@${EXPECTED_ALLM_DIGEST}"

FILE_PYTHON="$(tr -d '[:space:]' <"$PYTHON_VERSION_FILE")"
[[ "$FILE_PYTHON" == "$EXPECTED_PYTHON" ]] || die ".python-version (${FILE_PYTHON}) does not match environment.lock (${EXPECTED_PYTHON})"

echo "Environment verification"
echo

# Tooling availability
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
print_check "qwen3:8b" "$EXPECTED_QWEN_DIGEST" "$ACTUAL_QWEN" "  "

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
  # Accept either full repo digest ref or image@digest matching lock.
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
elif [[ "$RUNNING_REF" == "not-running" ]]; then
  echo "  AnythingLLM API: SKIP (stack not running)"
else
  if api_reachable "${ANYTHINGLLM_BASE_URL}/api/v1/system/ping"; then
    echo "  AnythingLLM API (${ANYTHINGLLM_BASE_URL}): PASS"
  else
    echo "  AnythingLLM API (${ANYTHINGLLM_BASE_URL}): FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
fi
echo

if [[ "$FAIL_COUNT" -ne 0 ]]; then
  echo "Environment verification FAILED (${FAIL_COUNT} check(s))"
  exit 1
fi

echo "Environment verification PASSED"
exit 0
