#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PYTHON="${ROOT}/.venv/bin/python"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "==> $*"
}

[[ -x "$VENV_PYTHON" ]] || die "missing ${VENV_PYTHON}. Run ./scripts/bootstrap.sh first."

info "Verifying environment"
"${ROOT}/scripts/verify-env.sh"

info "Running evaluations with ${VENV_PYTHON}"
# Preserve run_evaluations exit code (including known behavioral FAILs).
set +e
"$VENV_PYTHON" -m run_evaluations "$@"
status=$?
set -e

exit "$status"
