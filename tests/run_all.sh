#!/usr/bin/env bash
# Run the full data-integrity verification suite (all 5 tests/verify_*.py
# scripts) in sequence, fail-fast on the first non-zero exit.
#
# Required after any change to scripts/utils.py, scripts/config.py, or any
# step0*.py before merging (see README.md / DATA_INTEGRITY_PM_PLAN.md Phase B).
#
# Usage: bash tests/run_all.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # project root

SCRIPTS=(
  tests/verify_preprocessing.py
  tests/verify_integrity.py
  tests/verify_branch.py
  tests/verify_analysis.py
  tests/verify_practical_audit.py
)

echo "=== Data-integrity verification suite (${#SCRIPTS[@]} scripts) ==="
for s in "${SCRIPTS[@]}"; do
  echo ""
  echo "--- $s ---"
  if ! python3 "$s"; then
    echo ""
    echo "FAILED: $s — stopping (fail-fast). Fix before continuing."
    exit 1
  fi
done

echo ""
echo "=== ALL ${#SCRIPTS[@]} VERIFICATION SCRIPTS PASSED ==="
