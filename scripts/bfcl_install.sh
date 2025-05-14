#!/usr/bin/env bash
# install_bfcl.sh – clone BFCL at a fixed commit and install it
# into the currently-active BitAgent virtual-environment.

set -euo pipefail

# -------- settings ----------------------------------------------------------
REPO_URL="https://github.com/ShishirPatil/gorilla.git"
COMMIT="56d7a7c172ddbce25c1b9a6bd64acca7ed75063e"
EXTRAS=""            # e.g. set to "[oss_eval_sglang]" or "[oss_eval_vllm]" if desired
THIRD_PARTY_DIR="third_party"   # where the repo will live
# ---------------------------------------------------------------------------

# 1️⃣  Verify we’re inside the BitAgent venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: Activate BitAgent’s .venv first (source .venv/bin/activate)"; exit 1
fi

# 2️⃣  Pin the compatible NumPy version for both BitAgent and BFCL
python -m pip install --upgrade "numpy==2.0.1"

# 3️⃣  Fetch the exact Gorilla/BFCL commit
mkdir -p "$THIRD_PARTY_DIR"
cd "$THIRD_PARTY_DIR"

CLONE_DIR="gorilla_${COMMIT:0:7}"
if [[ ! -d "$CLONE_DIR" ]]; then
  git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
  cd "$CLONE_DIR"
  git fetch --depth 1 origin "$COMMIT"
  git checkout "$COMMIT"
else
  cd "$CLONE_DIR"
  git checkout "$COMMIT"
fi

# 4️⃣  Install BFCL editable into the current venv
cd berkeley-function-call-leaderboard
python -m pip install -e .${EXTRAS}

echo -e "\n✅  BFCL installed at commit $COMMIT inside $(basename "$VIRTUAL_ENV")"
