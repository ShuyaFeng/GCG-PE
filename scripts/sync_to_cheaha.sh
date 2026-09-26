#!/bin/bash
# Push this pipeline to Cheaha over SSH, without going through GitHub.
#
#   bash scripts/sync_to_cheaha.sh            # preview, changes nothing
#   bash scripts/sync_to_cheaha.sh --go       # actually transfer
#   bash scripts/sync_to_cheaha.sh --go /data/user/fengs/GCG-PE
#
# Uses the `cheaha` host from ~/.ssh/config, so no hostname or username is
# needed here. Only source is copied: the virtualenv is architecture-specific
# and must be built on the cluster, and runs/ plus the HF cache stay put so a
# sync never overwrites results or re-uploads model weights.
#
# rsync is incremental, so re-running after an edit transfers only the diff.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

REMOTE_HOST="${REMOTE_HOST:-cheaha}"
GO=0
DEST=""
for arg in "$@"; do
  case "$arg" in
    --go)   GO=1 ;;
    -*)     echo "unknown flag: $arg"; exit 2 ;;
    *)      DEST="$arg" ;;
  esac
done
DEST="${DEST:-GCG-PE}"          # relative paths land in $HOME on the remote

EXCLUDES=(
  --exclude=.git                 # sync source, not history
  --exclude=.venv                # build this on the cluster, not here
  --exclude=__pycache__
  --exclude='*.pyc'
  --exclude=.hf_cache            # model/dataset cache: let the cluster fetch it
  --exclude=runs                 # results live on the cluster; never overwrite
  --exclude=logs
  --exclude=.DS_Store
)

echo "from : $PWD/"
echo "to   : ${REMOTE_HOST}:${DEST}/"
echo

if [ "$GO" -eq 0 ]; then
  echo "=== DRY RUN (nothing is transferred; add --go to do it) ==="
  rsync -avzn --delete-after "${EXCLUDES[@]}" ./ "${REMOTE_HOST}:${DEST}/"
  echo
  echo "Looks right? Re-run with:  bash scripts/sync_to_cheaha.sh --go"
  exit 0
fi

# --delete-after removes files on the remote that no longer exist locally, so a
# renamed or deleted module does not linger and get imported by accident. The
# excludes above protect runs/, logs/ and the caches from it.
rsync -avz --delete-after "${EXCLUDES[@]}" ./ "${REMOTE_HOST}:${DEST}/"
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
  echo
  echo "rsync exited ${STATUS}. If it was an auth failure, open a session first"
  echo "(Cheaha may require two-factor on the first connection):"
  echo "    ssh ${REMOTE_HOST}"
  exit "$STATUS"
fi

cat <<EOF

Transferred. On the cluster, first time only:

    ssh ${REMOTE_HOST}
    cd ${DEST}
    module load Python            # if this fails: module avail Python, then pick one
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

Then, every time:

    cd ${DEST}
    bash scripts/v3_locate.sh
    sbatch scripts/v3_theory.sbatch
    sbatch scripts/v3_diagnostics.sbatch
    sbatch scripts/v3_e5_memorization.sbatch
EOF
