#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

rm -rf docs/tutorial_files
uv run --extra lab jupyter nbconvert \
    --to markdown \
    --execute \
    --output-dir docs \
    --output tutorial \
    notebooks/tutorial.ipynb

# nbconvert's markdown exporter leaves trailing whitespace on some lines
# (e.g. blank lines padded around image/output blocks) - strip it so the
# generated file doesn't trip the repo's trailing-whitespace pre-commit hook.
sed -i 's/[ \t]*$//' docs/tutorial.md

echo "Regenerated docs/tutorial.md and docs/tutorial_files/"
