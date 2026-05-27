#!/usr/bin/env bash
# Run from project root to see what you were last working on.

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

echo "━━━ Git status ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git -c color.status=always status --short | head -30

echo ""
echo "━━━ Local changes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
git diff --name-only

echo ""
echo "━━━ Recent commits (last 7 days) ━━━━━━━━━━━━━━━━━━━━━"
git log --oneline --since="7 days ago" --format="%C(yellow)%h%Creset  %C(cyan)%ar%Creset  %s" | head -10

echo ""
echo "━━━ Files touched in last commit ━━━━━━━━━━━━━━━━━━━━━"
git diff-tree --no-commit-id -r --name-only HEAD 2>/dev/null | head -20

echo ""
echo "━━━ Recently modified files (last 10 days) ━━━━━━━━━━━"
find . \
  -not -path './.git/*' \
  -not -path '*/__pycache__/*' \
  -not -name '*.pyc' \
  -not -path './.virtual_documents/*' \
  -not -path './.ipynb_checkpoints/*' \
  -mtime -10 \
  -type f \
  -printf "%T@ %Tb %Td %TH:%TM  %p\n" 2>/dev/null \
  | sort -rn | head -15 | awk '{$1=""; print substr($0,2)}'
