#!/bin/bash
# Deploy the dashboard to Cloudflare Pages.
#
# First-time setup (headless VM):
#   1. Create an API token at https://dash.cloudflare.com/profile/api-tokens
#      → "Create Token" → "Custom token" with these permissions:
#         - Account → Cloudflare Pages → Edit
#         - Account → Account Settings → Read (any account; needed for whoami)
#      → Copy the token.
#   2. Add to your .env file:    CLOUDFLARE_API_TOKEN=<token>
#                                CLOUDFLARE_ACCOUNT_ID=<your-account-id>
#      (Account ID is on the right sidebar of any zone in the Cloudflare dash,
#       or visible at the top of the API tokens page.)
#   3. Edit PROJECT_NAME below if you want a different subdomain.
#      Dashboard will be at https://<PROJECT_NAME>.pages.dev/dashboard.html
#
# Each subsequent run rebuilds the dashboard and uploads only changed files.
# URL stays stable across deploys.

set -e

PROJECT_NAME="sdf-inoculation"

cd "$(dirname "$0")/.."

# Load credentials from .env
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Ensure local node install is on PATH (installed by setup)
export PATH="$HOME/.local/node/bin:$PATH"

if ! command -v wrangler &> /dev/null; then
  echo "Error: wrangler is not installed. Run: npm install -g wrangler"
  exit 1
fi

if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
  echo "Error: CLOUDFLARE_API_TOKEN not set (add to .env)."
  exit 1
fi

echo "→ Regenerating dashboard..."
uv run python scripts/generate_dashboard.py

echo "→ Building staging dir (only files the dashboard fetches)..."
STAGING=$(mktemp -d)
trap "rm -rf $STAGING" EXIT

# Hardlinks — fast, no extra disk usage
cp -l outputs/dashboard.html "$STAGING/"

mkdir -p "$STAGING/plots"
cp -l outputs/plots/all_sweeps_comparison_*.png "$STAGING/plots/" 2>/dev/null || true

cd outputs
find runs \( \
    -name "*.json" -path "*/evals/petri/*/samples/*" -o \
    -name "transcript_*.json" -path "*/evals/petri_old/*" -o \
    -name "*.eval" -path "*/evals/mgs/*" -o \
    -name "summary.json" -path "*/evals/mgs/*" -o \
    -name "*.png" -path "*/plots/*" -o \
    -name "rollouts.json" \
  \) -print0 | while IFS= read -r -d '' f; do
  d="$STAGING/$(dirname "$f")"
  mkdir -p "$d"
  cp -l "$f" "$d/"
done
cd ..

# Include Inspect viewer bundle (static site for viewing Petri conversations)
if [ -d outputs/viewer ]; then
  echo "→ Including Inspect viewer bundle..."
  cp -rl outputs/viewer "$STAGING/viewer"
fi

# Cloudflare Pages worker (fixes HEAD requests for .eval files)
if [ -f outputs/_worker.js ]; then
  cp -l outputs/_worker.js "$STAGING/_worker.js"
fi

n_files=$(find "$STAGING" -type f | wc -l)
size=$(du -sh "$STAGING" | cut -f1)
echo "  staged $n_files files, $size"

echo "→ Deploying to Cloudflare Pages (project: $PROJECT_NAME)..."
wrangler pages deploy "$STAGING" \
  --project-name "$PROJECT_NAME" \
  --branch main \
  --commit-dirty true

echo
echo "✓ Deployed: https://${PROJECT_NAME}.pages.dev/dashboard.html"
