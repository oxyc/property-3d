#!/bin/sh
# Publish site/models as the only commit of the site-models branch, replacing the previous one so model
# versions don't pile up in history, then redeploy Pages (the workflow checks that branch out into site/models).
#
# usage: pipeline/push_models.sh [remote]   (default origin)
set -e
repo=$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)
remote=${1:-origin}
index=$(mktemp -d)/index
cd "$repo/site/models"
GIT_INDEX_FILE=$index git --git-dir="$repo/.git" --work-tree=. add -A -- . ':!.DS_Store'
tree=$(GIT_INDEX_FILE=$index git --git-dir="$repo/.git" write-tree)
commit=$(git --git-dir="$repo/.git" commit-tree --no-gpg-sign "$tree" -m "Published models")
rm -rf "$(dirname "$index")"
git --git-dir="$repo/.git" push --force "$remote" "$commit:refs/heads/site-models"
gh workflow run pages.yml -R oxyc/property-3d --ref main
