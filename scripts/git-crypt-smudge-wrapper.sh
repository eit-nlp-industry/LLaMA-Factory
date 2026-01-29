#!/bin/sh
# 保证 smudge 在仓库根、带 GIT_DIR 调用 git-crypt，否则子进程读不到密钥
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export GIT_DIR="$REPO_ROOT/.git"
cd "$REPO_ROOT" && exec git-crypt smudge "$@"
