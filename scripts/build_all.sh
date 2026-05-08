#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Building C++"
(cd "${REPO_ROOT}/c++" && ./build.sh)

echo "==> All targets built successfully"
