#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LAB_ROOT"
exec data/bin/kubernetes-mcp-server --config infra/kubernetes-mcp.toml
