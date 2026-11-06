#!/usr/bin/env bash
set -euo pipefail

OPENCODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ollama create glm4-opencode:9b -f "$OPENCODE_ROOT/Modelfile"
