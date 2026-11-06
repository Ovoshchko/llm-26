#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ollama create glm4-agent:9b -f "$LAB_ROOT/infra/Modelfile"
