#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K3D_BIN="${K3D_BIN:-$LAB_ROOT/data/bin/k3d}"
CLUSTER_NAME=agent-lab
export KUBECONFIG="$LAB_ROOT/data/cluster/kubeconfig.yaml"
umask 077

if [[ ! -x "$K3D_BIN" ]]; then
    K3D_BIN="$(command -v k3d || true)"
fi
if [[ -z "$K3D_BIN" ]]; then
    echo 'Install k3d: brew install k3d' >&2
    exit 1
fi

write_kubeconfig() {
    mkdir -p "$(dirname "$KUBECONFIG")"
    "$K3D_BIN" kubeconfig get "$CLUSTER_NAME" > "$KUBECONFIG"
}

status() {
    kubectl --context k3d-agent-lab get nodes
    kubectl --context k3d-agent-lab -n agent-lab get deployment,pods,service
}

up() {
    docker info > /dev/null
    if docker container inspect "k3d-$CLUSTER_NAME-server-0" > /dev/null 2>&1; then
        "$K3D_BIN" cluster start "$CLUSTER_NAME" --wait --timeout 180s
    else
        "$K3D_BIN" cluster create --config "$LAB_ROOT/infra/k3d.yaml" --servers-memory 2g
    fi
    write_kubeconfig
    kubectl --context k3d-agent-lab wait --for=condition=Ready nodes --all --timeout=120s
    printf 'Kubeconfig: %s\n' "$KUBECONFIG"
}

deploy() {
    write_kubeconfig
    local endpoint
    endpoint="${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}"
    mkdir -p "$LAB_ROOT/data/docker-build"
    printf '{"auths":{},"cliPluginsExtraDirs":["/Applications/Docker.app/Contents/Resources/cli-plugins"]}\n' > "$LAB_ROOT/data/docker-build/config.json"
    docker --config "$LAB_ROOT/data/docker-build" --host "$endpoint" build -t agent-lab/app:dev "$LAB_ROOT/infra/app"
    "$K3D_BIN" image import agent-lab/app:dev -c "$CLUSTER_NAME"
    kubectl --context k3d-agent-lab apply -f "$LAB_ROOT/infra/workload.yaml"
    kubectl --context k3d-agent-lab apply -f "$LAB_ROOT/infra/rbac.yaml"
    local image_id
    image_id="$(docker image inspect agent-lab/app:dev --format '{{.Id}}')"
    kubectl --context k3d-agent-lab -n agent-lab patch deployment api --type=merge \
        -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"agent-lab/image-id\":\"$image_id\"}}}}}"
    kubectl --context k3d-agent-lab -n agent-lab rollout status deployment/api --timeout=120s
    status
}

case "${1:-status}" in
    up) up ;;
    deploy) deploy ;;
    status) write_kubeconfig; status ;;
    stop) "$K3D_BIN" cluster stop "$CLUSTER_NAME" ;;
    *) echo 'Usage: bash scripts/cluster.sh {up|deploy|status|stop}' >&2; exit 2 ;;
esac
