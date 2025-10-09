#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="ai"
SRC_DIR="./data"
CLAIM_NAME="data"
POD_NAME="pvc-loader"
READ_ONLY=false
IMAGE="harbor.technopole-demo.it/ai/busybox:1.36"
TIMEOUT="60s"

usage() {
  echo "Usage: $0 [-n namespace] [-d dir] [-c claim] [-p pod] [-r] [--image IMG] [--timeout 180s]" >&2
  exit 1
}

# --- arg parsing ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) NAMESPACE="$2"; shift 2;;
    -d) SRC_DIR="$2"; shift 2;;
    -c) CLAIM_NAME="$2"; shift 2;;
    -p) POD_NAME="$2"; shift 2;;
    -r) READ_ONLY=true; shift;;
    --image) IMAGE="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

if [ "$READ_ONLY" = false ]; then
  [[ -d "$SRC_DIR" ]] || { echo "Directory '$SRC_DIR' not found"; exit 1; }
fi

cleanup() {
  echo "➜ Deleting pod $POD_NAME…"
  kubectl delete pod "$POD_NAME" -n "$NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- PVC precheck ---
echo "➜ Checking PVC '$CLAIM_NAME' in namespace '$NAMESPACE'…"
if ! kubectl get pvc "$CLAIM_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
  echo "❌ PVC '$CLAIM_NAME' not found in namespace '$NAMESPACE'"; exit 1
fi
PHASE="$(kubectl get pvc "$CLAIM_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}')"
if [[ "$PHASE" != "Bound" ]]; then
  echo "❌ PVC status is '$PHASE' (must be Bound)"; exit 1
fi

# --- create helper pod ---
echo "➜ Creating pod $POD_NAME in namespace $NAMESPACE (readOnly=$READ_ONLY, image=$IMAGE)"
kubectl delete pod "$POD_NAME" -n "$NAMESPACE" --ignore-not-found >/dev/null || true

RO_FLAG=$([ "$READ_ONLY" = true ] && echo "true" || echo "false")
OVERRIDES=$(cat <<EOF
{
  "apiVersion": "v1",
  "spec": {
    "restartPolicy": "Never",
    "volumes": [
      { "name": "pvc-vol", "persistentVolumeClaim": { "claimName": "${CLAIM_NAME}", "readOnly": ${RO_FLAG} } }
    ],
    "containers": [
      {
        "name": "shell",
        "image": "${IMAGE}",
        "imagePullPolicy": "IfNotPresent",
        "command": ["/bin/sh","-c","sleep 3600"],
        "volumeMounts": [ { "name": "pvc-vol", "mountPath": "/data", "readOnly": ${RO_FLAG} } ]
      }
    ]
  }
}
EOF
)

set +e
kubectl run "$POD_NAME" -n "$NAMESPACE" --image="$IMAGE" --restart=Never --overrides="$OVERRIDES" >/dev/null
RC=$?
set -e
if [[ $RC -ne 0 ]]; then
  echo "❌ kubectl run failed"; exit $RC
fi

# --- wait for readiness (and auto-diagnose on fail) ---
echo "➜ Waiting for pod readiness (timeout $TIMEOUT)…"
if ! kubectl wait --for=condition=Ready "pod/$POD_NAME" -n "$NAMESPACE" --timeout="$TIMEOUT"; then
  echo "❌ Pod not Ready. Events:"
  kubectl describe pod "$POD_NAME" -n "$NAMESPACE" | sed -n '/Events:/,$p'
  echo "ℹ️  Pod spec:"
  kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o yaml | sed -n '1,120p'
  exit 1
fi

# --- copy or inspect ---
if [ "$READ_ONLY" = true ]; then
  echo "➜ Read-only: listing /data"
  kubectl exec "$POD_NAME" -n "$NAMESPACE" -- sh -c 'ls -la /data || true'
  echo "ℹ️  Explore with: kubectl exec -it $POD_NAME -n $NAMESPACE -- sh"
else
  if command -v realpath >/dev/null 2>&1; then SRC_ABS="$(realpath "$SRC_DIR")"; else SRC_ABS="$(cd "$SRC_DIR" && pwd)"; fi
  echo "➜ Copying ${SRC_ABS} -> /data/"
  kubectl -n "$NAMESPACE" cp "${SRC_DIR}/." "$POD_NAME:/data/"
  echo "➜ chmod a+rX -R /data"
  kubectl exec "$POD_NAME" -n "$NAMESPACE" -- sh -c 'chmod -R a+rX /data || true'
fi

echo "✔️  Done! (pod will be removed automatically)"