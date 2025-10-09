#!/usr/bin/env bash
# pipeline.sh — interactively build & push selected images to Harbor and deploy
# Requires: docker, helm, kubectl
#
# Env (must be set):  HARBOR_USER, HARBOR_PASS
# Fixed config (edit here): REGISTRY, PROJECT, CHART_DIR, CHART_NAME
# Namespace flag: -n|--namespace (default: ai)

set -euo pipefail

# ---------- fixed config (edit for your env) ----------
APPS_DIR='./apps'
REGISTRY="harbor.technopole-demo.it"
PROJECT="ai"
CHART_DIR="./deploy"
CHART_NAME="umbrella"
RELEASE_NAME="ai"

# values
VALUES=("deploy/values-dev.yaml" "deploy/values-secret.yaml")

# Namespace
NAMESPACE="ai"

# Components list
COMPONENTS=(serving/gateway serving/transformer serving/predictor inference recommendation)

# ---------------- helpers ----------------
usage() {
  echo "Usage: $0 [-n|--namespace <ns>]"
  echo "Env: HARBOR_USER / HARBOR_PASS must be set"
  exit 1
}
log(){ printf "\033[1;34m➜ %s\033[0m\n" "$*"; }
ok(){  printf "\033[1;32m✔ %s\033[0m\n" "$*"; }
img_ref() { # $1=component $2=tag
  echo "${REGISTRY}/${PROJECT}/${1}:${2}"
}
build_and_push() { # $1=component $2=tag
  local comp="$1" tag="$2" df ref
  df="${APPS_DIR}/${1}/Dockerfile"
  ref="$(img_ref "$comp" "$tag")"
  [[ -f "$df" ]] || { echo "Dockerfile not found: $df"; exit 1; }
  log "Build & push $ref"
  docker build -t "$ref" -f "$df" .
  docker push "$ref"
}

# --------------- args --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--namespace) NAMESPACE="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown arg: $1"; usage ;;
  esac
done

# ------------- prechecks -----------------
: "${HARBOR_USER:?HARBOR_USER not set}"
: "${HARBOR_PASS:?HARBOR_PASS not set}"
[[ -d "$CHART_DIR" ]] || { echo "Chart dir not found: $CHART_DIR"; exit 1; }
for f in "${VALUES[@]}"; do [[ -f "$f" ]] || { echo "Missing values file: $f"; exit 1; }; done

# -------- Harbor login (images push) ----
log "Login to Harbor ($REGISTRY)"
docker login "$REGISTRY" -u "$HARBOR_USER" -p "$HARBOR_PASS" >/dev/null
ok "Harbor login OK"

# -------- prompt for tags (MANDATORY) ----
declare -A TAGS=()
echo
echo "Enter a tag for each component (cannot be empty):"
for comp in "${COMPONENTS[@]}"; do
  while : ; do
    read -r -p "  - $comp tag: " tag
    if [[ -n "${tag// }" ]]; then
      TAGS["$comp"]="$tag"
      break
    else
      echo "    Tag is required. Please enter a non-empty value."
    fi
  done
done
echo

# ---- build+push per component (all required) ---
for comp in "${COMPONENTS[@]}"; do
  path="${APPS_DIR}/$comp"
  [[ -d "$path" ]] || { echo "Path not found for $comp: $path"; exit 1; }
  build_and_push "$comp" "${TAGS[$comp]}"
done

# -------------- deploy (local chart) -----
log "Deploy Helm release 'ai' (namespace: $NAMESPACE) from local dir: $CHART_DIR"
helm upgrade --install ai "$CHART_DIR" \
  -n "$NAMESPACE" \
  --create-namespace \
  $(printf ' -f %s' "${VALUES[@]}") \
  --set-string "serving.gateway.image.tag=${TAGS[serving/gateway]}" \
  --set-string "serving.transformer.image.tag=${TAGS[serving/transformer]}" \
  --set-string "serving.predictor.image.tag=${TAGS[serving/predictor]}" \
  --set-string "bems.inference.image.tag=${TAGS['inference']}" \
  --set-string "bems.recommendation.image.tag=${TAGS['recommendation']}"

ok "Release updated"