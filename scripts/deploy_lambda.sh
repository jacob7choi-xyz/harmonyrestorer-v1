#!/usr/bin/env bash
# Deploys a published image to the Lambda function behind the function URL, as
# docs/adr/009-pay-per-use-hosting.md decisions 11 and 12 set out: the image is copied to
# ECR by digest, a version is published and warmed, the alias moves to it, and a smoke test
# through the function URL moves the alias back when it fails.
#
# Usage: scripts/deploy_lambda.sh
# Reads GHCR_IMAGE, DIGEST (the linux/amd64 manifest's digest), COMMIT, FUNCTION, REPO,
# ALIAS, BACKEND_URL and AWS_REGION from the environment, and AWS credentials the way the
# AWS CLI does. CI runs it in the deploy job (.github/workflows/ci.yml); the AWS side is set
# up by docs/aws-stage-1-runbook.md.

set -euo pipefail

: "${GHCR_IMAGE:?}" "${DIGEST:?}" "${COMMIT:?}" "${FUNCTION:?}" "${REPO:?}" "${ALIAS:?}"
: "${BACKEND_URL:?}" "${AWS_REGION:?}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'FAILED: %s\n' "$1" >&2
  exit 1
}

# An empty or malformed digest, for example a missing job output, stops the deploy here
[[ "$DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || fail "DIGEST is not a sha256 digest: '$DIGEST'"

# Reserved concurrency is 1. In ADR 009 precondition 4, a request sent while a job ran got
# 429, so the smoke test and the warm-up may get one too. curl retries on 429 with --retry;
# the AWS CLI gets more attempts here, though whether it retries a 429 was not tested.
export AWS_MAX_ATTEMPTS=10
CURL_RETRY=(--retry 6 --retry-delay 10)

# Copies the image to ECR unless ECR already holds the digest. Tags there are immutable,
# so pushing the same commit's tag a second time would fail.
copy_image() {
  local registry=$1 pushed
  if aws ecr describe-images --repository-name "$REPO" \
    --image-ids "imageDigest=$DIGEST" > /dev/null 2>&1; then
    echo "ECR already holds $DIGEST"
    return
  fi
  docker pull --platform linux/amd64 "$GHCR_IMAGE@$DIGEST"
  aws ecr get-login-password | docker login --username AWS --password-stdin "$registry" > /dev/null
  docker tag "$GHCR_IMAGE@$DIGEST" "$registry/$REPO:$COMMIT"
  pushed="$(docker push "$registry/$REPO:$COMMIT" | grep -o 'digest: sha256:[0-9a-f]\{64\}' \
    | cut -d' ' -f2 || true)"
  docker logout "$registry" > /dev/null
  if [ "$pushed" != "$DIGEST" ]; then
    echo "Tags in this repository cannot be overwritten; delete tag $COMMIT before a re-run" \
      "(docs/aws-stage-1-runbook.md, step 11)" >&2
    fail "ECR stored '${pushed:-nothing}' under tag $COMMIT for $DIGEST"
  fi
}

# Points the function's code at the image and prints the version published from it
publish_version() {
  local registry=$1
  aws lambda update-function-code --function-name "$FUNCTION" \
    --image-uri "$registry/$REPO@$DIGEST" --query LastUpdateStatus --output text > /dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION"
  aws lambda publish-version --function-name "$FUNCTION" --description "$COMMIT" \
    --query Version --output text
}

# A GET /health event in the shape the function URL sends
write_health_event() {
  cat > "$1" <<'EOF'
{"version": "2.0", "rawPath": "/health", "rawQueryString": "", "headers": {"host": "deploy"},
 "requestContext": {"http": {"method": "GET", "path": "/health", "protocol": "HTTP/1.1",
   "sourceIp": "198.51.100.7", "userAgent": "deploy"}, "requestId": "deploy-warm-up",
   "routeKey": "$default", "stage": "$default", "timeEpoch": 0},
 "isBase64Encoded": false}
EOF
}

# Starts the version before the alias moves to it, so that its first start falls on this
# call and not on a visitor. The one first start of a new image measured, on 2026-11-07,
# took about 78 s.
warm() {
  local version=$1 body
  write_health_event "$WORK/event.json"
  aws lambda invoke --function-name "$FUNCTION" --qualifier "$version" \
    --cli-binary-format raw-in-base64-out --cli-read-timeout 150 \
    --payload "file://$WORK/event.json" "$WORK/warm.out" > /dev/null
  body="$(tr -d '\000' < "$WORK/warm.out")"
  [[ "$body" == *'"status":"healthy"'* ]] \
    || fail "version $version did not report healthy when warmed"
}

move_alias() {
  aws lambda update-alias --function-name "$FUNCTION" --name "$ALIAS" \
    --function-version "$1" --query FunctionVersion --output text > /dev/null || return 1
  echo "$ALIAS points at version $1"
}

# One second of a 440 Hz tone, 16 kHz mono 16-bit, as the smoke test's upload
write_test_wav() {
  python3 -I -c '
import math, struct, sys, wave
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    w.writeframes(b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * n / 16000)))
        for n in range(16000)))
' "$1"
}

# ADR 009 decision 12: /health is healthy, a 1 s WAV completes, its download is audio/wav
smoke_test() {
  local health job content_type
  health="$(curl -fsS -m 60 "${CURL_RETRY[@]}" "$BACKEND_URL/health")" || return 1
  [[ "$health" == *'"status":"healthy"'* ]] || return 1
  write_test_wav "$WORK/smoke.wav" || return 1
  job="$(curl -fsS -m 120 "${CURL_RETRY[@]}" -F "file=@$WORK/smoke.wav;type=audio/wav" \
    "$BACKEND_URL/api/v1/denoise" | jq -r 'select(.status == "completed") | .job_id')" || return 1
  [ -n "$job" ] || return 1
  content_type="$(curl -fsS -m 60 "${CURL_RETRY[@]}" -o /dev/null -w '%{content_type}' \
    "$BACKEND_URL/api/v1/download/$job?format=wav")" || return 1
  [ "$content_type" = "audio/wav" ]
}

main() {
  local account registry previous version
  account="$(aws sts get-caller-identity --query Account --output text)"
  registry="$account.dkr.ecr.$AWS_REGION.amazonaws.com"
  copy_image "$registry"
  previous="$(aws lambda get-alias --function-name "$FUNCTION" --name "$ALIAS" \
    --query FunctionVersion --output text)"
  version="$(publish_version "$registry")"
  echo "published version $version; $ALIAS is on version $previous"
  warm "$version"
  move_alias "$version"
  if smoke_test; then
    echo "smoke test passed; version $version is live"
    return
  fi
  echo "smoke test failed"
  if [ "$previous" = "$version" ]; then
    fail "version $version was live before this deploy; there is no earlier version to restore"
  fi
  move_alias "$previous" \
    || fail "moving $ALIAS back to version $previous failed; it may still be on $version"
  fail "deploy of $DIGEST was rolled back to version $previous"
}

main "$@"
