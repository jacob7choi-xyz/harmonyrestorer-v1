#!/usr/bin/env bash
# Starts the runtime image with no network and checks that it serves.
#
# /health must return 200. With the default OpGAN engine that requires the startup check
# of the checkpoint's digest to have passed (backend/app/routes/health.py), so this runs
# the check in docs/adr/007-checkpoint-digest.md against the file baked into the image.
# That check hashes the file without loading the weights, so one short WAV is then sent
# through the API, which checks the digest again and loads the weights, and the job must
# complete and return a WAV.
#
# The container runs on a read-only filesystem with only /tmp writable, as Lambda runs
# images (docs/adr/009-pay-per-use-hosting.md), so a write anywhere else fails this check.
#
# The Lambda Web Adapter runs only on Lambda, so this checks that it is in the image and
# configured as ADR 009 sets out, not that it serves.
#
# Usage: scripts/check_runtime_image.sh IMAGE
# CI calls this after building the runtime stage; it needs Docker and python3.

set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${1:?usage: scripts/check_runtime_image.sh IMAGE}"
SAMPLE="frontend/public/sample-noisy.wav"
API="http://localhost:8000/api/v1"
CONTAINER="runtime-check-$$"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# The server's own log, without the requests this script made, which would bury it
fail() {
  printf 'FAILED: %s\n' "$1"
  docker logs "$CONTAINER" 2>&1 | grep -v '"GET /' | tail -n 50 || true
  exit 1
}

# Reads one field from the JSON object on stdin
json_field() {
  python3 -c 'import json, sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

# Runs in the container, which has no network, as the image's own user
in_container() {
  docker exec "$CONTAINER" "$@"
}

# Each request is bounded, so a hung server fails this check instead of the job timeout
request() {
  in_container curl --max-time 30 "$@"
}

docker run -d --name "$CONTAINER" --network none \
  --read-only --tmpfs /tmp:rw,size=1g \
  -e UPLOAD_DIR=/tmp/uploads -e PROCESSED_DIR=/tmp/processed \
  "$IMAGE" >/dev/null

# Waits only while nothing answers (curl reports 000); the first HTTP answer must be 200
code="000"
for _ in $(seq 1 60); do
  code="$(request -sS -o /tmp/health.json -w '%{http_code}' \
    "http://localhost:8000/health" 2>/dev/null)" || true
  [ "$code" = "000" ] || break
  sleep 2
done
in_container cat /tmp/health.json 2>/dev/null && printf '\n'
# An empty code means docker exec itself failed, usually because the container exited
[ "$code" = "200" ] || fail "/health answered ${code:-nothing}, not 200"

in_container test -x /opt/extensions/lambda-adapter \
  || fail "the Lambda Web Adapter is missing from /opt/extensions or not executable"
for setting in AWS_LWA_PORT=8000 AWS_LWA_READINESS_CHECK_PATH=/ \
  AWS_LWA_READINESS_CHECK_TIMEOUT_SECONDS=8 AWS_LWA_INVOKE_MODE=response_stream; do
  name="${setting%%=*}"
  value="$(in_container printenv "$name")" || value=""
  [ "$value" = "${setting#*=}" ] || fail "$name is '$value', not '${setting#*=}'"
done

# docker cp cannot write into a read-only container, so the file is piped in instead
docker exec -i "$CONTAINER" sh -c 'cat > /tmp/sample.wav' < "$SAMPLE"
job_id="$(request -fsS -F "file=@/tmp/sample.wav;type=audio/wav" "$API/denoise" \
  | json_field job_id)" || fail "the upload was not accepted"

status=""
for _ in $(seq 1 60); do
  status="$(request -fsS "$API/status/$job_id" | json_field status)" \
    || fail "could not read the status of job $job_id"
  case "$status" in
    completed | failed) break ;;
  esac
  sleep 2
done
[ "$status" = "completed" ] || fail "job $job_id was '$status' after 120 s, not completed"

request -fsS -o /tmp/restored.wav "$API/download/$job_id" \
  || fail "could not download the result of job $job_id"
in_container python -c '
import sys
import wave

with wave.open("/tmp/restored.wav") as w:
    frames, rate = w.getnframes(), w.getframerate()
print(f"restored WAV: {frames} frames at {rate} Hz")
sys.exit(0 if frames > 0 else 1)
' || fail "the downloaded file is not a WAV with audio frames"

printf 'RUNTIME IMAGE CHECK PASSED\n'
