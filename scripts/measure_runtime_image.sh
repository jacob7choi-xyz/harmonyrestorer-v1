#!/usr/bin/env bash
# Measures how long the runtime image takes to start and to restore audio of a few
# lengths, and how much memory its container used, for choosing where to deploy it.
#
# Each length runs in a fresh container, so each start includes importing torch and
# hashing the checkpoint, and the first job includes loading the weights, which the
# backend does on first use. A second job of the same length in the same container shows
# the time without that load. The input is generated noise at 16 kHz mono, the rate the
# OpGAN engine reads; uploads at other rates also pay for resampling in ffmpeg.
#
# CPUSET (default 0-1) is passed to docker run --cpuset-cpus, which limits the container
# to those cores; --cpus would only throttle it, and torch would still start one thread
# per host core. An empty CPUSET runs on every core. Either way the figures come from this
# host's processors and are only an approximation of another host.
#
# Usage: scripts/measure_runtime_image.sh IMAGE [SECONDS ...]   (default 10 60 600)
# Needs Docker and python3.

set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${1:?usage: scripts/measure_runtime_image.sh IMAGE [SECONDS ...]}"
shift
[ "$#" -gt 0 ] || set -- 10 60 600
CPUSET="${CPUSET-0-1}"
API="http://localhost:8000/api/v1"
CONTAINER="runtime-measure-$$"
WORK="$(mktemp -d)"
CPU_LIMIT=()
[ -z "$CPUSET" ] || CPU_LIMIT=(--cpuset-cpus "$CPUSET")

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

# To stderr, because most callers run inside a command substitution
fail() {
  {
    printf 'FAILED: %s\n' "$1"
    docker logs "$CONTAINER" 2>&1 | grep -v '"GET /' | tail -n 50 || true
  } >&2
  exit 1
}

# Seconds on a monotonic clock; python3 because macOS date has no %N
now() {
  python3 -c 'import time; print(f"{time.monotonic():.3f}")'
}

elapsed() {
  python3 -c 'import sys; print(f"{float(sys.argv[2]) - float(sys.argv[1]):.2f}")' "$1" "$2"
}

mib() {
  python3 -c 'import sys; print(f"{int(sys.argv[1]) / 1048576:.0f}")' "$1"
}

# A 16 kHz mono 16-bit WAV of random samples, SECONDS long
make_wav() {
  python3 - "$1" "$2" <<'EOF'
import os
import sys
import wave

seconds, path = int(sys.argv[1]), sys.argv[2]
with wave.open(path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16_000)
    w.writeframes(os.urandom(seconds * 16_000 * 2))
EOF
}

# Peak memory of the container's cgroup since it started, in bytes. Includes page cache.
cgroup_peak() {
  docker exec "$CONTAINER" cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo ""
}

# Peak resident memory of the server process (PID 1 after the CMD's exec), in bytes
server_hwm() {
  docker exec "$CONTAINER" sh -c "awk '/VmHWM/ {print \$2 * 1024}' /proc/1/status"
}

# Starts the container and returns once /health answers 200; prints the seconds taken
start_and_wait() {
  local t0 code
  t0="$(now)"
  docker run -d --name "$CONTAINER" --network none ${CPU_LIMIT[@]+"${CPU_LIMIT[@]}"} \
    "$IMAGE" >/dev/null
  code="000"
  for _ in $(seq 1 1200); do
    code="$(docker exec "$CONTAINER" curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
      "http://localhost:8000/health" 2>/dev/null)" || true
    [ "$code" = "000" ] || break
    sleep 0.25
  done
  [ "$code" = "200" ] || fail "/health answered ${code:-nothing}, not 200"
  elapsed "$t0" "$(now)"
}

# Uploads the WAV already copied into the container; prints curl's total seconds
upload() {
  local out code seconds
  out="$(docker exec "$CONTAINER" curl -sS --max-time 1800 -o /tmp/response.json \
    -w '%{http_code} %{time_total}' -F "file=@/tmp/input.wav;type=audio/wav" \
    "$API/denoise")" || fail "the upload request failed"
  read -r code seconds <<<"$out"
  [ "$code" = "200" ] || fail "the upload answered $code"
  docker exec "$CONTAINER" python -c '
import json, sys
body = json.load(open("/tmp/response.json"))
print(body["job_id"])
sys.exit(0 if body["status"] == "completed" else 1)
' >"$WORK/job_id" || fail "the job did not complete"
  printf '%.2f\n' "$seconds"
}

# Downloads the last job's result as MP3; prints curl's total seconds
download_mp3() {
  local out code seconds
  out="$(docker exec "$CONTAINER" curl -sS --max-time 600 -o /tmp/restored.mp3 \
    -w '%{http_code} %{time_total}' "$API/download/$(cat "$WORK/job_id")?format=mp3")" \
    || fail "the download request failed"
  read -r code seconds <<<"$out"
  [ "$code" = "200" ] || fail "the MP3 download answered $code"
  printf '%.2f\n' "$seconds"
}

measure() {
  local seconds="$1" startup peak_start job1 job2 mp3 peak_end hwm
  make_wav "$seconds" "$WORK/input.wav"
  startup="$(start_and_wait)"
  peak_start="$(cgroup_peak)"
  docker cp "$WORK/input.wav" "$CONTAINER:/tmp/input.wav"
  job1="$(upload)"
  job2="$(upload)"
  mp3="$(download_mp3)"
  peak_end="$(cgroup_peak)"
  hwm="$(server_hwm)"
  report "$(printf '%8s %9s %8s %8s %8s %12s %12s %10s' "$seconds" "$startup" "$job1" \
    "$job2" "$mp3" "${peak_start:+$(mib "$peak_start")}" "${peak_end:+$(mib "$peak_end")}" \
    "$(mib "$hwm")")"
  docker rm -f "$CONTAINER" >/dev/null
}

# torch's import time and thread count, in a separate container so neither start above
# includes it
torch_info() {
  docker run --rm --network none ${CPU_LIMIT[@]+"${CPU_LIMIT[@]}"} --entrypoint python "$IMAGE" -c '
import os, time
t = time.monotonic()
import torch
print(f"import torch: {time.monotonic() - t:.2f} s; torch threads: {torch.get_num_threads()};"
      f" CPUs the container may use: {len(os.sched_getaffinity(0))}")
'
}

# Each line goes out as soon as it is known, so a run that times out keeps earlier rows.
# In GitHub Actions it is also added to the job summary as preformatted text.
report() {
  printf '%s\n' "$1"
  [ -z "${GITHUB_STEP_SUMMARY:-}" ] || printf '    %s\n' "$1" >>"$GITHUB_STEP_SUMMARY"
}

report "image: $IMAGE, cpuset: ${CPUSET:-all cores}"
report "host: $(getconf _NPROCESSORS_ONLN) CPUs, $(awk -F': ' '/model name/ {print $2; exit}' \
  /proc/cpuinfo 2>/dev/null || uname -m)"
report "$(torch_info)"
report "Times in seconds, memory in MiB. Job times are the upload request, which returns"
report "after inference. Memory is the cgroup peak (page cache included) after startup and"
report "after all jobs, and the server process's peak resident memory."
report "$(printf '%8s %9s %8s %8s %8s %12s %12s %10s' audio_s startup_s job1_s job2_s mp3_s \
  peak_start peak_end server_hwm)"

for seconds in "$@"; do
  measure "$seconds"
done
