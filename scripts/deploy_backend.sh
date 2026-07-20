#!/usr/bin/env bash
# Deploy the backend to Cloud Run as a no-traffic candidate revision, assert
# its configuration, and (separately) promote it to production traffic.
#
# Usage:
#   scripts/deploy_backend.sh deploy    Build SHA-tagged image, deploy candidate
#   scripts/deploy_backend.sh promote   Shift 100% traffic to the latest revision
#
# The flags below are infrastructure policy, not conveniences:
#   --cpu-throttling      request-based billing (free tier applies); inference
#                         runs inside the request, so throttled CPU is correct
#   --max-instances 1     intended scaling and cost boundary (Cloud Run may
#                         briefly exceed it; not an absolute guarantee) and
#                         in-memory job-store correctness
#   min instances 0       scale-to-zero; minimum instances would incur idle
#                         charges even under request-based billing
#   --concurrency 4       lightweight requests stay responsive during an
#                         inference; expensive work is bounded separately by
#                         the in-process InferenceAdmission (capacity 1)
#   single uvicorn worker (Dockerfile CMD): the inference bound is
#                         process-local; more workers would multiply it
#
# Intended invariant per revision:
#   max instances x workers per instance x inference slots = 1
#
# Images are tagged with the git SHA so commit, image, and revision stay
# auditable; the resolved digest is printed as the immutable identity.

set -euo pipefail

MODE="${1:-deploy}"

PROJECT="project-c9884a54-abe1-429f-92a"
REGION="us-central1"
SERVICE="harmonyrestorer-backend"

EXPECT_MEMORY="4Gi"
EXPECT_CPU="2"
EXPECT_TIMEOUT="900"
EXPECT_CONCURRENCY="4"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

fail() {
  echo "FAILED: $1"
  if [ -n "${PREVIOUS_REVISION:-}" ] && [ "${PREVIOUS_REVISION}" != "none" ]; then
    echo "Rollback: gcloud run services update-traffic ${SERVICE} --region ${REGION} --project ${PROJECT} --to-revisions ${PREVIOUS_REVISION}=100"
  fi
  exit 1
}

if [ "${MODE}" = "promote" ]; then
  echo "==> Promoting latest revision to 100% traffic"
  gcloud run services update-traffic "${SERVICE}" --region "${REGION}" \
    --project "${PROJECT}" --to-latest
  echo "==> Promoted. Run production smoke tests now."
  exit 0
fi

[ "${MODE}" = "deploy" ] || fail "unknown mode '${MODE}' (use deploy or promote)"

[ -z "$(git status --porcelain)" ] || fail "working tree not clean; commit first so the image maps to a commit"
GIT_SHA=$(git rev-parse --short HEAD)
IMAGE="us-central1-docker.pkg.dev/${PROJECT}/harmonyrestorer/backend:${GIT_SHA}"

echo "==> Capturing current serving revision for rollback"
PREVIOUS_REVISION=$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --project "${PROJECT}" --format='value(status.latestReadyRevisionName)' || echo "none")
echo "    previous revision: ${PREVIOUS_REVISION}"

echo "==> Building and pushing linux/amd64 image ${IMAGE}"
docker buildx build --platform linux/amd64 -t "${IMAGE}" --push .

DIGEST=$(gcloud artifacts docker images describe "${IMAGE}" \
  --format='value(image_summary.digest)' 2>/dev/null || echo "unknown")
echo "    image digest: ${DIGEST}"

echo "==> Deploying candidate revision (no production traffic)"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT}" \
  --allow-unauthenticated \
  --no-traffic \
  --tag candidate \
  --port 8000 \
  --cpu-throttling \
  --max-instances 1 \
  --min-instances 0 \
  --concurrency "${EXPECT_CONCURRENCY}" \
  --memory "${EXPECT_MEMORY}" \
  --cpu "${EXPECT_CPU}" \
  --timeout "${EXPECT_TIMEOUT}" \
  --set-env-vars "CORS_ORIGINS=https://harmonyrestorer.online,ENABLE_DOCS=false,LOG_FORMAT=json,LOG_LEVEL=INFO"

CANDIDATE_REVISION=$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --project "${PROJECT}" --format='value(status.latestCreatedRevisionName)')
echo "    candidate revision: ${CANDIDATE_REVISION}"

echo "==> Asserting configuration on the candidate revision object itself"
# csv preserves empty fields for absent annotations; tab-separated value()
# output collapses them, silently shifting every later field left
REV=$(gcloud run revisions describe "${CANDIDATE_REVISION}" --region "${REGION}" --project "${PROJECT}" \
  --format='csv[no-heading](metadata.annotations["autoscaling.knative.dev/maxScale"],
                  metadata.annotations["autoscaling.knative.dev/minScale"],
                  metadata.annotations["run.googleapis.com/cpu-throttling"],
                  spec.containers[0].resources.limits.memory,
                  spec.containers[0].resources.limits.cpu,
                  spec.timeoutSeconds,
                  spec.containerConcurrency)')
IFS=',' read -r MAX_SCALE MIN_SCALE THROTTLING MEMORY CPU TIMEOUT CONCURRENCY <<< "${REV}"

# Tagged candidates outside the traffic split are not governed by service-level
# limits, so the revision's own maxScale annotation is the binding assertion.
[ "${MAX_SCALE}" = "1" ] || fail "candidate revision maxScale='${MAX_SCALE}', expected 1"
[ -z "${MIN_SCALE}" ] || [ "${MIN_SCALE}" = "0" ] || fail "minScale='${MIN_SCALE}', expected 0 or unset (scale-to-zero)"
[ "${THROTTLING:-true}" != "false" ] || fail "cpu-throttling disabled (instance-based billing)"
[ "${MEMORY}" = "${EXPECT_MEMORY}" ] || fail "memory='${MEMORY}', expected ${EXPECT_MEMORY}"
[ "${CPU}" = "${EXPECT_CPU}" ] || fail "cpu='${CPU}', expected ${EXPECT_CPU}"
[ "${TIMEOUT}" = "${EXPECT_TIMEOUT}" ] || fail "timeout='${TIMEOUT}', expected ${EXPECT_TIMEOUT}"
[ "${CONCURRENCY}" = "${EXPECT_CONCURRENCY}" ] || fail "concurrency='${CONCURRENCY}', expected ${EXPECT_CONCURRENCY}"

CANDIDATE_URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --project "${PROJECT}" \
  --format='value(status.traffic.filter("tag:candidate").extract("url").flatten())' 2>/dev/null || true)

echo "==> All candidate assertions passed"
echo "    commit:    ${GIT_SHA}"
echo "    digest:    ${DIGEST}"
echo "    revision:  ${CANDIDATE_REVISION}"
echo "    candidate URL: ${CANDIDATE_URL:-see console (tag: candidate)}"
echo "    Promote after Phase 6 gates: scripts/deploy_backend.sh promote"
echo "    Rollback: gcloud run services update-traffic ${SERVICE} --region ${REGION} --project ${PROJECT} --to-revisions ${PREVIOUS_REVISION}=100"
