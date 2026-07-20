#!/usr/bin/env bash
# Deploy the backend to Cloud Run and assert the deployed configuration.
#
# The flags below are infrastructure policy, not conveniences:
#   --cpu-throttling      request-based billing (free tier applies); inference
#                         runs inside the request, so throttled CPU is correct
#   --max-instances 1     cost blast-radius bound and in-memory job-store
#                         correctness (state is per-instance)
#   --concurrency 4       lightweight requests stay responsive during an
#                         inference; expensive work is bounded separately by
#                         the in-process InferenceAdmission (capacity 1)
#   single uvicorn worker (Dockerfile CMD): the inference bound is
#                         process-local; more workers would multiply it
#
# Invariant: max instances x workers per instance x inference slots = 1.

set -euo pipefail

PROJECT="project-c9884a54-abe1-429f-92a"
REGION="us-central1"
SERVICE="harmonyrestorer-backend"
IMAGE="us-central1-docker.pkg.dev/${PROJECT}/harmonyrestorer/backend:latest"

EXPECT_MEMORY="4Gi"
EXPECT_CPU="2"
EXPECT_TIMEOUT="900"
EXPECT_CONCURRENCY="4"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "==> Capturing current revision for rollback"
PREVIOUS_REVISION=$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --project "${PROJECT}" --format='value(status.latestReadyRevisionName)' || echo "none")
echo "    previous revision: ${PREVIOUS_REVISION}"

echo "==> Building and pushing linux/amd64 image"
docker buildx build --platform linux/amd64 -t "${IMAGE}" --push .

echo "==> Deploying"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT}" \
  --allow-unauthenticated \
  --port 8000 \
  --cpu-throttling \
  --max-instances 1 \
  --concurrency "${EXPECT_CONCURRENCY}" \
  --memory "${EXPECT_MEMORY}" \
  --cpu "${EXPECT_CPU}" \
  --timeout "${EXPECT_TIMEOUT}" \
  --set-env-vars "CORS_ORIGINS=https://harmonyrestorer.online,ENABLE_DOCS=false,LOG_FORMAT=json,LOG_LEVEL=INFO"

echo "==> Asserting deployed configuration"
SPEC=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --project "${PROJECT}" \
  --format='value(spec.template.metadata.annotations["autoscaling.knative.dev/maxScale"],
                  spec.template.metadata.annotations["run.googleapis.com/cpu-throttling"],
                  spec.template.spec.containers[0].resources.limits.memory,
                  spec.template.spec.containers[0].resources.limits.cpu,
                  spec.template.spec.timeoutSeconds,
                  spec.template.spec.containerConcurrency)')
read -r MAX_SCALE THROTTLING MEMORY CPU TIMEOUT CONCURRENCY <<< "${SPEC}"

fail() { echo "ASSERTION FAILED: $1"; echo "Rollback: gcloud run services update-traffic ${SERVICE} --region ${REGION} --project ${PROJECT} --to-revisions ${PREVIOUS_REVISION}=100"; exit 1; }

[ "${MAX_SCALE}" = "1" ] || fail "maxScale=${MAX_SCALE}, expected 1"
# Annotation is absent or 'true' when throttling (request-based billing) is on
[ "${THROTTLING:-true}" != "false" ] || fail "cpu-throttling disabled (instance-based billing)"
[ "${MEMORY}" = "${EXPECT_MEMORY}" ] || fail "memory=${MEMORY}, expected ${EXPECT_MEMORY}"
[ "${CPU}" = "${EXPECT_CPU}" ] || fail "cpu=${CPU}, expected ${EXPECT_CPU}"
[ "${TIMEOUT}" = "${EXPECT_TIMEOUT}" ] || fail "timeout=${TIMEOUT}, expected ${EXPECT_TIMEOUT}"
[ "${CONCURRENCY}" = "${EXPECT_CONCURRENCY}" ] || fail "concurrency=${CONCURRENCY}, expected ${EXPECT_CONCURRENCY}"

echo "==> All configuration assertions passed"
echo "    Rollback if needed: gcloud run services update-traffic ${SERVICE} --region ${REGION} --project ${PROJECT} --to-revisions ${PREVIOUS_REVISION}=100"
