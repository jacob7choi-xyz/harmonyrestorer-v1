# ADR-009: Deploy Stage 1 of the Backend as One AWS Lambda Function

**Date:** 2026-10-29, rewritten 2026-11-05

**Status:** proposed

---

## Context

The backend is not serving. Billing is disabled on the Google Cloud project, so the Cloud Run deployment in `scripts/deploy_backend.sh` cannot run, and an AWS account is available. The aim is to pay for compute while requests run rather than for an always-on instance. ADR 010 (accepted) put job records, result files and the rate limit behind ports and deferred dispatch. Stage 1 runs the current adapters on Lambda without changing them. Stage 2 replaces the adapters that work only inside one process.

### What the backend does today

Read at commit `3373ea5`.

- The image runs one uvicorn worker (`Dockerfile:93-95`), so a container serves its requests in one process.
- Job records and request counts are kept in the process's memory (`backend/app/services/job_store.py`, `backend/app/services/rate_limit.py`), and result files on the container's disk (`backend/app/services/file_store.py`). These are the only adapters.
- Inference admission allows `INFERENCE_CONCURRENCY` inferences at once per process, 1 by default, and an upload that finds no free slot gets 503 (`backend/app/routes/denoise.py:44`, `:271-277`). Transcoding has its own slot (`backend/app/services/transcode.py:32`).
- The upload request waits for inference to finish before it replies (`backend/app/routes/denoise.py:391`).
- Uploads and results are written under `UPLOAD_DIR` and `PROCESSED_DIR`, which default to `backend/uploads` and `backend/processed` (`backend/app/config.py:67-68`).
- A cleanup loop started in the app's lifespan removes COMPLETED and FAILED jobs past `JOB_TTL_SECONDS`, with their files (`backend/app/main.py:59-67`, `:107`). It runs only while the process runs.
- The per-IP job cap counts COMPLETED jobs until they expire (`backend/app/services/job_store.py:23`). With the defaults, `MAX_JOBS_PER_IP=3` and `JOB_TTL_SECONDS=3600` (`backend/app/config.py:136`, `:145`), one client identity can start three uploads an hour.
- Only `POST /api/v1/denoise` is rate limited (`backend/app/middleware.py:44`). Status and download requests are not.
- Uploads may be up to 50 MiB (`backend/app/config.py:104`), and the frontend checks the same limit in two places (`frontend/src/App.tsx:372`, `frontend/src/components/UploadArea.tsx:4-5`).

### What AWS's documentation says

Read word for word on 2026-11-05 from the Markdown versions of the AWS pages under References, the Lambda Web Adapter's README, and the Lambda pricing page, whose price was checked against AWS's public price file published on 2026-10-30.

- Lambda's documentation uses MB to mean 1,024 KB. A synchronous request and a synchronous response are each limited to 6 MB. A streamed response may be up to 200 MB; after the first 6 MB it is sent at up to 2 MBps.
- A function URL's invoke mode is `BUFFERED` by default, with a 6 MB maximum payload. `RESPONSE_STREAM` streams responses of up to 200 MB.
- A function URL's request body is base64-encoded when its content type is binary. The pages read do not say whether multipart form data counts as binary.
- The pages read give no request timeout for function URLs other than the function timeout, which is at most 900 s.
- API Gateway's integration timeout is 50 ms to 29 s for REST APIs, and raising it above 29 s "might require a reduction in your Region-level throttle quota". For HTTP APIs it is 30 s and cannot be raised.
- While an execution environment runs a request, it "is busy and cannot process other requests". When a function's concurrency exceeds its reserved concurrency, its function URL returns HTTP 429. A function URL accepts at most 10 requests per second for each unit of reserved concurrency. Reserved concurrency 0 "stops your Event Function from processing any events until you remove the limit".
- Lambda "terminates execution environments every few hours to allow for runtime updates and maintenance", and may terminate one "after a period of inactivity". `/tmp` is "unique to each execution environment".
- A function URL's authentication is `AWS_IAM` or `NONE`. When CORS is configured on a function URL, "Lambda automatically adds the CORS headers to all responses through the function URL". If the function also sends CORS headers, GET and POST responses can carry duplicates, and AWS recommends "configuring all CORS settings on the function URL, rather than sending CORS headers manually in the function response".
- The AWS WAF pages list the resource types it protects, and function URLs are not among them. CloudFront can front a function URL with origin access control, which requires `AWS_IAM` authentication.
- A container image must be in Amazon ECR in the same Region as the function, must target one architecture, and may be up to 10 GB uncompressed. The function must run on a read-only filesystem, with a writable `/tmp` of 512 MB to 10,240 MB.
- The init phase is limited to 10 s for on-demand functions; if it does not finish, Lambda retries it at the first invocation.
- Memory can be 128 MB to 10,240 MB; at 1,769 MB a function has the equivalent of one vCPU.
- Duration is billed in GB-seconds, and "Duration charges apply to code that runs in the handler of a function as well as initialization code that is declared outside of the handler." In US East (N. Virginia), x86 compute is $0.0000166667 per GB-second in the first pricing tier; other Regions were not checked. The free tier is one million requests and 400,000 GB-seconds a month.
- AWS Budgets sends notifications by email or SNS. Its actions apply IAM or SCP policies or target EC2 or RDS instances; Lambda is not a listed target.
- AWS's ECR policy example grants `ecr:GetAuthorizationToken` with `Resource` set to `*`.
- The Lambda Web Adapter, in the `awslabs` GitHub organization, runs an existing HTTP app in a container image and supports function URLs and response streaming. Its defaults are port 8080, readiness path `/`, healthy status 100 to 499, and invoke mode `buffered`. If its readiness timeout is unset it waits indefinitely; if the timeout expires, init fails. The README does not say whether it passes the caller's address to the app.

### What was measured

The runtime image was measured once in GitHub Actions (run 40984036981, commit `7eeeabd` on the branch `measure/runtime-sizing`, which holds `scripts/measure_runtime_image.sh`). For each length of audio it started a new container, timed how long `/health` took to return 200, and uploaded the same file twice. The first job includes loading the model weights. Every figure below matches the run's log.

| Audio | CPUs | Startup to `/health` 200 | First job | Second job | MP3 download | Peak memory, container / server process |
|-------|------|--------------------------|-----------|------------|--------------|-----------------------------------------|
| 10 s  | 2 | 3.83 s | 0.74 s | 0.63 s | 0.11 s | 455 / 592 MiB |
| 60 s  | 2 | 3.83 s | 3.06 s | 2.87 s | 0.28 s | 549 / 681 MiB |
| 600 s | 2 | 3.82 s | 24.72 s | 26.24 s | 2.07 s | 741 / 823 MiB |
| 10 s  | 4 | 3.39 s | 0.52 s | 0.41 s | 0.11 s | 455 / 593 MiB |
| 60 s  | 4 | 3.41 s | 2.20 s | 1.57 s | 0.26 s | 510 / 641 MiB |
| 600 s | 4 | 3.45 s | 15.13 s | 14.40 s | 1.91 s | 741 / 822 MiB |

> **Note:** These figures come from one run on one kind of runner (AMD EPYC 7763, 4 vCPUs), with the image already on the machine, so startup does not include pulling it from ECR. Startup includes importing torch and hashing the checkpoint. The input was random noise at 16 kHz mono. The container figure is the cgroup's peak, which includes page cache; the server figure is the server process's peak resident memory. The log reports 2 torch threads in both modes; the backend sets no thread count. None of this was measured on Lambda.

The published image (`sha256:fc1fd3b662e4d304ee7a8c56cb1ba5e27a119fc16a265bad06b1aa3a1996e6ed`, linux/amd64, 3,094,115,281 bytes as `docker image inspect` reports it) ran locally on 2026-11-05 with `docker run --read-only`, a tmpfs at `/tmp`, and `UPLOAD_DIR` and `PROCESSED_DIR` under `/tmp`. `/health` reported healthy, a 1 s upload completed, WAV and MP3 downloads returned 200, and no log line contained an error or a warning. The run used the default OpGAN engine under emulation on an arm64 host; its timings are not used here.

## Decision

Stage 1 runs the current image, with the Lambda Web Adapter added and the adapters unchanged, as one Lambda function behind a function URL. None of the AWS resources, the deploy job or the frontend changes exist yet; they are listed under Work and Preconditions.

1. **One function with reserved concurrency 1.** At most one execution environment runs. While it lasts, the in-memory `JobStore` and `RateLimiter`, the admission slot and the local `FileStore` hold the state for every request that Lambda passes to the function. Compute is billed per GB-second while the environment works, so one environment busy for a whole month bounds the compute cost.
2. **A function URL with `NONE` authentication, not API Gateway.** The slowest measured job, 26.24 s, leaves about 3 s under the REST API's 29 s integration timeout. The pages read give no timeout for a function URL other than the function timeout. Stage 1 runs inline, which ADR 010 section 4 allows where a request can stay open longer than the slowest job.
3. **CORS belongs to the function URL**, allowing only the frontend's origin and the methods GET and POST. `CORS_ORIGINS` is set to an empty value on the function (unset, it defaults to `http://localhost:3000`), so the app adds no CORS headers of its own. That includes the headers `backend/app/main.py:131-137` adds to the app's own 429 and 503 responses; the function URL's headers are expected to cover those responses instead.
4. **The Lambda Web Adapter runs the app.** It is added to the image, pinned by digest, with `AWS_LWA_PORT=8000`, `AWS_LWA_READINESS_CHECK_PATH=/`, `AWS_LWA_READINESS_CHECK_TIMEOUT_SECONDS=8` and `AWS_LWA_INVOKE_MODE=response_stream`, and the function URL uses invoke mode `RESPONSE_STREAM`. Streaming lets a 600 s WAV, about 19.2 MB, reach the client. Readiness uses `/`, which returns 200 whenever the app serves requests (`backend/app/routes/health.py:15-25`), because `/health` returns 503 when any of its checks fails and would hold init until the readiness timeout. The 8 s timeout sits under the 10 s init limit; startup measured 3.4 to 3.8 s. The local `FileStore` streams files from local disk, so the download guard stays (ADR 010, section 3).
5. **Memory 3,538 MB, two vCPUs at 1,769 MB each.** A 600 s job used about 91 GB-seconds on two vCPUs (3,538 MB for 26.24 s) and about 105 on four (7,076 MB for 15.13 s), so the cost per job is similar. Four vCPUs would shorten that job, and the time other users get 429, by about 11 s, but double the monthly ceiling from about $143 to about $292.
6. **Function timeout 120 s.** This is a judgment: about 4.6 times the slowest measured job, to leave room for a slower CPU on Lambda and for steps the measurement does not separate. It also caps one hung invocation at 120 s.
7. **`/tmp` of 2,048 MB and `MAX_ARTIFACT_BYTES` of 1 GiB.** With OpGAN, the byte budget reserves 81,125,888 bytes for a 4 MiB upload at the 600 s limit: the upload, the decoded size cap and the peak output (`backend/app/routes/denoise.py:288-292`). The other 1 GiB of `/tmp` is left for files the budget does not count, such as the UVR model directory and library temp files. With `DENOISER_ENGINE=uvr`, one such job reserves 851,160,336 bytes of the 1 GiB.
8. **Uploads are capped at 4 MiB** (`MAX_UPLOAD_BYTES=4194304`, and the same limit in the frontend). A 4 MiB body base64-encodes to 5,592,408 bytes, under the 6 MB limit of 6,291,456 bytes, leaving 699,048 bytes for the multipart envelope and the rest of the event; 5 MiB encodes to 6,990,508 bytes, over the limit. On this deployment, uploads between 4 MiB and 50 MiB get 413, a deliberate difference from today's limit.
9. **Job cap and TTL:** `MAX_JOBS_PER_IP=6` and `JOB_TTL_SECONDS=1200`, as `scripts/deploy_backend.sh:103` sets for Cloud Run, so one client identity can start six uploads in 20 minutes, and results stay downloadable for 20 minutes.
10. **The frontend retries `/status` with backoff on HTTP 429.** A 429 from Lambda arrives while another user's job holds the environment. Uploads and downloads are not retried.
11. **Deploys go through ECR by digest.** In the `Production` environment, a deploy job pulls the image the `publish` job pushed to GitHub Container Registry, by digest, pushes it to a private ECR repository with scan on push, and fails unless the digests match. It updates the function's code to that digest, publishes a version, and points the alias `live`, which the function URL serves, at it. It signs in to AWS with GitHub's OIDC token.
12. **Each deploy ends with a smoke test** through the function URL: `/health` returns 200 and healthy, a 1 s WAV upload completes, and its download returns `audio/wav`. If the test fails, the job points `live` back at the previous version.
13. **Alarms and logs.** CloudWatch alarms on the function's summed duration per hour and on throttles send email through SNS, alongside a budget alert. The function's log group keeps events for 14 days.
14. **The kill switch is manual.** An operator sets reserved concurrency to 0 with `aws lambda put-function-concurrency`, and sets it back to 1 to resume. The deploy role cannot change concurrency.

### Preconditions before the frontend uses the function

These are checked once on the deployed function, and the results are added to this record.

1. **Client identity.** Requests from two networks get different identities, and a client-supplied `X-Forwarded-For` does not change the identity. With `LOG_LEVEL=DEBUG` for the check, the rate-limit log line shows the identity (`backend/app/middleware.py:55`). uvicorn trusts forwarded headers from `127.0.0.1` (`Dockerfile:95`) and takes the rightmost untrusted address; whether the adapter forwards the caller's address is what this checks. If all callers share one identity, the function is not used for public traffic.
2. **Upload size.** A file of exactly 4 MiB uploads successfully, and a larger one is refused.
3. **Streaming.** A 600 s result, about 19.2 MB, downloads in full.
4. **Readable 429.** While a long job runs, a status request with the frontend's `Origin` gets 429 with an `Access-Control-Allow-Origin` header.
5. **Init.** A cold start's init duration, from the function's logs, is under 10 s.
6. **Quota.** The account allows 3,538 MB of memory and a reserved concurrency of 1.

### Work

None of this exists yet.

- In the repository: the image check in CI runs the container read-only with `/tmp` writable; the frontend takes its upload limit from a build setting; the frontend retries `/status` on 429; a test shows that an empty `CORS_ORIGINS` adds no CORS headers; the Lambda Web Adapter is added to the image.
- In the AWS account: the budget alert, the ECR repository, the OIDC provider and deploy role, the function, its alias and function URL, the alarms and the log retention.
- In GitHub: a required reviewer on `Production`, deployment from `main` only, and the role's ARN as a variable.
- The deploy job in `.github/workflows/ci.yml`.

### Security design

- **The checks on uploads stay on the server and fail closed.** Size and magic bytes are checked first (`backend/app/routes/denoise.py:252-264`), and decoded duration before inference (ADR 006).
- **No long-lived AWS keys.** The deploy role trusts only the OIDC subject `repo:jacob7choi-xyz/harmonyrestorer-v1:environment:Production` with audience `sts.amazonaws.com`. Its permissions name the one ECR repository and the one function, apart from `ecr:GetAuthorizationToken`, which AWS's own example grants on `*`. It can update code, publish versions and update the alias. It cannot change reserved concurrency, the function URL or its authentication and CORS settings.
- **The function's execution role carries logging only.** ECR access for a same-account image is granted as the ECR and Lambda pages describe.
- **The account ID stays out of the repository.** The role's ARN is a GitHub variable, and the deploy log is checked for the account ID in the image URI.
- **Reserved concurrency bounds compute cost; the budget only notifies.** The function URL is public, and no WAF sits in front of it.
- **Logs.** Job IDs appear in INFO logs and work only until their job expires, 20 minutes after completion. Client addresses are logged at DEBUG, and at WARNING only when an upload finds the inference slot busy.

## Consequences

### Positive
- Compute is billed for the time the environment works, including init, rather than for an always-on instance, and the first deployment needs no new port adapters.
- The image CI tested is the image deployed, identified by digest, and a deploy can be undone by pointing `live` at the previous version.
- On the GitHub runner, the slowest job (26.24 s) and the container's peak memory (741 MiB) were within the settings chosen here. They were not measured on Lambda.

### Negative
- **Other users are refused while a job runs.** Lambda answers their requests with 429 for as long as the job runs, 26.24 s for the longest measured job. Their uploads and downloads fail; only status polling retries.
- **Jobs and files are lost when Lambda replaces the environment**, which its documentation says happens every few hours and may happen after inactivity. Each deploy ends every job too. A status or download request for a lost job returns 404.
- **Uploads above 4 MiB are refused.** 4 MiB holds about 23.8 s of 16-bit 44.1 kHz stereo WAV, or about 4.4 minutes of 128 kbps MP3, against 50 MiB today.
- **Downloads are not rate limited.** Each download holds the environment while it streams, so a client repeating downloads of its own results can keep the environment busy for as long as those results live. Stage 1 accepts this; the alarms report it.
- **The compute ceiling is about $143 a month.** One environment busy for a 30-day month at 3,538 MB is about 8.96 million GB-seconds, or about $142.6 after the free tier at the US East price. Request charges, data transfer, ephemeral storage above 512 MB, CloudWatch logs and ECR storage are priced separately and are not included.
- **No WAF.** Reserved concurrency, the app's rate limit and the job caps are the only limits, and the last two depend on the client-identity check.

### Neutral
- `scripts/deploy_backend.sh` stays as it is.
- Lambda freezes the environment between invocations, so the cleanup loop runs only while one is in progress, and expired jobs and files can stay longer than their TTL.

## Stage 2

Stage 2 would lift the limits above by adding adapters behind ADR 010's ports: a `FileStore` on S3 that issues upload and download grants, so files no longer pass through the function, and a `JobStore` on DynamoDB with atomic caps and conditional state changes, plus a request limit shared across environments. ADR 010's rules apply: no networked `JobStore` before a `FileStore` that issues grants and keeps results every instance can read, and a rule, not yet written, for jobs left in PROCESSING. Stage 2 gets its own record.

## Open Questions

- The preconditions above are open until the deployed function is checked.
- The pages read do not say whether the 6 MB request limit applies to the encoded body; precondition 2 settles it in practice.
- How often Lambda replaces the environment under light use, and so how often jobs are lost, is not measured.
- The pages read do not say whether pulling the image counts toward the 10 s init limit; precondition 5 measures init as a whole.
- Whether to set torch's thread count is open; the measurement used 2 threads with 2 and 4 CPUs.
- The questions ADR 010 lists for stage 2 remain: presigned upload size limits, grant expiry, and how a download grant reaches the browser.

## Alternatives Considered

| Alternative | Why not chosen here |
|-------------|---------------------|
| API Gateway in front of the function | The 29 s REST integration timeout leaves about 3 s above the slowest measured job, and raising it might reduce the account's throttle quota. |
| CloudFront with WAF in front of the function URL | Adds WAF, but requires `AWS_IAM` authentication on the URL and more setup. It can be added if the alarms show abuse. |
| Four vCPUs | Shorter jobs at a similar cost per job, but about twice the monthly ceiling (decision 5). |
| Build the S3 and DynamoDB adapters first | Lifts the stage 1 limits, but puts the largest piece of work before any deployment. |
| Lightsail container service | Charged monthly by power and scale, whether the service is enabled or disabled. |
| ECS Express Mode on Fargate | Charges for Fargate compute and an Application Load Balancer; the pages read do not say a service can scale to zero. |
| App Runner | No longer open to new customers. |
| Cloud Run | Not usable while billing is disabled on the Google Cloud project; not rejected on technical grounds. |
| One always-on instance | Simplest, but pays for idle time. |

## References

AWS documentation, read on 2026-11-05 from the Markdown version of each page:

- https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html
- https://docs.aws.amazon.com/lambda/latest/dg/configuration-response-streaming.html
- https://docs.aws.amazon.com/lambda/latest/dg/config-rs-invoke-furls.html
- https://docs.aws.amazon.com/lambda/latest/dg/urls-invocation.html
- https://docs.aws.amazon.com/lambda/latest/dg/urls-configuration.html
- https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html
- https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html
- https://docs.aws.amazon.com/lambda/latest/dg/configuration-concurrency.html
- https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtime-environment.html
- https://docs.aws.amazon.com/lambda/latest/dg/configuration-memory.html
- https://docs.aws.amazon.com/lambda/latest/dg/configuration-ephemeral-storage.html
- https://docs.aws.amazon.com/lambda/latest/dg/images-create.html
- https://docs.aws.amazon.com/lambda/latest/dg/foundation-arch.html
- https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html
- https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-execution-service-limits-table.html
- https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-quotas.html
- https://docs.aws.amazon.com/waf/latest/developerguide/waf-chapter.html
- https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-lambda.html
- https://docs.aws.amazon.com/AmazonECR/latest/userguide/security_iam_id-based-policy-examples.html
- https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-controls.html
- https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html
- https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-container-services.html
- https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-overview.html
- https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html

Other sources, read on 2026-11-05:

- https://aws.amazon.com/lambda/pricing/, and AWS's public price file for Lambda in us-east-1, published 2026-10-30
- https://github.com/awslabs/aws-lambda-web-adapter (README)
