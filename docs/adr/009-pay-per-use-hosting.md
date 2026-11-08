# ADR-009: Deploy Stage 1 of the Backend as One AWS Lambda Function

**Date:** 2026-10-29, rewritten 2026-11-05, amended 2026-11-05 after the account's limits and the cost of data transfer were checked, amended 2026-11-07 with the precondition results from the first deployment, amended 2026-11-08 with the CI deploy, the startup measurements and corrections, accepted 2026-11-08

**Status:** accepted

---

## Context

When this was written (2026-11-05), the backend was not serving. Billing is disabled on the Google Cloud project, so the Cloud Run deployment in `scripts/deploy_backend.sh` cannot run, and an AWS account is available. The aim is to pay for compute while requests run rather than for an always-on instance. ADR 010 (accepted) put job records, result files and the rate limit behind ports and deferred dispatch. Stage 1 runs the current adapters on Lambda without changing them. Stage 2 replaces the adapters that work only inside one process.

### What the backend does today

Read at commit `8eb0acb`.

- The image runs one uvicorn worker (`Dockerfile:113`), so a container serves its requests in one process.
- Job records and request counts are kept in the process's memory (`backend/app/services/job_store.py`, `backend/app/services/rate_limit.py`), and result files on the container's disk (`backend/app/services/file_store.py`). These are the only adapters.
- Inference admission allows `INFERENCE_CONCURRENCY` inferences at once per process, 1 by default, and an upload that finds no free slot gets 503 (`backend/app/routes/denoise.py:44`, `:271-277`). Transcoding has its own slot (`backend/app/services/transcode.py:32`).
- The upload request waits for inference to finish before it replies (`backend/app/routes/denoise.py:391`).
- Uploads and results are written under `UPLOAD_DIR` and `PROCESSED_DIR`, which default to `backend/uploads` and `backend/processed` (`backend/app/config.py:77-78`).
- A cleanup loop started in the app's lifespan removes COMPLETED and FAILED jobs past `JOB_TTL_SECONDS`, with their files (`backend/app/main.py:59-67`, `:107`). It runs only while the process runs.
- The per-IP job cap counts COMPLETED jobs until they expire (`backend/app/services/job_store.py:22-24`). With the defaults, `MAX_JOBS_PER_IP=3` and `JOB_TTL_SECONDS=3600` (`backend/app/config.py:168`, `:177`), one client identity can start three uploads an hour. `MAX_TOTAL_JOBS` defaults to 100 (`:174`).
- Uploads are limited to 10 per 60 s per client (`backend/app/config.py:117`) and downloads, counted separately, to 60 per 3,600 s (`:126-135`), both counted in the process's memory (`backend/app/services/rate_limit.py`, routes at `backend/app/middleware.py:19-20`). Status requests are not limited.
- The client identity is the connection's address by default. With `CLIENT_IDENTITY=lambda_request_context` it is `http.sourceIp` from the `x-amzn-request-context` header, and an upload or download without a usable value gets 503 (`backend/app/client_identity.py:18-46`, `backend/app/config.py:145`).
- With `CORS_ORIGINS` empty, the app adds no CORS middleware (`backend/app/main.py:114-134`).
- Uploads may be up to 50 MiB by default (`backend/app/config.py:114`). The frontend checks the limit set when it is built, `VITE_MAX_UPLOAD_MB`, 50 when unset (`frontend/src/config.ts:1`, `:17`), in two places (`frontend/src/App.tsx:374`, `frontend/src/components/UploadArea.tsx:95`). It retries `/status` with backoff on 429 (`frontend/src/api/client.ts:142`).

### What AWS's documentation says

Read word for word on 2026-11-05 from the AWS pages under References (the Markdown version where one exists), the Lambda Web Adapter's README and source at tag `v1.1.0`, and AWS's public price files published on 2026-10-30.

- Lambda's documentation uses MB to mean 1,024 KB. A synchronous request and a synchronous response are each limited to 6 MB. A streamed response may be up to 200 MB; after the first 6 MB it is sent at up to 2 MBps.
- A function URL's invoke mode is `BUFFERED` by default, with a 6 MB maximum payload. `RESPONSE_STREAM` streams responses of up to 200 MB.
- A function URL's request body is base64-encoded when its content type is binary. The pages read do not say whether multipart form data counts as binary.
- The pages read give no request timeout for function URLs other than the function timeout, which is at most 900 s.
- API Gateway's integration timeout is 50 ms to 29 s for REST APIs, and raising it above 29 s "might require a reduction in your Region-level throttle quota". For HTTP APIs it is 30 s and cannot be raised.
- While an execution environment runs a request, it "is busy and cannot process other requests". When a function's concurrency exceeds its reserved concurrency, its function URL returns HTTP 429. A function URL accepts at most 10 requests per second for each unit of reserved concurrency. Reserved concurrency 0 "stops your Event Function from processing any events until you remove the limit".
- "You can reserve up to the Unreserved account concurrency value minus 100." The default account limit is 1,000 concurrent executions, and "New AWS accounts have reduced concurrency and memory quotas", which AWS raises "automatically based on your usage". The concurrency quota can be raised through Service Quotas.
- Lambda "terminates execution environments every few hours to allow for runtime updates and maintenance", and may terminate one "after a period of inactivity". `/tmp` is "unique to each execution environment".
- A function URL's authentication is `AWS_IAM` or `NONE`. With `NONE`, "your function's resource-based policy is always in effect and must grant public access before your function URL can receive requests", and AWS's example statements for public access carry the condition `lambda:FunctionUrlAuthType` equal to `NONE`.
- When CORS is configured on a function URL, "Lambda automatically adds the CORS headers to all responses through the function URL". If the function also sends CORS headers, GET and POST responses can carry duplicates, and AWS recommends "configuring all CORS settings on the function URL, rather than sending CORS headers manually in the function response".
- The payload Lambda sends for a function URL request includes `requestContext.http.sourceIp`, "The source IP address of the immediate TCP connection making the request". The pages read do not say whether the request's headers include `X-Forwarded-For`.
- The AWS WAF pages list the resource types it protects, and function URLs are not among them. CloudFront can front a function URL with origin access control, which requires `AWS_IAM` authentication.
- A container image must be in Amazon ECR in the same Region as the function, must target one architecture, and may be up to 10 GB uncompressed. The function must run on a read-only filesystem, with a writable `/tmp` of 512 MB to 10,240 MB.
- The init phase is limited to 10 s for on-demand functions; if it does not finish, Lambda retries it at the first invocation.
- Memory can be 128 MB to 10,240 MB; at 1,769 MB a function has the equivalent of one vCPU.
- Duration is billed in GB-seconds, and "Duration charges apply to code that runs in the handler of a function as well as initialization code that is declared outside of the handler." In US East (N. Virginia), x86 compute is $0.0000166667 per GB-second in the first pricing tier; the price files give the same compute and request prices for us-east-2, us-west-1, us-west-2 and ca-central-1. The free tier is one million requests and 400,000 GB-seconds a month.
- With response streaming, "you are charged for each GB written to the response stream by your function. You can stream the first 6 MB per request at no cost." The price is $0.008 per GB, with 100 GB a month free.
- Data transfer out to the internet costs $0.090 per GB for the first 10 TB a month in us-east-1 (price file), and "AWS customers receive 100GB of data transfer out to the internet free each month, aggregated across all AWS Services and Regions". The Lambda pricing page's sentence on data transfer covers transfer "from outside the region the function executed"; the pages read do not state the rate for function URL responses to the internet.
- Lambda's metrics page lists `Duration` ("does not include cold start time") and no metric for bytes streamed. A CloudWatch alarm's actions can "Invoke a Lambda function".
- AWS Budgets sends notifications by email or SNS. Its actions apply IAM or SCP policies or target EC2 or RDS instances; Lambda is not a listed target. "AWS Budgets information is updated up to three times a day." A cost budget can use unblended cost, "actual resource costs charged at time of usage", or net unblended cost, "after all discounts and credits are applied".
- New accounts get USD $100 in credits. On the Paid plan, "You are charged the standard pay-as-you go pricing for any usage beyond your Free Tier credit balance, or if you use AWS services where credits don't apply."
- AWS's ECR policy example grants `ecr:GetAuthorizationToken` with `Resource` set to `*`.
- The Lambda Web Adapter, now in the `aws` GitHub organization, runs an existing HTTP app in a container image and supports function URLs and response streaming. Its defaults are port 8080, readiness path `/`, healthy status 100 to 499, and invoke mode `buffered`. Release 1.1.0 adds `AWS_LWA_READINESS_CHECK_TIMEOUT_SECONDS`; if it is unset the adapter waits indefinitely, and if it expires, init fails. In its source at `v1.1.0` (`src/lib.rs:1591-1595`), the adapter sets `x-amzn-request-context` on each request with `HeaderMap::insert`, which the `http` crate documents as removing all previous values for the name, so a value the caller sends is replaced. `src/lib.rs` at that tag does not mention `X-Forwarded-For`.

### What this account allows

Read by the account owner with the AWS CLI on 2026-11-05, except the last two items, which carry their own dates. The account ID is left out here.

- The account is on the Paid plan, active, with USD $100.00 in credits.
- In us-east-1, Lambda reports 10 concurrent executions for the account, all 10 unreserved. The Service Quotas entry "Concurrent executions" (`L-B99A9384`) is 10 and adjustable. An increase to 1,000 was requested the same day, and the request's status was `CASE_OPENED`.
- The Service Quotas list for Lambda shows no quota for function memory, so whether 3,538 MB is accepted was not checked.
- On 2026-11-06, `L-B99A9384` read 1,000, and Lambda reported 1,000 concurrent executions, all unreserved.
- During the first setup (2026-11-06 to 2026-11-07), `create-function` with 3,538 MB was refused with "'MemorySize' value failed to satisfy constraint: Member must have value less than or equal to 3008".

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

An earlier published image (`sha256:fc1fd3b662e4d304ee7a8c56cb1ba5e27a119fc16a265bad06b1aa3a1996e6ed`, linux/amd64, 3,094,115,281 bytes as `docker image inspect` reports it) ran locally on 2026-11-05 with `docker run --read-only`, a tmpfs at `/tmp`, and `UPLOAD_DIR` and `PROCESSED_DIR` under `/tmp`. `/health` reported healthy, a 1 s upload completed, WAV and MP3 downloads returned 200, and no log line contained an error or a warning. The run used the default OpGAN engine under emulation on an arm64 host; its timings are not used here. CI's image job now runs the same read-only check (`scripts/check_runtime_image.sh`, since `3c8a5d3`).

### What the worst case costs

These figures are arithmetic on the prices above for a 30-day month with one execution environment, before credits. They assume that function URL responses to the internet are billed at the data transfer rate above, which the pages read do not state.

- Compute alone, with the environment busy the whole month at 3,008 MB (decision 5), is about 7.61 million GB-seconds, or about $120.2 after the free tier. At 3,538 MB, the first choice, it was about 8.96 million GB-seconds and $142.6.
- In this worst case, downloads add more than compute. After the first 6 MB of a response, one environment streams at up to 2 MBps. The pages read do not say how fast the first 6 MB go, so the figures here bracket it, for 19.2 MB results and counting 1,000 MB to the GB. If every byte goes at 2 MBps, a result takes 9.6 s, and a month holds 270,000 results and 5,184 GB. If the first 6 MB is not limited, a result takes about 6.6 s, and a month holds about 392,700 results and 7,540 GB. The first 6 MB of each response is not charged as streamed bytes, so 3,564 GB or 5,184 GB is charged, about $28 or $41 after the free 100 GB; data transfer on all of it, after its free 100 GB, is about $458 or $670.
- Taken together, one environment kept busy with downloads for a whole month costs roughly $610 to $830, or up to about $1.15 for each busy hour. At that rate $100 lasts about three and a half days.
- A per-client download limit does not bound this total, since many clients together can keep the environment busy. What bounds it is how long the environment is allowed to stay busy.

## Decision

Stage 1 runs the current image, with the Lambda Web Adapter added and the adapters unchanged, as one Lambda function behind a function URL. The AWS resources were created on 2026-11-06 and 2026-11-07; the deploy job and the frontend changes do not exist yet. They are listed under Work, and the checks under Preconditions and Precondition results.

1. **One function with reserved concurrency 1, and nothing deployed until the account allows it.** At most one execution environment runs. While it lasts, the in-memory `JobStore` and `RateLimiter`, the admission slot and the local `FileStore` hold the state for every request that Lambda passes to the function. Reserving 1 needs at least 101 unreserved concurrent executions, and this account had 10 on 2026-11-05; the quota read 1,000 on 2026-11-06. Deploying without reserved concurrency is ruled out here: Lambda could then run up to 10 environments, a request could reach an environment that does not hold its job, and the cost bound below would grow up to tenfold.
2. **A function URL with `NONE` authentication, not API Gateway.** The slowest measured job, 26.24 s, leaves about 3 s under the REST API's 29 s integration timeout. The pages read give no timeout for a function URL other than the function timeout. Stage 1 runs inline, which ADR 010 section 4 allows where a request can stay open longer than the slowest job.
3. **CORS belongs to the function URL**, allowing only the frontend's origin and the methods GET and POST. `CORS_ORIGINS` is set to an empty value on the function (unset, it defaults to `http://localhost:3000`), so the app adds no CORS headers of its own. That includes the headers `_add_cors` (`backend/app/main.py:114-134`) adds to the app's own 429 and 503 responses; the function URL's headers are expected to cover those responses instead.
4. **The Lambda Web Adapter runs the app.** Release 1.1.0 is in the image, pinned by digest (`Dockerfile:10`), with `AWS_LWA_PORT=8000`, `AWS_LWA_READINESS_CHECK_PATH=/`, `AWS_LWA_READINESS_CHECK_TIMEOUT_SECONDS=100` and `AWS_LWA_INVOKE_MODE=response_stream` (`Dockerfile:94-97`), and the function URL uses invoke mode `RESPONSE_STREAM`. Streaming is expected to let a 600 s WAV, about 19.2 MB, reach the client; precondition 3 checks this. Readiness uses `/`, which runs no checks (`backend/app/routes/health.py:15-25`), because `/health` returns 503 when any of its checks fails and would hold init until the readiness timeout. The timeout was first 8 s, to sit under the 10 s init limit, but on Lambda on 2026-11-07 the app was not ready within 8 s and the adapter failed init, so it is 100 s (`Dockerfile:87-92`). Of the five cold starts observed that day with the longer timeout, in three the 10 s init phase timed out and init finished during the first request; the two latest finished init within the phase, in 6.26 s and 6.10 s. The local `FileStore` streams files from local disk, so the download guard stays (ADR 010, section 3).
5. **Memory 3,008 MB, the most this account accepted, about 1.7 vCPUs.** The first choice was 3,538 MB, two vCPUs at 1,769 MB each: a 600 s job used about 91 GB-seconds on two vCPUs (3,538 MB for 26.24 s) and about 105 on four (7,076 MB for 15.13 s), so the cost per job is similar. Four vCPUs would shorten that job, and the time other users get 429, by about 11 s, but double the cost of each busy hour of compute. During the first setup the account refused more than 3,008 MB (What this account allows). At 3,008 MB on Lambda, two uploads of a 595 s MP3 ran for 21.79 s and 21.20 s by the function's REPORT lines, and the highest memory used in any invocation that day was 1,047 MB.
6. **Function timeout 120 s.** This is a judgment: about 4.6 times the slowest measured job, to leave room for a slower CPU on Lambda and for steps the measurement does not separate. It also caps one hung invocation at 120 s.
7. **`/tmp` of 2,048 MB and `MAX_ARTIFACT_BYTES` of 1 GiB.** With OpGAN, the byte budget reserves 81,125,888 bytes for a 4 MiB upload at the 600 s limit: the upload, the decoded size cap and the peak output (`backend/app/routes/denoise.py:288-292`). The other 1 GiB of `/tmp` is left for files the budget does not count, such as the UVR model directory and library temp files. With `DENOISER_ENGINE=uvr`, one such job reserves 851,160,336 bytes of the 1 GiB.
8. **Uploads are capped at 4 MiB** (`MAX_UPLOAD_BYTES=4194304`, and `VITE_MAX_UPLOAD_MB=4` for the frontend build). A 4 MiB body base64-encodes to 5,592,408 bytes, under the 6 MB limit of 6,291,456 bytes, leaving 699,048 bytes for the multipart envelope and the rest of the event; 5 MiB encodes to 6,990,508 bytes, over the limit. On this deployment, uploads between 4 MiB and 50 MiB get 413, a deliberate difference from today's limit.
9. **Every limit is set on the function, not left to its default.** `CLIENT_IDENTITY=lambda_request_context`, because the adapter replaces any caller copy of `x-amzn-request-context` and the pages read do not say whether an `X-Forwarded-For` header arrives; without one, every caller would reach the app from the adapter's address, `127.0.0.1` by default (adapter `src/lib.rs:449`), which uvicorn trusts (`Dockerfile:116`), and all callers would share one identity. The app refuses to start when `AWS_LAMBDA_FUNCTION_NAME`, which Lambda reserves, is set and the mode is `connection` (`backend/app/config.py:154-166`); the pages read do not confirm that Lambda sets that variable for a container image on a non-AWS base, so precondition 1 still checks the identity in use. `MAX_JOBS_PER_IP=6`, `JOB_TTL_SECONDS=1200` and `MAX_TOTAL_JOBS=24`, as `scripts/deploy_backend.sh:103` sets for Cloud Run, so one client identity can start six uploads in 20 minutes and results stay downloadable for 20 minutes. The upload limit of 10 per 60 s and the download limit of 60 per 3,600 s are set explicitly at their defaults.
10. **The frontend retries `/status` with backoff on HTTP 429.** A 429 from Lambda arrives while another user's job holds the environment. Uploads and downloads are not retried.
11. **Deploys go through ECR by digest.** In the `aws-backend` environment, a deploy job pulls the image the `publish` job pushed to GitHub Container Registry, by the digest of its linux/amd64 image rather than the multi-platform index, pushes it to a private ECR repository with scan on push and immutable tags, and fails unless the digests match. It updates the function's code to that digest, publishes a version, warms it with a signed call, and points the alias `live`, which the function URL serves, at it. It signs in to AWS with GitHub's OIDC token. The environment is not the repository's `Production` environment, which has the frontend deployments that `vercel[bot]` records, so that the reviewer and branch rules stay off those.
12. **Each deploy ends with a smoke test** through the function URL: `/health` returns 200 and healthy, a 1 s WAV upload completes, and its download returns `audio/wav`. If the test fails, the job points `live` back at the previous version.
13. **Budget, alarms and logs.** A $20 monthly budget on unblended cost, so that credits do not hide spending, emails at $5, $10 and $20 of actual cost and when the forecast passes $20. CloudWatch alarms on the function's summed duration and on throttles send email through SNS. The function's log group keeps events for 14 days. Budgets update up to three times a day, so they report spending; they do not stop it.
14. **The kill switch is automatic, with a manual fallback.** An alarm on the function's summed `Duration` over a sliding one-hour window, which CloudWatch evaluates every minute, invokes a small function that sets the function URL's authentication to `AWS_IAM`. The public statement in the resource-based policy is written with AWS's condition on the authentication type, so it applies only while the authentication is `NONE`, and anonymous requests are expected to stop being served; precondition 7 checks this. The small function's role may update this function's URL configuration and write its own logs, and nothing else (`docs/aws-stage-1-runbook.md`, step 8). The threshold is more than 15 minutes of `Duration` in the hour (`docs/aws-stage-1-runbook.md`, step 8), a judgment from the cost of a busy hour above. An operator can also make the same change by hand, or set reserved concurrency to 0 once the account's quota allows reserving. The deploy role can change neither.

### Preconditions before the function is created

0. **Quota.** The account's Lambda concurrency quota is at least 101, so that reserved concurrency 1 can be set.

### Preconditions before the frontend uses the function

These are checked once on the deployed function, and the results are added to this record.

1. **Client identity.** With `CLIENT_IDENTITY=lambda_request_context`, requests from two networks get different identities, and a client-supplied `X-Forwarded-For` or `x-amzn-request-context` does not change the identity. With `LOG_LEVEL=DEBUG` for the check, the rate-limit log line shows the identity (`backend/app/middleware.py:73`). If all callers share one identity, or uploads get 503 because the header is missing, the function is not used for public traffic.
2. **Upload size.** A file of exactly 4 MiB uploads successfully, and a larger one is refused.
3. **Streaming.** A result just under the 600 s limit, about 19 MB, downloads in full; the runbook uses 595 s, since uploads over the limit are refused and MP3 encoding can add padding.
4. **Readable 429.** While a long job runs, a status request with the frontend's `Origin` gets 429 with an `Access-Control-Allow-Origin` header.
5. **Init.** The first criterion, that a cold start's init duration is under 10 s, failed on 2026-11-07. It is amended to: once the image is cached, a cold start answers its first request within 20 s. The 20 s was chosen after the starts in Precondition results were observed. The first start of a new image is not covered by this criterion; a private request after each deploy is meant to take it.
6. **Memory.** The function is accepted with the memory decision 5 sets.
7. **Kill switch.** Triggering the kill switch once switches the URL to `AWS_IAM`, an anonymous request is then refused, and switching back to `NONE` restores service.

### Precondition results

Checked on 2026-11-07 on versions 1 to 4 of the function, all running the linux/amd64 image `sha256:1db760a622685495aa27c37badac957589600c09006c03135866bcb92b76db72` (tag `531b34e`), with the steps in `docs/aws-stage-1-runbook.md`. Times measured from the operator's machine include its network.

0. **Quota: met.** The quota read 1,000 (What this account allows).
1. **Client identity: met.** With `LOG_LEVEL=DEBUG`, an upload sent with a forged `X-Forwarded-For` and a forged `x-amzn-request-context` was logged under the sending network's real address, not the forged one. A request from a phone on a cellular network was logged under a different address. Two download requests from the first network's address, about 20 s before the phone's, are not explained; they were also logged under that network's real address. Whether a function URL request carries `X-Forwarded-For` was not checked.
2. **Upload size: met.** A WAV of 4,194,304 bytes got 200, and one of 4,194,306 bytes got 413 with "File too large (max 4 MB)".
3. **Streaming: met.** A 595 s MP3 of 3,570,740 bytes got 200, 23.4 s after the request was sent. Its download got 200 with 19,040,044 bytes, which matches 595 s of 16 kHz 16-bit mono and a 44-byte header. The download took 62.6 s on the operator's network, about 0.3 MB/s, so it does not show the streaming rate the cost figures use.
4. **Readable 429: met.** While that upload ran a second time, a status request with the frontend's `Origin` got 429 with `Access-Control-Allow-Origin: https://harmonyrestorer.online` and the body reason `ReservedFunctionConcurrentInvocationLimitExceeded`.
5. **Init: the first criterion failed; the amended one was met in the cold starts observed.** With the image's readiness timeout of 8 s, the adapter failed init twice (`Extension.Crash`). With 100 s (decision 4), there were five cold starts, all of the one image. In the first start of the image, the 10 s init phase timed out and the app was ready after about 78 s, inside the first request. In the next two, the init phase also timed out and init finished during the first request; each call took about 16 s. The two latest finished init within the phase, in 6.26 s and 6.10 s. The four starts after the first are the evidence for the amended criterion.
6. **Memory: changed.** 3,538 MB was refused; 3,008 MB was accepted, and `get-function-configuration` reports it.
7. **Kill switch: met.** With the alarm set to `ALARM` by hand, the kill-switch function logged the change, and the URL's `AuthType` read `AWS_IAM` 12 s after the alarm state was set. The first unsigned `/health` request, sent once `AuthType` read `AWS_IAM`, still got 200; the next two, 5 s apart, got 403 with `{"Message":"Forbidden"}`. After the authentication was set back to `NONE`, the first request got 403 and the next, 5 s later, got 200. In each direction, the first request after the configuration read the new value was still handled the old way, and the next, 5 s later, the new way.

### Startup measurements

Measured on 2026-11-07 and 2026-11-08 (UTC) from the function's CloudWatch logs: the adapter's readiness messages, the app's log lines, and the platform's `INIT_REPORT` and `REPORT` lines. On 2026-11-08, `PYTHONPROFILEIMPORTTIME=1` was also set on `$LATEST` for two forced cold starts each time, then removed, and `$LATEST`'s environment was compared with a snapshot taken before. Unless stated otherwise, times in an attempt are counted from the adapter's start of polling, taken as its "not ready after 2000ms" line less 2 s.

- In 9 of the first 11 log streams read, the first attempt ended before the app logged anything. In 8 of them the 10 s init phase timed out and init ran again during the first request. In the other, under the earlier 8 s readiness timeout, the adapter failed init.
- In the attempts that succeeded, the app's first log line came 4.0 to 8.7 s in, 0.3 to 1.4 s passed between it and the ffmpeg check's line, and 0.08 to 0.24 s between that and the checkpoint check's line. One start is left out of these ranges: at 05:15 UTC on 2026-11-07, the first with the 100 s readiness timeout, the first log line came 49.0 s in, followed by 25.3 s and 4.2 s.
- Before `588eac3`, the app imported the UVR wrapper with every engine (`backend/app/services/__init__.py:1`, `backend/app/services/engines.py:10`), and the wrapper imported audio-separator at module level. In two profiled starts, `app.services.denoiser` took 3.74 s and 2.44 s cumulative, of which `audio_separator.separator` took 3.60 s and 2.32 s, including torch (1.94 s and 1.16 s), which was first imported inside it. The start that finished inside the init phase had an init duration of 5.50 s.
- In the one discarded attempt profiled, counted from its first import line, torch finished 8.86 s in and the last import lines, `scipy.stats` modules, came at 9.95 s; `audio_separator.separator` had not finished importing. The imports compared took longer there than in the retry: fastapi 4.27 s against 0.24 s, numpy 0.83 s against 0.11 s, and torch 1.67 s against 1.16 s.
- After `588eac3`, with `$LATEST` running that commit's image (the image of version 9), two starts finished init inside the init phase, in 4.27 s and 4.20 s, with `app.services.denoiser` at 0.19 s and 0.17 s. Their first requests used 433 MB of memory, against 557 MB in the two profiled starts before. The deploy warm-up of version 9 still timed out the 10 s init phase, and its first request, which includes the retried init, took 4.17 s; for the warm-ups of versions 5 to 8, each a new image, it took 8.0 to 10.4 s.
- The `Duration` metric counted init that ran during a request: its datapoint for 05:15 UTC on 2026-11-07 was 78,604 ms, the same as that request's `REPORT` line, although AWS's metrics page says `Duration` does not include cold start time. Such starts therefore count toward the kill switch's alarm (decision 14).
- Cold starts after a quiet period, as a visitor meets them, were not measured on their own.

### Work

- Done in the repository: the image check in CI runs the container read-only with `/tmp` writable (`3c8a5d3`); the frontend takes its upload limit from a build setting (`c826abf`); the frontend retries `/status` on 429 (`a2617bb`); an empty `CORS_ORIGINS` adds no CORS middleware (`abd5282`); the Lambda Web Adapter is in the image (`1bf5aa5`); the client identity can come from the Lambda request context (`7b6ca6d`); downloads are rate limited per client (`8eb0acb`).
- Done in the repository after the first deployment: the image's readiness timeout is 100 s (`d85c00f`).
- Granted: the Lambda concurrency increase, read as 1,000 on 2026-11-06.
- Created in the AWS account on 2026-11-06 and 2026-11-07, with `docs/aws-stage-1-runbook.md`: the budget, the SNS topic and its confirmed email subscription, the ECR repository and its policy, the image, the execution role and the log groups with 14-day retention, the function with reserved concurrency 1, its versions and the alias `live`, the kill-switch function, its role and the two alarms, and the function URL. The function sets the readiness timeout of 100 s in its environment.
- Created on 2026-11-07 (UTC), with `docs/aws-stage-1-runbook.md` step 11: the OIDC provider, the deploy role, a rule keeping the 5 most recent images in the ECR repository, and the GitHub environment `aws-backend` with a required reviewer, deployments from `main` only, and the role's ARN, first as a variable and since 2026-11-08 as a secret.
- Done in the repository for deploys: the `publish` job reports the linux/amd64 digest (`9c2c2f9`), and a `deploy` job runs `scripts/deploy_lambda.sh` after approval (`c3eda74`). Its first run, on 2026-11-08 (UTC, CI run 42508482434), published version 5, moved `live` to it, and passed the smoke test.
- Done in the repository for startup: audio-separator is imported only when a UVR model is built (`588eac3`; Startup measurements).
- Set on Vercel on 2026-11-07 (US Eastern): `VITE_API_URL` to the function URL followed by `/api/v1`, and `VITE_MAX_UPLOAD_MB=4`. The site's bundle built after that contained the function URL and the call `Ov("4")`, which from the surrounding code is the minified upload-limit parser.

### Security design

- **The checks on uploads stay on the server and fail closed.** Size and magic bytes are checked first (`backend/app/routes/denoise.py:252-264`), and decoded duration before inference (ADR 006).
- **No long-lived AWS keys.** The deploy role trusts only the OIDC subject `repo:jacob7choi-xyz/harmonyrestorer-v1:environment:aws-backend` with audience `sts.amazonaws.com`. Its permissions name the one ECR repository and the one function, apart from `ecr:GetAuthorizationToken`, which AWS's own example grants on `*`. It can update code, publish versions, invoke the function's versions to warm them, and update the alias. It cannot change the function's configuration, reserved concurrency, the function URL or its authentication and CORS settings.
- **The function's execution role carries logging only.** Lambda pulls the image through a policy on the ECR repository, since the Lambda page on images says only one of the execution role and the repository policy needs to allow it.
- **The account ID stays out of the repository.** The role's ARN is a GitHub environment secret, and the deploy job sets `mask-aws-account-id: true` on the sign-in action, which by default does not mask it. The first deploy, on 2026-11-08, read the ARN from a variable, and the log line where GitHub prints the step's inputs showed the account ID; that run's logs were deleted the same day, and the ARN was moved to a secret.
- **Reserved concurrency limits how many environments run, and the kill switch limits how long the function keeps serving public requests once its alarm fires; the budget only notifies.** The function URL is public, and no WAF sits in front of it.
- **Logs.** Job IDs appear in INFO logs and work only until their job expires, 20 minutes after completion. Client addresses are logged at DEBUG, and at WARNING only when an upload finds the inference slot busy. The request context header carries the account ID. The app does not log it; the adapter logs request headers at its debug level (adapter `src/lib.rs:1624`), so that level is not turned on for this function.

## Consequences

### Positive
- Compute is billed for the time the environment works, including init, rather than for an always-on instance, and the first deployment needs no new port adapters.
- With the deploy job, the image CI tested would be the image deployed, identified by digest, and a deploy could be undone by pointing `live` at the previous version.
- On the GitHub runner, the slowest job (26.24 s) and the container's peak memory (741 MiB) were within the settings chosen here. On Lambda at 3,008 MB, the 595 s job ran for 21.79 s and 21.20 s, and no invocation that day used more than 1,047 MB (decision 5).

### Negative
- **A visitor who arrives at a cold start waits for it.** The first start of a new image was ready after about 78 s, inside the first request; the later cold starts observed took 6 s to 16 s (Precondition results, 5). When init finishes during the first request, that time appears to count toward the request's 120 s function timeout (`Dockerfile:87-92`).
- **Other users are refused while a job runs.** Lambda answers their requests with 429 for as long as the job runs, 26.24 s for the longest measured job. Their uploads and downloads fail; only status polling retries.
- **Jobs and files are lost when Lambda replaces the environment**, which its documentation says happens every few hours and may happen after inactivity. Each deploy is expected to end every job too, since requests then reach the new version's environment. A status or download request for a lost job returns 404.
- **Uploads above 4 MiB are refused.** 4 MiB holds about 23.8 s of 16-bit 44.1 kHz stereo WAV, or about 4.4 minutes of 128 kbps MP3, against 50 MiB today.
- **Data transfer can cost more than compute.** One client can download 60 times an hour, and many clients together can keep the environment streaming; the worst case above is roughly $610 to $830 a month without the kill switch. On the Paid plan, cost past the $100 in credits is charged to the account's payment method.
- **The kill switch reacts to alarms, not instantly.** Spending between the start of abuse and the alarm's action is not prevented, and the alarm's period and threshold set how much that can be.
- **No WAF.** Reserved concurrency, the kill switch, the app's rate limits and the job caps are the only limits, and the last two depend on the client-identity check.

### Neutral
- `scripts/deploy_backend.sh` stays as it is.
- Lambda freezes the environment between invocations, so the cleanup loop runs only while one is in progress, and expired jobs and files can stay longer than their TTL.

## Stage 2

Stage 2 would lift the limits above by adding adapters behind ADR 010's ports: a `FileStore` on S3 that issues upload and download grants, so files no longer pass through the function, and a `JobStore` on DynamoDB with atomic caps and conditional state changes, plus a request limit shared across environments. ADR 010's rules apply: no networked `JobStore` before a `FileStore` that issues grants and keeps results every instance can read, and a rule, not yet written, for jobs left in PROCESSING. Stage 2 gets its own record.

## Open Questions

- Whether a function URL request carries `X-Forwarded-For`; precondition 1 checks the identity either way.
- The rate at which function URL responses to the internet are billed; the cost figures above assume the general data transfer rate.
- The pages read do not say whether the 6 MB request limit applies to the encoded body. A 4 MiB upload succeeded on the function; larger uploads are refused by the app first, so where the platform's limit falls was not tested.
- How often Lambda replaces the environment under light use, and so how often jobs are lost, is not measured.
- Why the first start of a new image took about 78 s was not determined; the pages read do not say whether pulling the image counts toward the 10 s init limit. In the one discarded attempt profiled, the imports compared were slower than in the retry; why was not determined.
- How long cold starts take once the image is cached, and how often the 10 s init phase times out; the starts observed are in Precondition results, 5, and Startup measurements.
- Whether to set torch's thread count is open; the measurement used 2 threads with 2 and 4 CPUs.
- The questions ADR 010 lists for stage 2 remain: presigned upload size limits, grant expiry, and how a download grant reaches the browser.

## Alternatives Considered

| Alternative | Why not chosen here |
|-------------|---------------------|
| Deploy now without reserved concurrency | Up to 10 environments could run, the in-memory state would split between them, and the cost bound would grow up to tenfold. |
| API Gateway in front of the function | The 29 s REST integration timeout leaves about 3 s above the slowest measured job, and raising it might reduce the account's throttle quota. |
| CloudFront with WAF in front of the function URL | Adds WAF, but requires `AWS_IAM` authentication on the URL and more setup. It can be added if the alarms show abuse. |
| Four vCPUs | Shorter jobs at a similar cost per job, but about twice the cost of each busy hour of compute (decision 5), and more memory than this account accepted on 2026-11-07. |
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
- https://docs.aws.amazon.com/lambda/latest/dg/monitoring-metrics-types.html
- https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarm-actions.html
- https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-execution-service-limits-table.html
- https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-quotas.html
- https://docs.aws.amazon.com/waf/latest/developerguide/waf-chapter.html
- https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-lambda.html
- https://docs.aws.amazon.com/AmazonECR/latest/userguide/security_iam_id-based-policy-examples.html
- https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-controls.html
- https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html
- https://docs.aws.amazon.com/cost-management/latest/userguide/create-cost-budget.html
- https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html
- https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html
- https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-container-services.html
- https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-overview.html
- https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html

Other sources, read on 2026-11-05:

- https://aws.amazon.com/lambda/pricing/, and AWS's public price files for Lambda (us-east-1, us-east-2, us-west-1, us-west-2, ca-central-1) and for data transfer (us-east-1), published 2026-10-30
- https://aws.amazon.com/ec2/pricing/on-demand/ (data transfer out allowance)
- https://github.com/aws/aws-lambda-web-adapter, README and `src/lib.rs` at tag `v1.1.0`, and release notes for `v1.1.0`
- https://docs.rs/http/1.3.1/http/header/struct.HeaderMap.html (`insert`), and the request context types in `lambda_http` 1.3.0 and `aws_lambda_events` 1.2.0, the versions the adapter's `Cargo.lock` pins
