# ADR-010: Ports for Job State, Files, Rate Limits and Dispatch

**Date:** 2026-11-03

**Status:** accepted

---

## Context

The goal is a backend that can run on more than one provider: AWS with pay-per-use billing first, Google Cloud later. The backend ships as a container image, which CI publishes to the GitHub Container Registry (ADR 008). Much of what ties the backend to one long-lived process is where it keeps its state.

### State the backend keeps in one process

The application code was read at commit `612fc64`.

| State | Where | Assumes one process because |
|---|---|---|
| Job records | `backend/app/services/jobs.py:46` | A status or download request that reaches another process returns 404 (`backend/app/routes/denoise.py:361`, `:383`) |
| Job caps (global and per IP) | `jobs.py:101-123` | The count and the insert run under a `threading.Lock` (`jobs.py:48`), which only serialises threads within one process |
| Download guard | `jobs.py:49` | It is an in-memory record that guards a file the app streams from its own local disk (`denoise.py:393`) |
| Result and upload files | `settings.upload_dir`, `settings.processed_dir` | They are on the local disk of one instance |
| Rate limiter counts | `backend/app/middleware.py:44` | Each process counts separately, so N processes would allow up to N times the limit |
| Expiry | `jobs.py:233`, run every 300 s by `backend/app/main.py:58`, started at `main.py:105` | It deletes only the records and files this process holds |
| Inference admission | `denoise.py:41`, one slot by default (`backend/app/config.py:129`) | It limits concurrent inference within one process |
| Artifact byte budget | `backend/app/services/artifacts.py:121` | It counts files on one instance's local disk |
| Transcode slot | `backend/app/services/transcode.py:32` | It limits concurrent conversions within one process |

The deployment script states the invariant intended to keep this correct: "max instances x workers per instance x inference slots = 1" (`scripts/deploy_backend.sh:24`). The image runs one worker (`Dockerfile:93-95`). ADR 001 accepted in-memory job storage and an in-memory rate limiter as MVP trade-offs, and rejected a Redis job queue "from day one" as premature (`docs/adr/001-enterprise-refactor.md:83`, `:86`, `:99`).

### How an upload runs today

Inference runs inside the upload request. The handler starts the job as a task (`denoise.py:322-323`), waits for it behind `asyncio.shield` so that a cancelled request does not cancel the inference (`denoise.py:341-342`), and replies only when the job has finished. The handler's docstring calls this a transitional contract (`denoise.py:183-187`). The deployment script ties it to Cloud Run's request-based CPU: "inference runs inside the request, so throttled CPU is correct" (`deploy_backend.sh:10-11`).

The frontend already polls for the result after uploading. It takes the `job_id` from the upload response and polls `/status/{id}` every 1.5 s until the job is `completed` or `failed`, or polling times out (`frontend/src/App.tsx:264`, `frontend/src/api/client.ts:62-130`). With today's backend, the first poll would normally see `completed`, because the job has finished before the upload replies. This was read from the code, not observed in a browser.

### What the platforms document

Read on 2026-11-03 at the URLs under References.

- AWS Lambda freezes an execution environment once the runtime and its extensions have finished and no events are pending. Its documentation says that background processes that did not complete when the function ended "resume if Lambda reuses the execution environment", and asks that background work complete before the code exits.
- A synchronous Lambda invocation accepts and returns at most "6 MB each for request and response". The maximum function timeout is 900 seconds for standard functions.
- For a Regional API Gateway REST API, the integration timeout can be set from 50 milliseconds to 29 seconds. The quota can be raised, which the documentation says might require reducing the account's throttle quota.
- DynamoDB deletes expired items "within a few days of their expiration time", and its documentation advises filtering expired items out of query results until then. A conditional write fails if its condition is false, which can be used for compare-and-set state changes.
- An SQS event source mapping processes each message at least once, and the Lambda documentation recommends idempotent processing.
- AWS WAF rate-based rules accept a minimum limit of 10 requests over windows of 60, 120, 300 or 600 seconds, and the documentation says they are "not intended for precise request-rate limiting".
- A Lambda execution environment that is processing a request "cannot process other requests", and each environment has its own `/tmp`. Lambda Managed Instances are an exception and support concurrent invocations; this ADR does not consider them.
- Cloud Run allocates CPU only during request processing under request-based billing, and for the whole instance lifetime under instance-based billing.

### What was measured

The runtime image was measured on a GitHub runner with 4 vCPUs (run 40984036981, commit `7eeeabd` on branch `measure/runtime-sizing`). This was one run, with random noise as input and the image started from local layers. With 600 s of audio, one job took about 25 to 26 s when limited to 2 vCPUs and 14 to 15 s on all 4. Startup to a healthy `/health` took 3.4 to 3.8 s. The container's peak memory (cgroup) was about 741 MiB.

Sizes that matter against the 6 MB limit:
- Uploads are capped at 50 MiB (`config.py:86`).
- A 600 s result, extrapolated from a 2 s clip converted with ffmpeg 9.0.2 (the image uses 7.1.5), is about 19.2 MB as WAV, 9.0 MB as FLAC and 1.9 MB as MP3.

## Decision

### 1. Four ports

Each port is a `typing.Protocol`, as `Denoiser` already is (`backend/app/services/engines.py:36`, ADR 005).

| Port | Responsibility |
|---|---|
| `JobStore` | Create, read and change job records, enforce the job caps, and report expiry |
| `FileStore` | Store and read uploads and results, and issue time-limited upload and download grants |
| `RateLimiter` | Decide whether one more upload from a client identity is allowed |
| `Dispatcher` | Run an accepted job, either before the upload replies or after |

### 2. The first adapters keep today's HTTP behaviour

The first adapter for each port is intended to reproduce what the code does now: an in-memory `JobStore`, a local-disk `FileStore`, an in-memory `RateLimiter` and an `InlineDispatcher`. "No behaviour change" means that the HTTP API keeps the same status codes, headers and response fields for the same requests. Values such as job IDs and timestamps differ per request, as they do today. The one intended exception is the expiry edge case described under "Expiry is computed, not observed" in section 3.

The `InlineDispatcher` must keep three properties of `denoise.py:322-342`:
- a cancelled request does not cancel the inference;
- the admission slot and the byte reservation have one owner at a time: the handler until the task is created, then the task, which releases both when it finishes (`denoise.py:61-74`);
- if the task cannot be created, the handler releases the slot, and the reservation if it made one (`denoise.py:335-339`).

Tests that reach into the current private state are expected to change to use the ports. These include the lifecycle tests in `backend/tests/test_admission.py:47-133`, `_clear_rate_limiter` (`backend/tests/conftest.py:13`), the seeding of `job_manager._jobs` (for example `conftest.py:26`), and the direct use of `_TRANSCODE_SLOTS` in `backend/tests/test_transcode.py`.

### 3. Rules every adapter follows

- **Caps are enforced atomically on create.** Creating a job checks the global and per-IP caps and inserts the record as one atomic step. How a cloud adapter achieves this is decided in that adapter's ADR. An adapter that cannot do it does not satisfy the port.
- **State changes are conditional.** A change names the state it expects, for example QUEUED to PROCESSING, and fails if the record is in another state. A job moves to PROCESSING at most once, and processing must be safe to repeat after a failure.
- **Expiry is computed, not observed.** A job counts as expired from its timestamps and the configured TTL. Reads treat an expired job as absent, whether or not it has been deleted. Physical deletion is cleanup and may happen later. For the local adapter this changes one edge case: today `get_job` does not check the TTL (`jobs.py:126-129`), so a job past its TTL is still served until the next cleanup pass, up to 300 s later by default; under this rule it is treated as absent once more than the TTL has passed since it finished.
- **Downloads are bounded by time.** When a `FileStore` serves results through download grants, the app no longer streams the file, and the in-memory download guard no longer applies. A grant is issued only if it expires before the job does. Files are deleted only after their job expires, so a grant is never valid after its file may have been deleted. A `FileStore` that streams results from local disk, as the first local adapter does, keeps the download guard; only an adapter that issues grants removes it.

### 4. Dispatch

The `Dispatcher` port is built now, and `InlineDispatcher` is its first adapter. Whether a platform runs jobs inline or through a queue is decided per platform, when its adapter is written, with these criteria.

| Inline fits a platform when | A queue fits a platform when |
|---|---|
| A request can stay open longer than the slowest job, for example on Lambda, where the function timeout can be up to 900 s | The entry point has a shorter timeout than the slowest job, for example API Gateway's 29 s default against about 25 s measured for 600 s of audio on 2 vCPUs |
| The fewest moving parts matter most | Retries matter, or a job must survive the client disconnecting |
| | Processing is idempotent, since queues deliver at least once |

The frontend polls until the job is finished, so both are expected to work without frontend changes. This was read from the code (`frontend/src/api/client.ts:62-130`), not tested with a queue.

### 5. One composition root

One module builds the adapters at startup from configuration. An unknown adapter name stops startup, as an unknown `DENOISER_ENGINE` does when settings are loaded (`config.py:59-67`; the engine is built by `engines.py:97`). Routes and services receive their adapters from this module instead of importing module-level singletons such as `job_manager` (`jobs.py:276`).

### 6. Contract tests

Each port has one test suite that every adapter must pass. The local adapters run it in CI. How cloud adapters are tested (emulators or real accounts) is decided with the first cloud adapter.

### 7. What does not become a port

Inference admission, the artifact byte budget, the transcode slot, the loaded model and the startup checks stay per process. Each concerns the CPU, memory, local scratch space or local tools of one instance. On Lambda, as documented above, an environment that is processing a request does not take another, and each has its own `/tmp`, so they would still apply per environment.

### 8. Build order

1. `RateLimiter`. It is the smallest, and HTTP tests already cover it (`backend/tests/test_security.py`), so it should exercise the composition root and the contract-test layout with less risk than the other ports.
2. `JobStore`, which is carved out of `JobManager`.
3. `FileStore`.
4. `Dispatcher`.

Each step is its own plan and commits, with its contract tests in the same commits.

> **Note:** Progress as of 2026-11-04. Steps 1 and 2 have landed: `RateLimiter` in commits `1b83802`, `e69a70c` and `e77a29c`, and `JobStore` in commits `25ed37a`, `1325dbe`, `f17821f` and `e63d59b`.
>
> The in-memory `JobStore` is narrower than the expiry rule in section 3 in two ways. A job exactly at its TTL still reads as present, because expiry needs strictly more than the TTL, as cleanup used before. Among the reads, only `get` and `expired_ids` treat an expired job as absent. `counts` includes every stored record until cleanup deletes it, and an expired COMPLETED job still counts toward the caps until then, because its files stay until cleanup removes them.
>
> Two rules follow for later adapters:
>
> - No networked `JobStore` should go live before a `FileStore` that issues grants replaces the download guard and keeps results where every instance can read them. The guard protects a download's files from cleanup until a cleanup pass finds the hold older than `DOWNLOAD_TTL_SECONDS`, and only while no `await` pauses between a route reading the job and taking the hold, or between cleanup reading the guards and deleting the files and records (`cleanup_expired` in `backend/app/services/jobs.py`). Separately, with results on local disk, a shared record could point to a file on another instance, and the download route would answer 404 (the `output_path.exists()` check in `backend/app/routes/denoise.py`). The in-memory store's methods never await; a store reached over the network would normally await on network I/O.
> - A durable `JobStore` needs a rule for jobs left in PROCESSING. If the process is killed while the worker thread runs, or the inference task is cancelled, nothing records a final status (`_process_job` in `backend/app/routes/denoise.py`), so the job stays PROCESSING and keeps its place under the caps. An in-memory store loses such records when the process restarts; a durable store would keep them.

### Security

- Storage keys are built only from the server-generated UUID job ID (`denoise.py:239`), never from a client filename or other client input.
- The magic-byte check, the ffmpeg decode and the fail-closed duration check stay on the server and run before inference, including when the client uploads directly to storage through a grant. The server also checks the stored object's size after upload, whatever the grant enforces. The byte reservation is computed from the duration cap (`denoise.py:246-252`), so weakening the cap would weaken the memory budget.
- The rate limiter keys on a client identity produced by one function, and which proxies are trusted to report the client address is set in the composition root. Today `FORWARDED_ALLOW_IPS` is not set by the deployment script and defaults to `127.0.0.1` in the image (`Dockerfile:95`).
- Each cloud adapter uses credentials limited to its own resources, and grants expire after a short time.

## Consequences

### Positive
- A second provider is expected to need new adapters and a deployment, rather than changes to the routes or the job lifecycle.
- The contract tests give evidence that an adapter behaves like the local one, for the behaviour the suite covers.
- Conditional state changes and computed expiry should also make it safer to run more than one instance on a single provider.

### Negative
- More code and indirection: four protocols, four local adapters and a composition root before any cloud adapter exists.
- Several tests that reach into private state are expected to be rewritten against the ports.
- Atomic caps on a shared store can cost extra reads or writes per upload, depending on the adapter.

### Neutral
- The upload API's request and response fields are not expected to change. Whether the first poll already sees `completed` depends on the dispatcher.
- The cleanup loop stays for the local adapters. Cloud adapters may rely on their store's expiry instead, since reads already treat expired jobs as absent.

## Open Questions

- What "a grant is issued only if it expires before the job does" (section 3) means for a request that arrives close to the job's expiry: capping the grant's expiry at the job's, or refusing the grant. A grant issued after a format conversion would likely need the same rule.
- How uploads would reach a Lambda function. A synchronous invocation accepts at most 6 MB and uploads may be up to 50 MiB, so uploads would likely need a grant, which would change the upload request that the Neutral consequences above expect to stay the same.
- How a download grant would reach the browser. If the API answered with a redirect to a signed URL, MDN lists redirects across origins among the cases where the `Origin` header is `null`; the playback `fetch` in `frontend/src/App.tsx` would then need the bucket's CORS to allow the request, and the Docker frontend's `connect-src 'self'` (`frontend/nginx.conf`) would likely block it. None of this has been tested in a browser.
- Whether an S3 presigned upload can enforce a maximum object size. The S3 pages for presigned POST policies could not be read with the fetch tool used on 2026-11-03. The size check after upload (Security, above) does not depend on the answer.
- Whether a Lambda function URL has its own payload limit beyond the 6 MB synchronous limit, and whether AWS WAF can sit in front of a function URL directly.
- The DynamoDB maximum item size, and Firestore's TTL timing.
- Which client address Cloud Run passes to the app, and therefore what `FORWARDED_ALLOW_IPS` should be. This needs a live request log.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Stay single-process on Cloud Run only | Ties the backend to one provider's model of one long-lived instance; the project's goal is provider-neutral |
| Redis and a task queue such as Celery from the start | Usually adds infrastructure that runs, and is billed, while idle, which works against pay-per-use hosting; ADR 001 rejected it for the MVP |
| Commit to a queue for every platform now | The frontend code is expected to work with both, and inline is simpler where requests can stay open; the choice can be made per platform with measurements |
| Make admission and the byte budget shared | They limit one instance's CPU and scratch space, which stay per instance on the platforms considered here |

## References

- AWS Lambda execution environment: https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtime-environment.html
- AWS Lambda quotas: https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html
- API Gateway REST API quotas: https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-execution-service-limits-table.html
- DynamoDB TTL: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html
- DynamoDB condition expressions: https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Expressions.ConditionExpressions.html
- Lambda with SQS: https://docs.aws.amazon.com/lambda/latest/dg/with-sqs.html
- AWS WAF rate-based rule settings: https://docs.aws.amazon.com/waf/latest/developerguide/waf-rule-statement-type-rate-based-high-level-settings.html
- AWS WAF rate-based rule caveats: https://docs.aws.amazon.com/waf/latest/developerguide/waf-rule-statement-type-rate-based-caveats.html
- AWS Lambda concurrency: https://docs.aws.amazon.com/lambda/latest/dg/lambda-concurrency.html
- Cloud Run CPU allocation: https://docs.cloud.google.com/run/docs/configuring/cpu-allocation
