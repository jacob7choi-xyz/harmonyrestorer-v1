# ADR-007: Check the OpGAN Checkpoint's Digest Before Loading It

**Date:** 2026-10-25

**Status:** accepted

---

## Context

The trained checkpoint, `checkpoints/final.pt`, is not in the repository (`docs/benchmarks.md:20`). The runtime image copies whatever file is at that path when it is built (`Dockerfile:76`), and the OpGAN service loaded whatever file `OPGAN_CHECKPOINT` named (`config.py`). No code in `backend/app/` hashed the checkpoint or compared it with a recorded digest.

Loading with `torch.load(..., weights_only=True)` limits what a checkpoint can execute while it is loaded. It does not identify the weights. In a scratch run on 2026-10-25, on macOS with MPS rather than in the image, a checkpoint saved with the same keys and randomly initialized weights logged "OpGAN model loaded (epoch 999)" with no warning, and on the same 2 s input it produced 16 kHz output with an RMS of 0.0141, against 0.2029 from `final.pt`. This shows that a substituted file is not detected; it does not show how likely a substitution is.

After the change, in the linux/amd64 runtime image (emulated on Apple silicon), an upload with `final.pt` completed with 16 kHz mono 16-bit output. With the substitute mounted through `OPGAN_CHECKPOINT`, an upload returned HTTP 200 with `"status":"failed"`, and the ERROR log showed both digests.

`benchmark/manifests/training_corpus.json` already records `checkpoint_sha256`, and the local `final.pt` matches it. Nothing read that field.

## Decision

1. **The expected digest is a constant.** `OPGAN_CHECKPOINT_SHA256` in `config.py` holds the SHA-256 of the trained checkpoint. It is not read from the environment: an environment variable would be set in the same place as `OPGAN_CHECKPOINT`, so one wrong configuration could set both. Serving a different model therefore needs a code change. This does not by itself mean the change is reviewed: `scripts/deploy_backend.sh` builds from the local working tree. The benchmark protocol also asks for a fresh partition for a model changed in response to test results (`docs/benchmark-protocol.md:167-169`), and the manifest records the trained model's digest. `build_denoiser` passes the constant to the service, which requires it as a keyword argument.
2. **The digest is checked when the model is loaded.** The service reads the file once, hashes those bytes, and passes the same bytes to `torch.load` through `io.BytesIO`. The file is not read a second time, so it cannot change between the check and the load, and bytes that fail the check never reach the loader. Hashing the 18,027,171-byte file took 0.0089 s in one run on macOS, probably from the page cache.
3. **A mismatch fails closed.** The service logs both digests at ERROR and raises `CheckpointMismatchError`, and the generator is not cached, so the next job checks again. The job fails through the existing handler in `JobManager.process`. There is no setting to skip the check.
4. **The two records are tied by a test.** `benchmark/tests/test_training_corpus_manifest.py` asserts that the constant equals the manifest's `checkpoint_sha256`. It lives there because the image's test stage copies only `pyproject.toml` and `backend/` (`Dockerfile:66-67`), and the manifest is in neither; pytest's `pythonpath` lets it import `app.config`.

## Consequences

### Positive
- In the backend service, a checkpoint other than the one the benchmark manifest records is refused before it is deserialized.
- The check covers the image, local development and an `OPGAN_CHECKPOINT` override alike.

### Negative
- Deploying a retrained model needs a commit that changes the constant and the manifest together.
- The checkpoint's bytes, 18,027,171 for `final.pt`, are held in memory while they are hashed and loaded. After a successful load this happens once per process; after a mismatch nothing is cached, so each OpGAN job reads and hashes the file again.
- CI cannot run the check against the real checkpoint, because the file is not in the repository. Only local runs show that `final.pt` matches.
- The check runs on the first OpGAN job, not at startup, so `/health` does not report a mismatched or missing checkpoint.

  > **Note:** Updated 2026-10-26. With the OpGAN engine, the app now also reads and hashes the checkpoint once at startup, and `/health` reports `opgan_checkpoint_at_startup` and returns 503 until the process restarts if the file was missing or mismatched (both run), or could not be read (from the code, which catches `OSError`; not run). `/health` reflects the startup result only: a file replaced after startup leaves it at 200 while the load-time check above refuses the file, and a file fixed after startup leaves it at 503 until a restart. The file is read twice per process. With the UVR engine the check is not run and not reported.

### Neutral
- `dataset/infer.py` and `dataset/train.py` load checkpoints with their own `torch.load` calls and are not covered.
- UVR model files are downloaded by audio-separator without a hash check (`SECURITY.md`); this decision does not change that.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| An `OPGAN_CHECKPOINT_SHA256` environment variable | Would be set in the same place as `OPGAN_CHECKPOINT`, so one wrong configuration could set both. |
| Reading the digest from the benchmark manifest at runtime | The runtime image does not contain `benchmark/`, and the backend does not import from it. |
| A digest file next to the checkpoint | Someone able to replace the checkpoint would likely be able to replace a file beside it. |
| Checking the digest in the Dockerfile at build time | Covers only the image and adds a third copy of the digest. |
| Hashing the path, then loading the path | The file could change between the two reads. |
| A startup check reported in `/health` | Left as a separate change. In the image run with the substitute, `/health` returned 200 before and after the upload whose job failed. Added on 2026-10-26; see the note under Negative. |
