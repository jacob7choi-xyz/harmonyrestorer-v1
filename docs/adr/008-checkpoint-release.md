# ADR-008: Publish the OpGAN Checkpoint as a Release Asset So CI Can Build the Runtime Image

**Date:** 2026-10-29

**Status:** accepted

---

## Context

The runtime image copies `checkpoints/final.pt` when it is built (`Dockerfile:76`). The file is not in the repository: `.gitignore:71` (`*.pt`) and `.gitignore:86` (`checkpoints/`) exclude it, and it has never been committed. Before this decision, CI built only the test stage (`.github/workflows/ci.yml:80`), and ADR 007 lists as a negative consequence that CI cannot run the digest check against the real checkpoint.

The image that is deployed is built elsewhere. `scripts/deploy_backend.sh:81` builds the default target from the local working tree and pushes it, and `scripts/deploy_backend.sh:89` deploys it by tag. Lines 83-85 look up the pushed digest only to print it, and print "unknown" if the lookup fails. Before this decision, no CI workflow built the runtime stage.

The checkpoint is 18,027,171 bytes. Loading it with `torch.load(..., weights_only=True)` shows six entries: `epoch` (100), `generator_state_dict`, `discriminator_state_dict`, `optimizer_g_state_dict`, `optimizer_d_state_dict` and `best_val_loss`. It contains no audio files. The training data is a separate set of files, 33 GB under `data/` (ADR 002), and this decision does not change where it lives.

The training recordings come from the Internet Archive item `MusopenCollectionAsFlac` (`dataset/acquire.py:41`). On 2026-10-28 the item's metadata gave `licenseurl` as `http://creativecommons.org/publicdomain/mark/1.0/`, the Public Domain Mark 1.0. This was checked at the item level only, not for each recording. The item is in the Internet Archive's `community` collection, and the label was not confirmed with Musopen.

## Decision

1. **The checkpoint is published as an asset of an immutable GitHub Release.** Release immutability is enabled for the repository before the release is created, because it applies only to releases created afterwards. The release, tagged `opgan-checkpoint-v1`, is created as a draft, `final.pt` is attached, the digest GitHub records for the asset is compared with `OPGAN_CHECKPOINT_SHA256`, and only then is the release published. GitHub's documentation says that once such a release is published, its assets cannot be replaced or deleted and its tag cannot be moved. The release notes offer the file under the MIT license and credit the training source.
2. **CI builds and starts the runtime image with the real checkpoint.** The image job downloads the asset over HTTPS and compares its SHA-256 with `checkpoint_sha256` in `benchmark/manifests/training_corpus.json`, which a test keeps equal to `OPGAN_CHECKPOINT_SHA256`. That comparison only makes a wrong download fail before the build. The check that guards loading is still the one in `read_verified_checkpoint` (ADR 007). The script makes the file readable by all users, because the runtime stage copies it with its mode and runs as a non-root user. The job then builds `--target runtime`, checks that no compiler tool is on the image's `PATH`, starts the container with `--network none`, and requires `/health` to return 200 (`scripts/check_runtime_image.sh`). With the default OpGAN engine, `/health` returns 200 only if the startup digest check passed (`backend/app/routes/health.py:47-51`). The startup check hashes the file without loading the weights, so the job also sends one audio file through the API, which checks the digest again and loads the weights.
3. **CI's permissions do not change.** The repository is public, and `scripts/fetch_checkpoint.py` downloads the asset without credentials. The workflow keeps `contents: read` (`.github/workflows/ci.yml:16-17`).

## Consequences

### Positive
- The image job builds the runtime stage with the checkpoint the backend expects, and checks that it starts, passes its digest check and completes one job.
- The checkpoint has a copy outside the local machine.

### Negative
- The weights are public. Deleting the release later would not recall copies already downloaded, and GitHub's documentation says an immutable release's tag name cannot be reused after the release is deleted.
- The image job depends on downloading the asset from github.com, and fails if the download fails.
- Whether model weights can be licensed as copyrighted work is not settled here; the release notes state the terms on offer. The training data's license was checked at the item level only.
- A retrained model needs a new release under a new tag, as well as the commit that changes the constant and the manifest together (ADR 007).

### Neutral
- The deployed image is still built by `scripts/deploy_backend.sh` from the local working tree, so it is not yet the image CI builds. The backend test suite still runs in the test stage, not the runtime stage.
- The checkpoint also holds the discriminator and optimizer state, which the backend does not read. Publishing only the generator would change the file's digest and the benchmark manifest, so it is left out.

## Remaining Work

These depend on where the backend is deployed and are not part of this decision. Deployment to Google Cloud is deferred while the deployment is made independent of any one provider.

- A CI job that pushes the tested runtime image to the GitHub Container Registry on pushes to `main`. It would be the only job given `packages: write`, it would log in only after every build and test step, and it would push the image it tested rather than a rebuild.
- Deploying that image by digest. `scripts/deploy_backend.sh` targets Cloud Run, deploys by tag, and prints "unknown" when it cannot read the digest (lines 83-85); a deploy by digest would stop instead.
- A choice of visibility for the container package, which would contain the weights. GitHub's documentation says a newly published package is private by default and that a public package cannot be made private again. Which visibility a platform can pull from depends on the platform. For Cloud Run, its documentation (last updated 2026-10-26 when read) says it can deploy public GitHub Container Registry images directly, and private ones through an Artifact Registry remote repository. Other platforms were not checked.

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Commit the checkpoint to the repository | The file would stay in every clone's history; removing it later would need a history rewrite. |
| Git LFS | Adds a client tool for contributors and CI. A release asset needed no new tooling. |
| A storage bucket with a cloud provider | Needs an account with that provider, and ties CI to it. |
| Keep the checkpoint only on the local machine | CI cannot build or check the runtime image. |
| A release without immutability | The asset could be replaced under the same tag. The digest check would refuse a replaced file, but immutability prevents the replacement. |
| Publish only the generator weights | Changes the digest and the benchmark manifest for a smaller file; not needed for this decision. |
