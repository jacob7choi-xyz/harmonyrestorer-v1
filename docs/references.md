# References

Research papers relevant to HarmonyRestorer's audio restoration approach, organized by topic.

## Core Music Restoration and Historical Recordings

1. **Moliner & Valimaki** -- "A Two-Stage U-Net for High-Fidelity Denoising of Historical Recordings" (ICASSP 2022)
   Time-frequency two-stage U-Net targeting shellac/78-RPM artifacts (hiss, crackle, thumps). Strong baseline for historical music denoising.
   [arXiv:2202.08702](https://arxiv.org/abs/2202.08702)

2. **Fonseca et al.** -- "Learning to Denoise Historical Music" (ISMIR 2020)
   SEANet audio-to-audio model with adversarial + reconstruction losses. Synthetic pairs from clean music + real historical noise segments. Gold-standard recipe for paired training data.
   [arXiv:2008.02027](https://arxiv.org/abs/2008.02027)

3. **Diffusion-Based Denoising of Historical Recordings** (Colibri/MDIB report)
   First serious treatment of diffusion models for historical audio denoising. Generative counterpart to U-Net and SEANet for analog artifacts.

4. **Kiranyaz et al.** -- "Blind Restoration of Real-World Audio by 1D Operational GANs" (2022)
   Time-domain 1D Op-GAN handling multiple unknown artifacts (noise, clipping, bandwidth loss, reverberation) in a blind setting. **This is the architecture HarmonyRestorer implements.**
   [arXiv:2212.14618](https://arxiv.org/abs/2212.14618)

5. **Boukun et al.** -- "A Variational Autoencoder Approach for Denoising and Inpainting" (Interspeech 2024)
   TVAE-based blind zero-shot restoration for denoising and inpainting. Unified generative view of dropouts + noise in music.

## Architecture Foundations (Self-ONNs, Op-GANs, Time vs T-F)

6. **Ince et al.** -- "Self-Organized Operational Neural Networks with Generative Neurons" (2020)
   Core Self-ONN paper: generative neurons and learned operators. **Underpins the 1D Op-GAN architecture used in HarmonyRestorer.**
   [arXiv:2004.11778](https://arxiv.org/abs/2004.11778)

7. **"Joint Time-Frequency and Time Domain Learning for Speech Enhancement"** (2020)
   Contrasts time-domain vs T-F processing and proposes a joint architecture. Useful conceptual reference when choosing domains for restoration.

## GAN-Based Enhancement and Super-Resolution

8. **Kuleshov et al.** -- "Adversarial Audio Super-Resolution with Unsupervised Feature Losses"
   Canonical GAN-based audio super-resolution. Maps low-bandwidth audio to wideband with feature losses from a pre-trained network.

9. **AudioSR** -- "AudioSR: Versatile Audio Super-resolution at Scale"
   Diffusion-based super-resolution for music, speech, and SFX. Upsamples 2-16 kHz bandwidth to 24 kHz / 48 kHz sampling rate.
   [Project page](https://audioldm.github.io/audiosr/)

10. **FlashSR** -- "FlashSR: One-step Versatile Audio Super-resolution via Diffusion"
    Single-step diffusion for audio SR. Low-step, high-quality generation relevant for latency-sensitive restoration.
    [arXiv:2501.10807](https://arxiv.org/abs/2501.10807)

11. **"Audio Super-Resolution with Latent Bridge Models"** (OpenReview 2026)
    Latent bridge model for bandwidth extension with arbitrary input-output sampling rates.

## Robust Enhancement and Blind Denoising

12. **Kim et al.** -- "HiFi-GAN: High-Fidelity Denoising and Dereverberation Based on Deep Learning"
    GAN-based model mapping noisy/reverberant audio to clean studio-like quality. Architecture and loss design widely reused in music enhancement.

13. **Zhao** -- "Robust Speech Enhancement in Noisy and Reverberant Environments Using Deep Neural Networks" (2020)
    Comprehensive study of DNN robustness to noise and reverberation. Good template for evaluating robust music models.

14. **Madhav et al.** -- "Speech Denoising without Clean Training Data: a Noise2Noise Approach"
    Learns denoising from noisy-noisy pairs only. Relevant if clean stems are unavailable for training data.

## Synthetic Data Generation

15. **Fonseca et al.** -- training procedure in "Learning to Denoise Historical Music"
    Extracts pure noise from silent sections of historical records and mixes with clean modern music. This is the approach HarmonyRestorer's dataset pipeline is based on (see also entry #2).
    [arXiv:2008.02027](https://arxiv.org/abs/2008.02027)

## Priority Reading Order

For HarmonyRestorer specifically:

1. Kiranyaz 2022 (our architecture)
2. Ince 2020 (Self-ONN theory behind our model)
3. Moliner & Valimaki 2022 (closest competitor to benchmark against)
4. Fonseca 2020 (synthetic noise recipe similar to ours)
5. AudioSR (alternative approach if OpGAN underperforms)

## Evaluation Metrics

Papers above commonly use these metrics for benchmarking:

- **SDR** (Signal-to-Distortion Ratio) -- overall signal quality improvement
- **PESQ** (Perceptual Evaluation of Speech Quality) -- perceptual quality score
- **STOI** (Short-Time Objective Intelligibility) -- intelligibility measure
- **SI-SNR** (Scale-Invariant Signal-to-Noise Ratio) -- source separation quality
