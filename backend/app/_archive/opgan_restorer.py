"""Inference wrapper for the OpGAN generator.

Handles resampling, stereo channels, fixed-size framing with overlap-add,
crossfade stitching, and peak normalization — so the generator only ever
sees clean [B, 1, frame_len] tensors.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample as ta_resample

logger = logging.getLogger(__name__)


def _to_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert audio to [C, T] float32 tensor.

    Args:
        x: Audio as [T] mono or [C, T] stereo, float32/float64.

    Returns:
        Tensor [C, T] float32 on CPU.

    Raises:
        ValueError: If shape is not [T] or [C, T] with C in {1, 2}.
    """
    t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)

    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim == 2 and t.shape[0] in (1, 2):
        pass
    else:
        raise ValueError("Audio must be shape [T] or [C, T] with C in {1, 2}.")

    return t.to(torch.float32).contiguous()


def _peak_normalize_(wav: torch.Tensor, headroom: float = 0.9) -> float:
    """In-place peak normalization to keep |x| <= headroom.

    Args:
        wav: Audio tensor to normalize in-place.
        headroom: Target peak level (default 0.9).

    Returns:
        Applied scale factor (<= 1.0).
    """
    peak = torch.amax(torch.abs(wav))
    if peak > 0:
        scale = min(1.0, headroom / float(peak))
        if scale < 1.0:
            wav.mul_(scale)
        return scale
    return 1.0


def _make_crossfade(fade: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Create linear crossfade windows that sum to 1.0 at each index."""
    if fade <= 0:
        return None, None
    fade_out = torch.linspace(1.0, 0.0, steps=fade, dtype=torch.float32)
    fade_in = 1.0 - fade_out
    return fade_out, fade_in


class OpGANRestorer:
    """Inference wrapper for the OpGAN generator.

    Responsibilities:
        - Resample to model SR (default 16kHz) and back
        - Stereo handling (process L/R independently)
        - Fixed-size framing (32k samples) with overlap
        - Linear crossfade in overlaps (artifact-free stitching)
        - Batching frames for throughput
        - Peak normalization to avoid clipping

    Args:
        generator: Trained OpGAN generator module.
        model_sr: Sample rate the generator expects.
        frame_len: Frame length in samples.
        overlap: Overlap ratio between adjacent frames [0, 1).
        fade_ms: Crossfade duration in milliseconds.
        batch_frames: Number of frames per generator batch.
        headroom: Peak normalization target.
        progress_cb: Optional callback (progress: float, stage: str).
    """

    def __init__(
        self,
        generator: torch.nn.Module,
        *,
        model_sr: int = 16_000,
        frame_len: int = 32_000,
        overlap: float = 0.5,
        fade_ms: int = 20,
        batch_frames: int = 8,
        headroom: float = 0.9,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> None:
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1)")

        self.g = generator.eval()
        self.model_sr = int(model_sr)
        self.frame_len = int(frame_len)
        self.overlap = float(overlap)
        self.fade_ms = int(fade_ms)
        self.batch_frames = int(batch_frames)
        self.headroom = float(headroom)
        self.progress_cb = progress_cb

        # Precompute crossfade length (clamped to hop size)
        hop = max(1, int(self.frame_len * (1.0 - self.overlap)))
        fade = int(self.model_sr * self.fade_ms / 1000.0)
        self.fade = max(0, min(fade, hop))
        self.fade_out, self.fade_in = _make_crossfade(self.fade)

        # Device from model parameters (fallback: best available)
        try:
            self.device = next(self.g.parameters()).device
        except StopIteration:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self._check_generator_signature()

    # ---------- public API ----------

    @torch.inference_mode()
    def restore_track(self, audio: np.ndarray | torch.Tensor, sr: int) -> np.ndarray:
        """Run full restoration pipeline.

        Args:
            audio: [T] mono or [C, T] stereo, float32 in [-1, 1].
            sr: Input sample rate.

        Returns:
            Restored audio as np.ndarray with same shape and original SR.
        """
        self._emit(0.0, "prepare")

        x = _to_tensor(audio)
        C, T_in = x.shape

        _peak_normalize_(x, headroom=self.headroom)

        if int(sr) != self.model_sr:
            x = self._resample_channels(x, from_sr=sr, to_sr=self.model_sr)

        # Process each channel independently (generator expects mono)
        outs = []
        for ch in range(C):
            self._emit(0.05 + 0.45 * (ch / max(1, C - 1)), f"process_ch_{ch}")
            y_ch = self._restore_channel(x[ch : ch + 1, :])
            outs.append(y_ch)

        y = torch.cat(outs, dim=0)

        if int(sr) != self.model_sr:
            y = self._resample_channels(y, from_sr=self.model_sr, to_sr=sr)

        # Match original length exactly after round-trip resampling
        T_out = y.shape[1]
        if T_out > T_in:
            y = y[:, :T_in]
        elif T_out < T_in:
            y = torch.nn.functional.pad(y, (0, T_in - T_out))

        y = torch.clamp(y, -1.0, 1.0)
        self._emit(0.98, "finalize")
        return y.cpu().numpy()

    # ---------- helpers ----------

    def _emit(self, prog: float, stage: str) -> None:
        """Fire progress callback if set."""
        if self.progress_cb is not None:
            try:
                self.progress_cb(float(max(0.0, min(1.0, prog))), stage)
            except Exception:
                logger.debug("Progress callback failed for stage %s", stage)

    def _check_generator_signature(self) -> None:
        """Verify generator accepts [B, 1, frame_len] and returns same shape."""
        dummy = torch.zeros(1, 1, self.frame_len, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            out = self.g(dummy)

        if not isinstance(out, torch.Tensor):
            raise TypeError("Generator must return a torch.Tensor")
        if list(out.shape) != [1, 1, self.frame_len]:
            raise ValueError(
                f"Generator must return [B, 1, {self.frame_len}], got {list(out.shape)}"
            )

    def _resample_channels(
        self, x: torch.Tensor, *, from_sr: int, to_sr: int
    ) -> torch.Tensor:
        """Resample each channel independently with torchaudio."""
        if from_sr == to_sr:
            return x
        C, _ = x.shape
        resampled = []
        for c in range(C):
            y_c = ta_resample(x[c], orig_freq=from_sr, new_freq=to_sr)
            resampled.append(y_c.unsqueeze(0))
        return torch.cat(resampled, dim=0)

    def _restore_channel(self, x_ch: torch.Tensor) -> torch.Tensor:
        """Restore a single mono channel via overlap-add.

        Args:
            x_ch: [1, T] mono audio at model_sr.

        Returns:
            [1, T] restored audio.
        """
        x_ch = x_ch.to(self.device)

        _, T = x_ch.shape
        N, hop = self._num_frames_and_hop(T)

        # Pad so last frame is fully covered
        pad_needed = (N - 1) * hop + self.frame_len - T
        if pad_needed > 0:
            x_ch = torch.nn.functional.pad(x_ch, (0, pad_needed))

        starts = [i * hop for i in range(N)]

        out = torch.zeros_like(x_ch)
        weights = torch.zeros_like(x_ch)

        fade = int(self.fade)
        fade_out = self.fade_out.to(x_ch.device) if fade > 0 else None
        fade_in = self.fade_in.to(x_ch.device) if fade > 0 else None

        batches = [
            starts[i : i + self.batch_frames]
            for i in range(0, len(starts), self.batch_frames)
        ]
        total = len(batches)
        done = 0

        for bidx, batch_starts in enumerate(batches):
            B = len(batch_starts)
            frames = torch.zeros(
                B, 1, self.frame_len, dtype=torch.float32, device=self.device
            )
            for i, s in enumerate(batch_starts):
                seg = x_ch[:, s : s + self.frame_len]
                frames[i, 0, : seg.shape[1]] = seg

            with torch.inference_mode():
                yb = self.g(frames)

            for i, s in enumerate(batch_starts):
                idx_global = bidx * self.batch_frames + i
                seg = yb[i, 0]
                e = s + self.frame_len

                win = torch.ones_like(seg)
                if fade > 0 and fade_in is not None and fade_out is not None:
                    if idx_global > 0:
                        win[: self.fade] = fade_in
                    if idx_global < N - 1:
                        win[-self.fade :] = torch.minimum(win[-self.fade :], fade_out)

                out[0, s:e] += seg * win
                weights[0, s:e] += win

            done += 1
            self._emit(0.5 + 0.45 * (done / total), "overlap_add")

        # Normalize overlapping regions
        eps = 1e-8
        mask = weights > eps
        out[mask] = out[mask] / weights[mask]

        return out[:, :T]

    def _num_frames_and_hop(self, T: int) -> tuple[int, int]:
        """Calculate number of frames and hop size for a given length."""
        hop = max(1, int(self.frame_len * (1.0 - self.overlap)))
        if T <= self.frame_len:
            return 1, hop
        n = 1 + (max(0, T - self.frame_len) + hop - 1) // hop
        return n, hop


@torch.inference_mode()
def restore_file(
    input_path: str,
    output_path: str,
    restorer: OpGANRestorer,
) -> None:
    """Convenience function: restore audio file to file.

    Args:
        input_path: Path to input audio file.
        output_path: Path to write restored audio.
        restorer: Configured OpGANRestorer instance.
    """
    wav, sr = torchaudio.load(input_path)
    y = restorer.restore_track(wav.numpy(), sr)
    torchaudio.save(output_path, torch.from_numpy(y), sr)
