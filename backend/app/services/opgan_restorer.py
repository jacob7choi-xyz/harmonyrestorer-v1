# backend/app/services/opgan_restorer.py
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample as ta_resample


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Accepts [T] mono or [2, T] stereo float32/float64 in [-1, 1].
    Returns torch tensor [C, T] float32 on CPU.
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(x)

    if t.ndim == 1:
        t = t.unsqueeze(0)  # [1, T]
    elif t.ndim == 2 and t.shape[0] in (1, 2):
        pass
    else:
        raise ValueError("Audio must be shape [T] or [C, T] with C in {1,2}.")

    return t.to(torch.float32).contiguous()


def _peak_normalize_(wav: torch.Tensor, headroom: float = 0.9) -> float:
    """
    In-place peak normalization to keep |x| <= headroom (default 0.9).
    Returns the applied scale factor (<= 1.0). We do NOT scale back after restore.
    """
    peak = torch.amax(torch.abs(wav))
    if peak > 0:
        scale = min(1.0, headroom / float(peak))
        if scale < 1.0:
            wav.mul_(scale)
        return scale
    return 1.0


def _make_crossfade(fade: int) -> tuple[torch.Tensor, torch.Tensor]:
    if fade <= 0:
        return None, None
    # Linear complements that sum to exactly 1.0 at each index.
    fade_out = torch.linspace(1.0, 0.0, steps=fade, dtype=torch.float32)  # 1 → 0
    fade_in  = 1.0 - fade_out                                             # 0 → 1
    return fade_out, fade_in


class OpGANRestorer:
    """
    Inference wrapper for your OpGAN generator.

    Responsibilities:
      • Resample to model SR (default 16k) and back
      • Stereo handling (process L/R independently)
      • Fixed-size framing (32k samples) with overlap
      • Cosine crossfade in overlaps (artifact-free)
      • Batching frames for speed
      • Peak normalization to avoid clipping

    Assumptions validated with you:
      - Model SR = 16_000 Hz
      - Generator expects [B, 1, 32000]
      - Process each stereo channel separately
      - Use torchaudio for I/O/resampling
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
        progress_cb: Optional[Callable[[float, str], None]] = None,  # (progress, stage)
    ):
        assert 0.0 <= overlap < 1.0, "overlap must be in [0,1)"
        self.g = generator.eval()  # inference only
        self.model_sr = int(model_sr)
        self.frame_len = int(frame_len)
        self.overlap = float(overlap)
        self.fade_ms = int(fade_ms)
        self.batch_frames = int(batch_frames)
        self.headroom = float(headroom)
        self.progress_cb = progress_cb

        # Precompute crossfade length in samples (clamped to hop)
        hop = max(1, int(self.frame_len * (1.0 - self.overlap)))
        fade = int(self.model_sr * self.fade_ms / 1000.0)
        self.fade = max(0, min(fade, hop))  # fade cannot exceed hop
        self.fade_out, self.fade_in = _make_crossfade(self.fade)

        # Choose device from the model (fallback CPU)
        try:
            self.device = next(self.g.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # Guard: generator must accept [B, 1, frame_len]
        self._check_generator_signature()

    # ---------- public API ----------

    @torch.inference_mode()
    def restore_track(self, audio: np.ndarray | torch.Tensor, sr: int) -> np.ndarray:
        """
        Run the full restoration pipeline on float audio.

        Args:
            audio: [T] mono or [2, T] stereo, float32 in [-1,1]
            sr:    input sampling rate

        Returns:
            np.ndarray with same channel layout and original sr.
        """
        self._emit(0.0, "prepare")

        # To [C, T] float32 tensor on CPU
        x = _to_tensor(audio)  # [C, T]
        C, T_in = x.shape

        # Peak-normalize (we do NOT scale back; keeps headroom)
        _ = _peak_normalize_(x, headroom=self.headroom)

        # Resample to model SR if needed (channel-wise)
        if int(sr) != self.model_sr:
            x = self._resample_channels(x, from_sr=sr, to_sr=self.model_sr)

        # Process per channel to satisfy mono generator
        outs = []
        for ch in range(C):
            self._emit(0.05 + 0.45 * (ch / max(1, C - 1)), f"process_ch_{ch}")
            y_ch = self._restore_channel(x[ch:ch+1, :])  # [1, Tm]
            outs.append(y_ch)

        y = torch.cat(outs, dim=0)  # [C, Tm]

        # Resample back to original SR if needed
        if int(sr) != self.model_sr:
            y = self._resample_channels(y, from_sr=self.model_sr, to_sr=sr)

        # Ensure exact original length after round-trip resampling
        T_out = y.shape[1]
        if T_out > T_in:
            y = y[:, :T_in]
        elif T_out < T_in:
            y = torch.nn.functional.pad(y, (0, T_in - T_out))

        # Clamp to [-1,1] and return as numpy [C, T]
        y = torch.clamp(y, -1.0, 1.0)
        self._emit(0.98, "finalize")
        return y.cpu().numpy()

    # ---------- helpers ----------

    def _emit(self, prog: float, stage: str) -> None:
        if self.progress_cb:
            try:
                self.progress_cb(float(max(0.0, min(1.0, prog))), stage)
            except Exception:
                pass

    def _check_generator_signature(self) -> None:
        # Quick shape sanity check using zeros
        dummy = torch.zeros(1, 1, self.frame_len, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            out = self.g(dummy)
        assert isinstance(out, torch.Tensor), "Generator must return a torch.Tensor"
        assert list(out.shape) == [1, 1, self.frame_len], (
            f"Generator must return [B,1,{self.frame_len}], got {list(out.shape)}"
        )

    def _resample_channels(self, x: torch.Tensor, *, from_sr: int, to_sr: int) -> torch.Tensor:
        """Resample each channel independently with torchaudio."""
        if from_sr == to_sr:
            return x
        C, T = x.shape
        y = []
        for c in range(C):
            y_c = ta_resample(x[c], orig_freq=from_sr, new_freq=to_sr)  # [T’]
            y.append(y_c.unsqueeze(0))
        return torch.cat(y, dim=0)

    def _restore_channel(self, x_ch: torch.Tensor) -> torch.Tensor:
        """
        x_ch: [1, Tm] at model_sr
        Returns: [1, Tm] restored
        """
        _, T = x_ch.shape
        N, hop = self._num_frames_and_hop(T)
        # Pad so we cover last frame fully
        pad_needed = (N - 1) * hop + self.frame_len - T
        if pad_needed > 0:
            x_ch = torch.nn.functional.pad(x_ch, (0, pad_needed))

        # Build frame start indices
        starts = [i * hop for i in range(N)]
        idxs = list(range(N))

        # Prepare output buffer
        out = torch.zeros_like(x_ch)
        weights = torch.zeros_like(x_ch)

        # Crossfade windows to device
        fade = int(self.fade)
        fade_out = self.fade_out.to(x_ch.device) if fade > 0 else None
        fade_in = self.fade_in.to(x_ch.device) if fade > 0 else None

        # Batch frames for generator
        batches = [starts[i:i + self.batch_frames] for i in range(0, len(starts), self.batch_frames)]
        total = len(batches)
        done = 0

        for bidx, batch_starts in enumerate(batches):
            B = len(batch_starts)
            frames = torch.zeros(B, 1, self.frame_len, dtype=torch.float32, device=self.device)
            for i, s in enumerate(batch_starts):
                seg = x_ch[:, s:s + self.frame_len]
                frames[i, 0, : seg.shape[1]] = seg

            with torch.inference_mode():
                yb = self.g(frames)  # [B,1,W]

            # >>> minimal change: compute global frame index
            for i, s in enumerate(batch_starts):
                idx_global = bidx * self.batch_frames + i
                seg = yb[i, 0]  # [W]
                e = s + self.frame_len

                # Build the window for this segment: ones, then apply edge fades
                win = torch.ones_like(seg)
                if self.fade > 0:
                    fi = self.fade_in.to(seg.device)
                    fo = self.fade_out.to(seg.device)
                    if idx_global > 0:            # not first frame → fade-in on start
                        win[:self.fade] = fi
                    if idx_global < N - 1:        # not last frame → fade-out on end
                        win[-self.fade:] = torch.minimum(win[-self.fade:], fo)

                # Apply window to the segment and accumulate
                out[0, s:e] += seg * win
                weights[0, s:e] += win

            # <<< minimal change ends
            done += 1
            self._emit(0.5 + 0.45 * (done / total), "overlap_add")

        # Normalize by weights to handle overlaps    
        eps = 1e-8
        mask = weights > eps
        out[mask] = out[mask] / weights[mask]

        # Trim to original length T
        return out[:, :T]

    def _num_frames_and_hop(self, T: int) -> tuple[int, int]:
        hop = max(1, int(self.frame_len * (1.0 - self.overlap)))
        if T <= self.frame_len:
            return 1, hop
        # ceil division for frames
        n = 1 + (max(0, T - self.frame_len) + hop - 1) // hop
        return n, hop


# ---------- Convenience CLI helpers (optional) ----------

@torch.inference_mode()
def restore_file(
    input_path: str,
    output_path: str,
    restorer: OpGANRestorer,
) -> None:
    """
    Minimal file-in/file-out using torchaudio.
    """
    wav, sr = torchaudio.load(input_path)     # [C, T] float32/float64, C in {1,2}
    y = restorer.restore_track(wav.numpy(), sr)  # [C, T’] np.float32
    torchaudio.save(output_path, torch.from_numpy(y), sr)
