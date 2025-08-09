import numpy as np
import torch
import pytest

from backend.app.services.opgan_restorer import OpGANRestorer


# ---------- Fixtures ----------

class IdentityGen(torch.nn.Module):
    """Pass-through 'model' that mirrors your generator's interface."""
    def __init__(self, frame_len=32_000):
        super().__init__()
        self.frame_len = frame_len
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # keep same clamp behavior as your real generator
        return torch.clamp(x, -1.0, 1.0)

@pytest.fixture(scope="module")
def sr_model() -> int:
    return 16_000

@pytest.fixture(scope="module")
def frame_len() -> int:
    return 32_000

@pytest.fixture(scope="module")
def identity_gen(frame_len):
    return IdentityGen(frame_len=frame_len).eval()

@pytest.fixture(scope="module")
def restorer(identity_gen, sr_model, frame_len):
    return OpGANRestorer(
        identity_gen,
        model_sr=sr_model,
        frame_len=frame_len,
        overlap=0.5,
        fade_ms=20,
        batch_frames=4,
        headroom=0.9,
    )


# ---------- Helpers ----------

def stereo_tones(sr: int, seconds: float) -> np.ndarray:
    """Two pure tones at different freqs, different gains."""
    T = int(sr * seconds)
    t = np.linspace(0, seconds, T, endpoint=False, dtype=np.float32)
    left  = 0.3 * np.sin(2*np.pi*440.0*t).astype(np.float32)
    right = 0.2 * np.sin(2*np.pi*554.37*t).astype(np.float32)
    return np.stack([left, right], axis=0)  # [2, T]


# ---------- Tests ----------

def test_ola_identity_is_perfect(restorer, sr_model):
    """Crossfaded overlap-add should perfectly reconstruct with identity gen."""
    x = stereo_tones(sr_model, seconds=4.3)             # not multiple of 2s
    y = restorer.restore_track(x, sr_model)             # [2, T]
    assert y.shape == x.shape
    err = float(np.max(np.abs(y - x)))
    assert err < 2e-5, f"Overlap-add introduced artifacts: max abs err {err}"


def test_shapes_padding_and_trim(restorer, sr_model):
    """Arbitrary length should be preserved after pad/trim logic."""
    T = int(sr_model * 2.7)
    x = np.random.randn(2, T).astype(np.float32) * 0.01
    y = restorer.restore_track(x, sr_model)
    assert y.shape == x.shape
    assert y.dtype == np.float32


@pytest.mark.parametrize("sr_in", [22_050, 24_000, 44_100])
def test_resample_roundtrip_shape_preserved(restorer, sr_in):
    """Input sr != model sr should roundtrip to the same shape."""
    T = int(sr_in * 1.31)
    x = np.random.randn(1, T).astype(np.float32) * 0.01  # mono is fine
    y = restorer.restore_track(x, sr_in)
    assert y.shape == x.shape


def test_headroom_normalization_applied(restorer, sr_model):
    """If input peaks >1.0, we pre-normalize to <=0.9 and keep it safe."""
    T = int(sr_model * 1.0)
    x = np.zeros((1, T), dtype=np.float32)
    x[0, T//3:T//3+100] = 1.2  # a short clip > 1.0
    y = restorer.restore_track(x, sr_model)
    peak = float(np.max(np.abs(y)))
    assert peak <= 0.9 + 1e-4, f"Peak normalization not applied, got {peak:.3f}"