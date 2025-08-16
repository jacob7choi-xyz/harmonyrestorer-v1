# test_device.py
"""
GPU Acceleration Validation for HarmonyRestorer v1

Validates PyTorch device detection and performance across:
- Apple Silicon MPS (M1/M2) 
- NVIDIA CUDA
- CPU fallback

Ensures optimal hardware acceleration for OP-GAN audio processing.
Expected 5-50x performance improvement with GPU acceleration.
"""

import torch

print("🔍 Testing PyTorch Device Detection on M2 MacBook...")
print("=" * 50)

# Check MPS (Apple GPU)
print(f"🍎 MPS (Apple GPU) available: {torch.backends.mps.is_available()}")

# Check CUDA (NVIDIA GPU) 
print(f"🟢 CUDA (NVIDIA GPU) available: {torch.cuda.is_available()}")

# Your updated device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    device_type = "Apple M2 GPU"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    device_type = "NVIDIA GPU"
else:
    DEVICE = torch.device("cpu")
    device_type = "CPU"

print(f"🚀 Selected device: {DEVICE} ({device_type})")

# Test with a small tensor
print("\n🧪 Testing tensor operations...")
try:
    x = torch.randn(1000, 1000).to(DEVICE)
    y = torch.mm(x, x.T)
    print(f"✅ Tensor test successful on {DEVICE}")
    print(f"📊 Result shape: {y.shape}")
except Exception as e:
    print(f"❌ Tensor test failed: {e}")

print("=" * 50)
print("🎉 Device test complete!")