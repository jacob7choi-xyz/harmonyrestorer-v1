"""
ML Models Package for HarmonyRestorer

This package contains the core AI models for audio restoration:
- Self-Organized Operational Neural Networks (Self-ONNs)
- 1D Operational GANs for blind audio restoration
- Loss functions and training utilities

Based on the research paper:
"Blind Restoration of Real-World Audio by 1D Operational GANs"
"""

from .self_onn import SelfONN, Conv1DSelfONN

# Version info
__version__ = "1.0.0"
__author__ = "HarmonyRestorer Team"

# Package-level imports for easy access
__all__ = [
    "SelfONN",
    "Conv1DSelfONN",
]

# Optional: Import Op-GAN components when they're available
try:
    from .op_gan import OpGANGenerator, OpGANDiscriminator, OpGANLoss
    __all__.extend([
        "OpGANGenerator", 
        "OpGANDiscriminator", 
        "OpGANLoss"
    ])
except ImportError:
    # Op-GAN components not yet available
    pass

print("🎵 HarmonyRestorer ML Models package loaded successfully!")