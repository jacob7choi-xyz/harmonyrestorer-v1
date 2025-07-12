"""
ML Models Package for HarmonyRestorer

This package contains the core AI models for audio restoration:
- Self-Organized Operational Neural Networks (Self-ONNs)
- 1D Operational GANs for blind audio restoration
- Loss functions and training utilities
- Gradient health monitoring and optimization tools

Based on the research paper:
"Blind Restoration of Real-World Audio by 1D Operational GANs"

Enhanced with gradient stability improvements and performance optimizations.
"""

from .self_onn import (
    OptimizedSelfONN as SelfONN, 
    OptimizedConv1DSelfONN as Conv1DSelfONN,
    GradientHealthMonitor,
    SelfONNOptimizer,
    PerformanceProfiler
)

# Version info
__version__ = "1.1.0"  # Updated version for stability improvements
__author__ = "HarmonyRestorer Team"

# Package-level imports for easy access
__all__ = [
    "SelfONN",
    "Conv1DSelfONN",
    "GradientHealthMonitor",
    "SelfONNOptimizer", 
    "PerformanceProfiler"
]

# Optional: Import Op-GAN components when they're available
try:
    from .op_gan import (
        OpGANGenerator, 
        OpGANDiscriminator, 
        OpGANLoss,
        GradientHealthMonitor as OpGANGradientMonitor  # Alias to avoid conflicts
    )
    __all__.extend([
        "OpGANGenerator", 
        "OpGANDiscriminator", 
        "OpGANLoss",
        "OpGANGradientMonitor"
    ])
    
    # Create convenient factory function
    def create_op_gan_models(input_length=32000, q_gen=3, q_disc=2):
        """
        Factory function to create optimized Op-GAN models
        
        Args:
            input_length: Audio segment length (default: 32000 for 2s at 16kHz)
            q_gen: Number of operators in generator Self-ONNs (default: 3)
            q_disc: Number of operators in discriminator Self-ONNs (default: 2)
            
        Returns:
            tuple: (generator, discriminator, loss_function)
        """
        generator = OpGANGenerator(input_length=input_length, q=q_gen)
        discriminator = OpGANDiscriminator(input_length=input_length, q=q_disc)
        loss_fn = OpGANLoss()
        
        return generator, discriminator, loss_fn
    
    __all__.append("create_op_gan_models")
    
except ImportError as e:
    # Op-GAN components not yet available or have issues
    print(f"⚠️  Op-GAN components not loaded: {e}")
    pass

# Performance and debugging utilities
def get_package_info():
    """Get package information and model statistics"""
    info = {
        'version': __version__,
        'author': __author__,
        'available_models': __all__,
        'gradient_monitoring': 'GradientHealthMonitor' in __all__,
        'op_gan_available': 'OpGANGenerator' in __all__
    }
    return info

def test_model_imports():
    """Test all model imports and return status"""
    status = {}
    
    # Test Self-ONN imports
    try:
        model = SelfONN(64, 32, q=3)
        conv_model = Conv1DSelfONN(1, 16, 5, q=3)
        status['self_onn'] = '✅ Working'
    except Exception as e:
        status['self_onn'] = f'❌ Error: {e}'
    
    # Test Op-GAN imports
    if 'OpGANGenerator' in __all__:
        try:
            gen, disc, loss = create_op_gan_models()
            status['op_gan'] = '✅ Working'
        except Exception as e:
            status['op_gan'] = f'❌ Error: {e}'
    else:
        status['op_gan'] = '⚠️ Not available'
    
    # Test utilities
    try:
        monitor = GradientHealthMonitor()
        profiler = PerformanceProfiler()
        status['utilities'] = '✅ Working'
    except Exception as e:
        status['utilities'] = f'❌ Error: {e}'
    
    return status

# Add to exports
__all__.extend(['get_package_info', 'test_model_imports'])

print("🎵 HarmonyRestorer ML Models package loaded successfully!")
print(f"📦 Version: {__version__}")
print(f"🧠 Available models: {len(__all__)} components")

# Quick status check
if __name__ == "__main__":
    print("\n🧪 Running import tests...")
    status = test_model_imports()
    for component, result in status.items():
        print(f"  {component}: {result}")
    
    print(f"\n📊 Package info:")
    info = get_package_info()
    for key, value in info.items():
        print(f"  {key}: {value}")