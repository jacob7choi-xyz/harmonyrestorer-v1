"""
Complete test script to verify all ML models are working correctly
Run this from the backend/ directory to test everything
"""

import torch
import torch.nn as nn
import sys
import traceback

def test_self_onn():
    """Test Self-ONN components"""
    print("🧠 Testing Self-ONN Components...")
    
    try:
        from app.ml_models.self_onn import OptimizedSelfONN as SelfONN, OptimizedConv1DSelfONN as Conv1DSelfONN
        
        # Test basic Self-ONN
        layer = SelfONN(input_size=64, output_size=32, q=5)
        test_input = torch.randn(8, 64)
        output = layer(test_input)
        
        assert output.shape == (8, 32), f"Expected (8, 32), got {output.shape}"
        print(f"  ✅ SelfONN: {test_input.shape} → {output.shape}")
        
        # Test operator usage
        usage = layer.get_operator_usage()
        assert usage.shape == (32, 5), f"Expected (32, 5), got {usage.shape}"
        print(f"  ✅ Operator usage tracking: {usage.shape}")
        
        # Test Conv1D Self-ONN
        conv_layer = Conv1DSelfONN(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2, q=5)
        audio_input = torch.randn(4, 1, 1000)
        conv_output = conv_layer(audio_input)
        
        expected_length = (1000 + 2*2 - 5) // 2 + 1  # 500
        assert conv_output.shape == (4, 16, 500), f"Expected (4, 16, 500), got {conv_output.shape}"
        print(f"  ✅ Conv1DSelfONN: {audio_input.shape} → {conv_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Self-ONN test failed: {e}")
        traceback.print_exc()
        return False

def test_op_gan():
    """Test Op-GAN components"""
    print("\n🎵 Testing Op-GAN Components...")
    
    try:
        from app.ml_models.op_gan import OpGANGenerator, OpGANDiscriminator, OpGANLoss
        
        # Test Generator
        generator = OpGANGenerator(input_length=32000, q=5)
        test_audio = torch.randn(2, 1, 32000)
        generated = generator(test_audio)
        
        assert generated.shape == (2, 1, 32000), f"Expected (2, 1, 32000), got {generated.shape}"
        print(f"  ✅ Generator: {test_audio.shape} → {generated.shape}")
        
        # Test Discriminator
        discriminator = OpGANDiscriminator(input_length=32000, q=5)
        disc_output = discriminator(generated)
        
        # Discriminator output should be [batch, 1, some_length]
        assert disc_output.shape[0] == 2, f"Expected batch size 2, got {disc_output.shape[0]}"
        assert disc_output.shape[1] == 1, f"Expected 1 channel, got {disc_output.shape[1]}"
        print(f"  ✅ Discriminator: {generated.shape} → {disc_output.shape}")
        
        # Test Loss Function
        loss_fn = OpGANLoss(lambda_temporal=10, lambda_spectral=5)
        target_audio = torch.randn(2, 1, 32000)
        
        # Test generator loss
        gen_loss, loss_dict = loss_fn.generator_loss(disc_output, generated, target_audio)
        assert isinstance(gen_loss, torch.Tensor), "Generator loss should be a tensor"
        assert 'adversarial' in loss_dict, "Loss dict should contain adversarial loss"
        assert 'temporal' in loss_dict, "Loss dict should contain temporal loss"
        assert 'spectral' in loss_dict, "Loss dict should contain spectral loss"
        print(f"  ✅ Generator Loss: {gen_loss.item():.4f}")
        
        # Test discriminator loss
        real_output = discriminator(target_audio)
        disc_loss = loss_fn.discriminator_loss(real_output, disc_output.detach())
        assert isinstance(disc_loss, torch.Tensor), "Discriminator loss should be a tensor"
        print(f"  ✅ Discriminator Loss: {disc_loss.item():.4f}")
        
        # Test operator usage summary
        usage_summary = generator.get_operator_usage_summary()
        assert len(usage_summary) == 10, f"Expected 10 layers, got {len(usage_summary)}"
        print(f"  ✅ Operator usage summary: {len(usage_summary)} layers")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Op-GAN test failed: {e}")
        traceback.print_exc()
        return False

def test_model_parameters():
    """Test model parameter counts"""
    print("\n📊 Testing Model Parameters...")
    
    try:
        from app.ml_models.op_gan import OpGANGenerator, OpGANDiscriminator
        
        generator = OpGANGenerator(input_length=32000, q=5)
        discriminator = OpGANDiscriminator(input_length=32000, q=5)
        
        gen_params = sum(p.numel() for p in generator.parameters())
        disc_params = sum(p.numel() for p in discriminator.parameters())
        total_params = gen_params + disc_params
        
        print(f"  ✅ Generator parameters: {gen_params:,}")
        print(f"  ✅ Discriminator parameters: {disc_params:,}")
        print(f"  ✅ Total parameters: {total_params:,}")
        
        # Check if parameters are reasonable (should be less than 100M for efficiency)
        assert total_params < 100_000_000, f"Model too large: {total_params:,} parameters"
        print(f"  ✅ Model size is reasonable")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Parameter test failed: {e}")
        traceback.print_exc()
        return False

def test_imports():
    """Test all imports work correctly"""
    print("\n📦 Testing Package Imports...")
    
    try:
        # Test package-level imports
        from app.ml_models import SelfONN, Conv1DSelfONN
        print("  ✅ Package-level Self-ONN imports")
        
        from app.ml_models import OpGANGenerator, OpGANDiscriminator, OpGANLoss
        print("  ✅ Package-level Op-GAN imports")
        
        # Test direct imports
        from app.ml_models.self_onn import OptimizedSelfONN as DirectSelfONN
        from app.ml_models.op_gan import OpGANGenerator as DirectGenerator
        print("  ✅ Direct module imports")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 HarmonyRestorer ML Models - Complete Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_self_onn,
        test_op_gan,
        test_model_parameters
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
            
    print("\n" + "=" * 60)
    print(f"🧪 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your ML models are ready! 🚀")
        print("\nNext steps:")
        print("  1. Create training script")
        print("  2. Prepare audio dataset")
        print("  3. Train the Op-GAN model")
        print("  4. Integrate with FastAPI backend")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)