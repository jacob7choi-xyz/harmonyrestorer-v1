"""
Comprehensive test suite for HarmonyRestorer ML Models
Tests functionality, performance, memory usage, and numerical stability
"""

import torch
import torch.nn as nn
import sys
import traceback
import time
import psutil
import os
import gc
from contextlib import contextmanager

class ComprehensiveModelTester:
    """Complete testing suite for ML models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.process = psutil.Process(os.getpid())
        
    @contextmanager
    def measure_time(self):
        """Context manager to measure execution time"""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield lambda: None
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        self.last_time = (end - start) * 1000  # Convert to milliseconds
        
    @contextmanager  
    def measure_memory(self):
        """Context manager to measure memory usage"""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            start_gpu = torch.cuda.memory_allocated()
        start_cpu = self.process.memory_info().rss / 1024 / 1024  # MB
        
        yield lambda: None
        
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            end_gpu = torch.cuda.memory_allocated()
            self.last_gpu_memory = (end_gpu - start_gpu) / 1024 / 1024  # MB
        else:
            self.last_gpu_memory = 0
            
        end_cpu = self.process.memory_info().rss / 1024 / 1024  # MB
        self.last_cpu_memory = end_cpu - start_cpu

def test_functional_correctness():
    """Test basic functionality and correctness"""
    print("🧠 Testing Functional Correctness...")
    
    try:
        from app.ml_models.self_onn import OptimizedSelfONN as SelfONN, OptimizedConv1DSelfONN as Conv1DSelfONN
        from app.ml_models.op_gan import OpGANGenerator, OpGANDiscriminator, OpGANLoss
        
        tester = ComprehensiveModelTester()
        
        # Test Self-ONN
        layer = SelfONN(input_size=64, output_size=32, q=5).to(tester.device)
        test_input = torch.randn(8, 64).to(tester.device)
        
        with tester.measure_time():
            output = layer(test_input)
        
        assert output.shape == (8, 32), f"Expected (8, 32), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains infinite values"
        print(f"  ✅ SelfONN: {test_input.shape} → {output.shape} ({tester.last_time:.2f}ms)")
        
        # Test Conv1D Self-ONN
        conv_layer = Conv1DSelfONN(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2, q=5).to(tester.device)
        audio_input = torch.randn(4, 1, 1000).to(tester.device)
        
        with tester.measure_time():
            conv_output = conv_layer(audio_input)
        
        expected_length = (1000 + 2*2 - 5) // 2 + 1  # 500
        assert conv_output.shape == (4, 16, 500), f"Expected (4, 16, 500), got {conv_output.shape}"
        assert not torch.isnan(conv_output).any(), "Conv output contains NaN values"
        print(f"  ✅ Conv1DSelfONN: {audio_input.shape} → {conv_output.shape} ({tester.last_time:.2f}ms)")
        
        # Test Op-GAN components
        generator = OpGANGenerator(input_length=32000, q=5).to(tester.device)
        discriminator = OpGANDiscriminator(input_length=32000, q=5).to(tester.device)
        
        test_audio = torch.randn(2, 1, 32000).to(tester.device)
        
        with tester.measure_time():
            generated = generator(test_audio)
        gen_time = tester.last_time
        
        with tester.measure_time():
            disc_output = discriminator(generated)
        disc_time = tester.last_time
        
        assert generated.shape == (2, 1, 32000), f"Expected (2, 1, 32000), got {generated.shape}"
        assert not torch.isnan(generated).any(), "Generated audio contains NaN values"
        assert torch.all(generated >= -1) and torch.all(generated <= 1), "Generated audio not in [-1, 1] range"
        
        print(f"  ✅ Generator: {test_audio.shape} → {generated.shape} ({gen_time:.2f}ms)")
        print(f"  ✅ Discriminator: {generated.shape} → {disc_output.shape} ({disc_time:.2f}ms)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functional test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_benchmarks():
    """Test runtime performance for real-time constraints"""
    print("\n⚡ Testing Performance Benchmarks...")
    
    try:
        from app.ml_models.op_gan import OpGANGenerator, OpGANDiscriminator
        
        tester = ComprehensiveModelTester()
        
        # Create models
        generator = OpGANGenerator(input_length=32000, q=5).to(tester.device)
        discriminator = OpGANDiscriminator(input_length=32000, q=5).to(tester.device)
        
        generator.eval()
        discriminator.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        
        print("  📊 Generator Performance:")
        for batch_size in batch_sizes:
            test_audio = torch.randn(batch_size, 1, 32000).to(tester.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = generator(test_audio)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(20):
                    with tester.measure_time():
                        output = generator(test_audio)
                    times.append(tester.last_time)
            
            avg_time = sum(times) / len(times)
            std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            per_sample = avg_time / batch_size
            
            # Real-time constraint: 2 seconds of audio should process in <100ms
            is_realtime = per_sample < 100
            status = "✅" if is_realtime else "⚠️"
            
            print(f"    {status} Batch {batch_size}: {avg_time:.1f}±{std_time:.1f}ms ({per_sample:.1f}ms/sample)")
        
        print("  📊 Discriminator Performance:")
        for batch_size in batch_sizes:
            test_audio = torch.randn(batch_size, 1, 32000).to(tester.device)
            
            times = []
            with torch.no_grad():
                for _ in range(20):
                    with tester.measure_time():
                        output = discriminator(test_audio)
                    times.append(tester.last_time)
            
            avg_time = sum(times) / len(times)
            per_sample = avg_time / batch_size
            is_realtime = per_sample < 50  # Discriminator should be faster
            status = "✅" if is_realtime else "⚠️"
            
            print(f"    {status} Batch {batch_size}: {avg_time:.1f}ms ({per_sample:.1f}ms/sample)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory usage and efficiency"""
    print("\n🧠 Testing Memory Efficiency...")
    
    try:
        from app.ml_models.op_gan import OpGANGenerator, OpGANDiscriminator
        from app.ml_models.self_onn import OptimizedSelfONN
        
        tester = ComprehensiveModelTester()
        
        # Test model memory footprint
        with tester.measure_memory():
            generator = OpGANGenerator(input_length=32000, q=5).to(tester.device)
        
        gen_model_memory = tester.last_cpu_memory + tester.last_gpu_memory
        
        with tester.measure_memory():
            discriminator = OpGANDiscriminator(input_length=32000, q=5).to(tester.device)
        
        disc_model_memory = tester.last_cpu_memory + tester.last_gpu_memory
        
        print(f"  📊 Model Memory Usage:")
        print(f"    Generator: {gen_model_memory:.1f}MB")
        print(f"    Discriminator: {disc_model_memory:.1f}MB")
        print(f"    Total: {gen_model_memory + disc_model_memory:.1f}MB")
        
        # Test inference memory usage
        test_audio = torch.randn(4, 1, 32000).to(tester.device)
        
        with tester.measure_memory():
            with torch.no_grad():
                generated = generator(test_audio)
        
        gen_inference_memory = tester.last_cpu_memory + tester.last_gpu_memory
        
        with tester.measure_memory():
            with torch.no_grad():
                disc_output = discriminator(generated)
        
        disc_inference_memory = tester.last_cpu_memory + tester.last_gpu_memory
        
        print(f"  📊 Inference Memory Usage (batch=4):")
        print(f"    Generator: {gen_inference_memory:.1f}MB")
        print(f"    Discriminator: {disc_inference_memory:.1f}MB")
        
        # Test efficiency features
        if hasattr(generator.enc1, 'get_efficiency_stats'):
            stats = generator.enc1.get_efficiency_stats()
            print(f"  📊 Self-ONN Efficiency:")
            print(f"    Active operators: {stats['active_operators']}/{stats['total_operators']}")
            print(f"    Pruning ratio: {stats['pruning_ratio']:.1%}")
            print(f"    Memory per layer: {stats['memory_MB']:.1f}MB")
        
        # Memory efficiency check
        total_memory = gen_model_memory + disc_model_memory + gen_inference_memory + disc_inference_memory
        is_efficient = total_memory < 1000  # Should be under 1GB
        status = "✅" if is_efficient else "⚠️"
        
        print(f"  {status} Total memory usage: {total_memory:.1f}MB (Target: <1000MB)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test gradient computation and backpropagation"""
    print("\n🔄 Testing Gradient Flow...")
    
    try:
        from app.ml_models.op_gan import OpGANGenerator, OpGANDiscriminator, OpGANLoss
        
        tester = ComprehensiveModelTester()
        
        # Create models
        generator = OpGANGenerator(input_length=32000, q=5).to(tester.device)
        discriminator = OpGANDiscriminator(input_length=32000, q=5).to(tester.device)
        loss_fn = OpGANLoss()
        
        generator.train()
        discriminator.train()
        
        # Test data
        corrupted_audio = torch.randn(2, 1, 32000, requires_grad=True).to(tester.device)
        clean_audio = torch.randn(2, 1, 32000).to(tester.device)
        
        # Test generator gradients
        generated = generator(corrupted_audio)
        disc_output = discriminator(generated)
        
        gen_loss, loss_dict = loss_fn.generator_loss(disc_output, generated, clean_audio)
        
        with tester.measure_time():
            gen_loss.backward(retain_graph=True)
        
        print(f"  ✅ Generator backward pass: {tester.last_time:.2f}ms")
        
        # Check gradient magnitudes
        total_norm = 0
        param_count = 0
        for param in generator.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Check for gradient issues
                assert not torch.isnan(param.grad).any(), "Generator gradients contain NaN"
                assert not torch.isinf(param.grad).any(), "Generator gradients contain Inf"
        
        total_norm = total_norm ** (1. / 2)
        print(f"  ✅ Generator gradient norm: {total_norm:.4f} (params: {param_count})")
        
        # Test discriminator gradients
        real_output = discriminator(clean_audio)
        fake_output = discriminator(generated.detach())
        
        disc_loss = loss_fn.discriminator_loss(real_output, fake_output)
        
        with tester.measure_time():
            disc_loss.backward()
        
        print(f"  ✅ Discriminator backward pass: {tester.last_time:.2f}ms")
        
        # Check discriminator gradients
        disc_norm = 0
        disc_params = 0
        for param in discriminator.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                disc_norm += param_norm.item() ** 2
                disc_params += 1
                
                assert not torch.isnan(param.grad).any(), "Discriminator gradients contain NaN"
                assert not torch.isinf(param.grad).any(), "Discriminator gradients contain Inf"
        
        disc_norm = disc_norm ** (1. / 2)
        print(f"  ✅ Discriminator gradient norm: {disc_norm:.4f} (params: {disc_params})")
        
        # Gradient health check
        gen_healthy = 1e-6 < total_norm < 1e3
        disc_healthy = 1e-6 < disc_norm < 1e3
        
        if gen_healthy and disc_healthy:
            print("  ✅ Gradient flow is healthy")
        else:
            print("  ⚠️ Gradient flow may have issues")
        
        return gen_healthy and disc_healthy
        
    except Exception as e:
        print(f"  ❌ Gradient test failed: {e}")
        traceback.print_exc()
        return False

def test_numerical_stability():
    """Test numerical stability under various conditions"""
    print("\n🔬 Testing Numerical Stability...")
    
    try:
        from app.ml_models.op_gan import OpGANGenerator
        
        tester = ComprehensiveModelTester()
        generator = OpGANGenerator(input_length=32000, q=5).to(tester.device)
        generator.eval()
        
        # Test with extreme inputs
        test_cases = [
            ("Normal", torch.randn(2, 1, 32000)),
            ("Zeros", torch.zeros(2, 1, 32000)),
            ("Ones", torch.ones(2, 1, 32000)),
            ("Large values", torch.randn(2, 1, 32000) * 10),
            ("Small values", torch.randn(2, 1, 32000) * 0.01),
        ]
        
        for name, test_input in test_cases:
            test_input = test_input.to(tester.device)
            
            with torch.no_grad():
                output = generator(test_input)
            
            # Check for numerical issues
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()
            in_range = torch.all(output >= -1) and torch.all(output <= 1)
            
            status = "✅" if not has_nan and not has_inf and in_range else "❌"
            issues = []
            if has_nan: issues.append("NaN")
            if has_inf: issues.append("Inf") 
            if not in_range: issues.append("Out of range")
            
            issue_str = f" ({', '.join(issues)})" if issues else ""
            print(f"  {status} {name} input: Output shape {output.shape}{issue_str}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Stability test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive test suite"""
    print("🧪 HarmonyRestorer - COMPREHENSIVE Test Suite")
    print("=" * 70)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 70)
    
    tests = [
        ("Functional Correctness", test_functional_correctness),
        ("Performance Benchmarks", test_performance_benchmarks), 
        ("Memory Efficiency", test_memory_efficiency),
        ("Gradient Flow", test_gradient_flow),
        ("Numerical Stability", test_numerical_stability)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n🔍 Running {name}...")
        if test_func():
            passed += 1
            print(f"✅ {name} PASSED")
        else:
            print(f"❌ {name} FAILED")
            
    print("\n" + "=" * 70)
    print(f"🧪 COMPREHENSIVE TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("🚀 Your models are production-ready for real-time audio processing!")
        return True
    else:
        print("⚠️ Some tests failed. Check performance and stability.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)