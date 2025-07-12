import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .self_onn import OptimizedSelfONN as SelfONN, OptimizedConv1DSelfONN as Conv1DSelfONN

class OpGANGenerator(nn.Module):
    """
    Enhanced 1D Operational GAN Generator with gradient stability improvements
    Based on "Blind Restoration of Real-World Audio by 1D Operational GANs"
    
    Architecture: 10-layer U-Net with Self-ONNs
    - Encoder: 5 Self-ONN layers with downsampling
    - Decoder: 5 Self-ONN layers with upsampling + skip connections
    
    Improvements:
    - Gradient clipping integration
    - Numerical stability
    - Performance optimizations
    """
    
    def __init__(self, input_length=32000, q=3, gradient_clip_value=1.0):  # Reduced q from 5 to 3
        super(OpGANGenerator, self).__init__()
        self.input_length = input_length
        self.q = q
        self.gradient_clip_value = gradient_clip_value
        
        # Encoder layers (downsampling) - reduced q for stability
        self.enc1 = Conv1DSelfONN(1, 16, kernel_size=5, stride=2, padding=2, q=q)
        self.enc2 = Conv1DSelfONN(16, 32, kernel_size=5, stride=2, padding=2, q=q)
        self.enc3 = Conv1DSelfONN(32, 64, kernel_size=5, stride=2, padding=2, q=q)
        self.enc4 = Conv1DSelfONN(64, 128, kernel_size=5, stride=2, padding=2, q=q)
        self.enc5 = Conv1DSelfONN(128, 128, kernel_size=5, stride=2, padding=2, q=q)
        
        # Decoder layers with enhanced architecture
        # Use regular conv for bottleneck to improve stability
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(16, 128),  # Group norm for better stability
            nn.LeakyReLU(0.1, inplace=True)  # Reduced negative slope
        )
        
        # Decoder layers (upsampling with skip connections)
        self.dec1 = Conv1DSelfONN(128, 128, kernel_size=5, stride=1, padding=2, q=q)
        self.dec2 = Conv1DSelfONN(256, 64, kernel_size=5, stride=1, padding=2, q=q)  # 128 + 128 from skip
        self.dec3 = Conv1DSelfONN(128, 32, kernel_size=5, stride=1, padding=2, q=q)  # 64 + 64 from skip
        self.dec4 = Conv1DSelfONN(64, 16, kernel_size=5, stride=1, padding=2, q=q)   # 32 + 32 from skip
        
        # Final layer as regular conv for stability
        self.final_conv = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )
        
        # Activation functions with reduced slopes
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)  # Reduced from 0.2
        
        # Lighter dropout for regularization
        self.dropout = nn.Dropout(0.2)  # Reduced from 0.5
        
        # Initialize weights for stability
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Xavier initialization with smaller gain
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Enhanced forward pass with input validation and gradient stability
        
        Args:
            x: Input corrupted audio [batch_size, 1, 32000]
            
        Returns:
            restored_audio: Enhanced audio [batch_size, 1, 32000]
        """
        # Input validation and clamping
        x = torch.clamp(x, -1.0, 1.0)
        
        # Encoder with skip connection storage
        # Layer 1: 32000 -> 16000
        enc1_out = self.leaky_relu(self.enc1(x))  # [batch, 16, 16000]
        
        # Layer 2: 16000 -> 8000  
        enc2_out = self.leaky_relu(self.enc2(enc1_out))  # [batch, 32, 8000]
        
        # Layer 3: 8000 -> 4000
        enc3_out = self.leaky_relu(self.enc3(enc2_out))  # [batch, 64, 4000]
        
        # Layer 4: 4000 -> 2000
        enc4_out = self.leaky_relu(self.enc4(enc3_out))  # [batch, 128, 2000]
        
        # Layer 5: 2000 -> 1000 (bottleneck)
        enc5_out = self.leaky_relu(self.enc5(enc4_out))  # [batch, 128, 1000]
        
        # Enhanced bottleneck processing
        bottleneck_out = self.bottleneck(enc5_out)  # [batch, 128, 1000]
        
        # Decoder with skip connections and upsampling
        # Layer 1: 1000 -> 2000
        dec1_out = F.interpolate(self.leaky_relu(self.dec1(bottleneck_out)), 
                                scale_factor=2, mode='linear', align_corners=False)  # [batch, 128, 2000]
        dec1_out = self.dropout(dec1_out)
        
        # Layer 2: Skip connection + 2000 -> 4000
        skip4 = torch.cat([dec1_out, enc4_out], dim=1)  # [batch, 256, 2000]
        dec2_out = F.interpolate(self.leaky_relu(self.dec2(skip4)), 
                                scale_factor=2, mode='linear', align_corners=False)  # [batch, 64, 4000]
        dec2_out = self.dropout(dec2_out)
        
        # Layer 3: Skip connection + 4000 -> 8000
        skip3 = torch.cat([dec2_out, enc3_out], dim=1)  # [batch, 128, 4000]
        dec3_out = F.interpolate(self.leaky_relu(self.dec3(skip3)), 
                                scale_factor=2, mode='linear', align_corners=False)  # [batch, 32, 8000]
        dec3_out = self.dropout(dec3_out)
        
        # Layer 4: Skip connection + 8000 -> 16000
        skip2 = torch.cat([dec3_out, enc2_out], dim=1)  # [batch, 64, 8000]
        dec4_out = F.interpolate(self.leaky_relu(self.dec4(skip2)), 
                                scale_factor=2, mode='linear', align_corners=False)  # [batch, 16, 16000]
        
        # Final layer: Skip connection + 16000 -> 32000 (output)
        skip1 = torch.cat([dec4_out, enc1_out], dim=1)  # [batch, 32, 16000]
        final_input = F.interpolate(skip1, scale_factor=2, mode='linear', align_corners=False)  # [batch, 32, 32000]
        
        # Apply final convolution with tanh
        restored_audio = self.final_conv(final_input)  # [batch, 1, 32000]
        
        # Ensure output is in valid range
        restored_audio = torch.clamp(restored_audio, -1.0, 1.0)
        
        return restored_audio
    
    def clip_gradients(self):
        """Clip gradients for all Self-ONN layers"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_value)
        
        # Also clip individual Self-ONN layers if they have the method
        for module in self.modules():
            if hasattr(module, 'clip_gradients'):
                module.clip_gradients()
    
    def get_operator_usage_summary(self):
        """
        Returns operator usage across all Self-ONN layers for analysis
        """
        usage_summary = {}
        
        # Encoder layers
        for i, layer in enumerate([self.enc1, self.enc2, self.enc3, self.enc4, self.enc5], 1):
            if hasattr(layer, 'operator_probs'):
                usage_summary[f'enc{i}'] = F.softmax(layer.operator_probs, dim=1).mean(dim=0)
            
        # Decoder layers  
        for i, layer in enumerate([self.dec1, self.dec2, self.dec3, self.dec4], 1):
            if hasattr(layer, 'operator_probs'):
                usage_summary[f'dec{i}'] = F.softmax(layer.operator_probs, dim=1).mean(dim=0)
            
        return usage_summary


class OpGANDiscriminator(nn.Module):
    """
    Enhanced 1D Operational GAN Discriminator with performance optimizations
    
    Architecture: Reduced complexity for better performance
    - Mix of Self-ONN and regular convolutions
    - Optimized for speed while maintaining effectiveness
    """
    
    def __init__(self, input_length=32000, q=2, gradient_clip_value=1.0):  # Reduced q from 5 to 2
        super(OpGANDiscriminator, self).__init__()
        self.input_length = input_length
        self.q = q
        self.gradient_clip_value = gradient_clip_value
        
        # First layer: Self-ONN for feature extraction
        self.conv1 = Conv1DSelfONN(1, 32, kernel_size=4, stride=2, padding=1, q=q)  # Reduced channels
        
        # Rest: Regular convolutions for speed
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(64, 256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.GroupNorm(64, 256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Final classification layer
        self.conv6 = nn.Conv1d(256, 1, kernel_size=4, stride=2, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Enhanced forward pass with input validation
        
        Args:
            x: Input audio [batch_size, 1, 32000]
            
        Returns:
            validity: Probability that input is real [batch_size, 1, output_length]
        """
        # Input validation
        x = torch.clamp(x, -1.0, 1.0)
        
        # Progressive downsampling with stability
        x = self.conv1(x)          # [batch, 32, 16000] - Self-ONN layer
        x = self.conv2(x)          # [batch, 64, 8000]
        x = self.conv3(x)          # [batch, 128, 4000]
        x = self.conv4(x)          # [batch, 256, 2000]
        x = self.conv5(x)          # [batch, 256, 2000]
        x = self.conv6(x)          # [batch, 1, 1000]
        
        return x
    
    def clip_gradients(self):
        """Clip gradients for stable training"""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_value)


class OpGANLoss(nn.Module):
    """
    Enhanced composite loss function with numerical stability
    Combines adversarial, temporal, and spectral losses with gradient management
    """
    
    def __init__(self, lambda_temporal=10, lambda_spectral=5, n_fft=128, hop_length=64):
        super(OpGANLoss, self).__init__()
        self.lambda_temporal = lambda_temporal
        self.lambda_spectral = lambda_spectral
        self.n_fft = n_fft  # Reduced for performance
        self.hop_length = hop_length  # Reduced for performance
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def temporal_loss(self, generated, target):
        """Enhanced L1 temporal domain loss (more stable than L2)"""
        return self.l1_loss(generated, target)
    
    def spectral_loss(self, generated, target):
        """Enhanced STFT magnitude loss with numerical stability"""
        try:
            # Create window for STFT
            window = torch.hann_window(self.n_fft, device=generated.device)
            
            # Compute STFT for both signals with stability improvements
            gen_stft = torch.stft(
                generated.squeeze(1), 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                window=window,
                return_complex=True,
                normalized=True  # Add normalization for stability
            )
            target_stft = torch.stft(
                target.squeeze(1), 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                window=window,
                return_complex=True,
                normalized=True
            )
            
            # Use magnitude with small epsilon for numerical stability
            gen_mag = torch.abs(gen_stft) + 1e-8
            target_mag = torch.abs(target_stft) + 1e-8
            
            return self.l1_loss(gen_mag, target_mag)
            
        except Exception as e:
            # Fallback to temporal loss if STFT fails
            print(f"STFT failed, using temporal loss: {e}")
            return self.temporal_loss(generated, target)
    
    def generator_loss(self, discriminator_output, generated, target):
        """
        Enhanced generator loss with label smoothing and clamping
        """
        # Adversarial loss with label smoothing for stability
        real_labels = torch.ones_like(discriminator_output) * 0.9  # Label smoothing
        adversarial_loss = self.mse_loss(discriminator_output, real_labels)
        
        # Temporal loss  
        temporal_loss = self.temporal_loss(generated, target)
        
        # Spectral loss
        spectral_loss = self.spectral_loss(generated, target)
        
        # Combined loss with clamping to prevent explosion
        total_loss = (adversarial_loss + 
                     self.lambda_temporal * temporal_loss + 
                     self.lambda_spectral * spectral_loss)
        
        # Clamp total loss to prevent gradient explosion
        total_loss = torch.clamp(total_loss, 0, 100)
        
        return total_loss, {
            'adversarial': adversarial_loss.item(),
            'temporal': temporal_loss.item(), 
            'spectral': spectral_loss.item(),
            'total': total_loss.item()
        }
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Enhanced discriminator loss with label smoothing
        """
        # Use label smoothing for more stable training
        real_labels = torch.ones_like(real_output) * 0.9
        fake_labels = torch.zeros_like(fake_output) + 0.1
        
        real_loss = self.mse_loss(real_output, real_labels)
        fake_loss = self.mse_loss(fake_output, fake_labels)
        
        total_loss = 0.5 * (real_loss + fake_loss)
        
        # Clamp to prevent explosion
        total_loss = torch.clamp(total_loss, 0, 10)
        
        return total_loss


class GradientHealthMonitor:
    """Monitor and manage gradient health during training"""
    
    def __init__(self, max_grad_norm=1.0, log_frequency=10):
        self.max_grad_norm = max_grad_norm
        self.log_frequency = log_frequency
        self.step_count = 0
        self.gradient_history = []
    
    def check_and_clip_gradients(self, model, model_name=""):
        """Check gradient health and apply clipping"""
        self.step_count += 1
        
        # Compute gradient norm
        total_norm = 0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # Log occasionally
        if self.step_count % self.log_frequency == 0:
            print(f"Step {self.step_count} {model_name}: Grad norm={total_norm:.2e}")
        
        # Clip if needed
        if total_norm > self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            if self.step_count % self.log_frequency == 0:
                print(f"  🔧 Gradients clipped from {total_norm:.2e} to {self.max_grad_norm}")
        
        # Store statistics
        self.gradient_history.append({
            'step': self.step_count,
            'total_norm': total_norm,
            'param_count': param_count
        })
        
        return total_norm


# Enhanced test function
if __name__ == "__main__":
    print("🎵 Testing Enhanced 1D Op-GAN Implementation...")
    
    # Test Generator with reduced complexity
    generator = OpGANGenerator(input_length=32000, q=3)  # Reduced q
    test_input = torch.randn(2, 1, 32000)  # Smaller batch for testing
    
    print(f"Generator input shape: {test_input.shape}")
    
    # Time the generator
    import time
    start_time = time.time()
    with torch.no_grad():
        generated_output = generator(test_input)
    gen_time = (time.time() - start_time) * 1000
    
    print(f"Generator output shape: {generated_output.shape}")
    print(f"Generator time: {gen_time:.2f}ms (Target: <50ms)")
    
    # Test Discriminator with reduced complexity
    discriminator = OpGANDiscriminator(input_length=32000, q=2)  # Reduced q
    
    start_time = time.time()
    with torch.no_grad():
        disc_output = discriminator(generated_output)
    disc_time = (time.time() - start_time) * 1000
    
    print(f"Discriminator output shape: {disc_output.shape}")
    print(f"Discriminator time: {disc_time:.2f}ms (Target: <50ms)")
    
    # Test Enhanced Loss Function
    loss_fn = OpGANLoss(lambda_temporal=10, lambda_spectral=5)
    target_audio = torch.randn(2, 1, 32000)
    
    # Test gradient computation
    generator.train()
    discriminator.train()
    
    # Forward pass
    generated = generator(test_input)
    disc_fake = discriminator(generated)
    disc_real = discriminator(target_audio)
    
    # Compute losses
    gen_loss, loss_dict = loss_fn.generator_loss(disc_fake, generated, target_audio)
    disc_loss = loss_fn.discriminator_loss(disc_real, disc_fake.detach())
    
    print(f"\nLoss Values:")
    print(f"Generator loss: {gen_loss.item():.4f}")
    print(f"Discriminator loss: {disc_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    # Test gradient computation
    monitor = GradientHealthMonitor()
    
    gen_loss.backward(retain_graph=True)
    gen_grad_norm = monitor.check_and_clip_gradients(generator, "Generator")
    
    disc_loss.backward()
    disc_grad_norm = monitor.check_and_clip_gradients(discriminator, "Discriminator")
    
    print(f"\nGradient Norms:")
    print(f"Generator: {gen_grad_norm:.2e} (Should be <10)")
    print(f"Discriminator: {disc_grad_norm:.2e} (Should be <10)")
    
    # Model parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nModel Parameters:")
    print(f"Generator: {gen_params:,} parameters")
    print(f"Discriminator: {disc_params:,} parameters")
    print(f"Total: {gen_params + disc_params:,} parameters")
    print(f"Research paper target: ~1.1M parameters")
    
    print(f"\n✅ Enhanced 1D Op-GAN implementation complete!")
    print(f"🎯 Performance: Gen={gen_time:.1f}ms, Disc={disc_time:.1f}ms")
    print(f"🎯 Gradients: Gen={gen_grad_norm:.2e}, Disc={disc_grad_norm:.2e}")
    print(f"🚀 Ready for stable training!")