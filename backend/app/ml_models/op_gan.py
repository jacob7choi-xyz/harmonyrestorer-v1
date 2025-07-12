import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .self_onn import OptimizedSelfONN as SelfONN, OptimizedConv1DSelfONN as Conv1DSelfONN

class OpGANGenerator(nn.Module):
    """
    1D Operational GAN Generator for Blind Audio Restoration
    Based on "Blind Restoration of Real-World Audio by 1D Operational GANs"
    
    Architecture: 10-layer U-Net with Self-ONNs
    - Encoder: 5 Self-ONN layers with downsampling
    - Decoder: 5 Self-ONN layers with upsampling + skip connections
    """
    
    def __init__(self, input_length=32000, q=5):
        super(OpGANGenerator, self).__init__()
        self.input_length = input_length
        self.q = q  # Number of operators in Self-ONNs
        
        # Encoder layers (downsampling)
        self.enc1 = Conv1DSelfONN(1, 16, kernel_size=5, stride=2, padding=2, q=q)
        self.enc2 = Conv1DSelfONN(16, 32, kernel_size=5, stride=2, padding=2, q=q)
        self.enc3 = Conv1DSelfONN(32, 64, kernel_size=5, stride=2, padding=2, q=q)
        self.enc4 = Conv1DSelfONN(64, 128, kernel_size=5, stride=2, padding=2, q=q)
        self.enc5 = Conv1DSelfONN(128, 128, kernel_size=5, stride=2, padding=2, q=q)
        
        # Decoder layers (upsampling with skip connections)
        self.dec1 = Conv1DSelfONN(128, 128, kernel_size=5, stride=1, padding=2, q=q)
        self.dec2 = Conv1DSelfONN(256, 64, kernel_size=5, stride=1, padding=2, q=q)  # 128 + 128 from skip
        self.dec3 = Conv1DSelfONN(128, 32, kernel_size=5, stride=1, padding=2, q=q)  # 64 + 64 from skip
        self.dec4 = Conv1DSelfONN(64, 16, kernel_size=5, stride=1, padding=2, q=q)   # 32 + 32 from skip
        self.dec5 = Conv1DSelfONN(32, 1, kernel_size=5, stride=1, padding=2, q=q)    # 16 + 16 from skip
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass through the Op-GAN Generator
        
        Args:
            x: Input corrupted audio [batch_size, 1, 32000]
            
        Returns:
            restored_audio: Enhanced audio [batch_size, 1, 32000]
        """
        # Input shape: [batch, 1, 32000]
        
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
        
        # Decoder with skip connections and upsampling
        # Layer 1: 1000 -> 2000
        dec1_out = F.interpolate(self.leaky_relu(self.dec1(enc5_out)), 
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
        
        # Layer 5: Skip connection + 16000 -> 32000 (output)
        skip1 = torch.cat([dec4_out, enc1_out], dim=1)  # [batch, 32, 16000]
        output = F.interpolate(self.dec5(skip1), 
                              scale_factor=2, mode='linear', align_corners=False)  # [batch, 1, 32000]
        
        # Apply tanh activation for final output
        restored_audio = self.tanh(output)
        
        return restored_audio
    
    def get_operator_usage_summary(self):
        """
        Returns operator usage across all Self-ONN layers for analysis
        """
        usage_summary = {}
        
        # Encoder layers
        for i, layer in enumerate([self.enc1, self.enc2, self.enc3, self.enc4, self.enc5], 1):
            usage_summary[f'enc{i}'] = F.softmax(layer.operator_probs, dim=1).mean(dim=0)
            
        # Decoder layers  
        for i, layer in enumerate([self.dec1, self.dec2, self.dec3, self.dec4, self.dec5], 1):
            usage_summary[f'dec{i}'] = F.softmax(layer.operator_probs, dim=1).mean(dim=0)
            
        return usage_summary


class OpGANDiscriminator(nn.Module):
    """
    1D Operational GAN Discriminator for Audio Restoration
    
    Architecture: 6 Self-ONN layers as specified in the paper
    """
    
    def __init__(self, input_length=32000, q=5):
        super(OpGANDiscriminator, self).__init__()
        self.input_length = input_length
        self.q = q
        
        # Discriminator layers - strides: [2, 2, 2, 2, 1, 2]
        self.conv1 = Conv1DSelfONN(1, 64, kernel_size=4, stride=2, padding=1, q=q)
        self.conv2 = Conv1DSelfONN(64, 128, kernel_size=4, stride=2, padding=1, q=q)
        self.conv3 = Conv1DSelfONN(128, 256, kernel_size=4, stride=2, padding=1, q=q)
        self.conv4 = Conv1DSelfONN(256, 512, kernel_size=4, stride=2, padding=1, q=q)
        self.conv5 = Conv1DSelfONN(512, 512, kernel_size=4, stride=1, padding=1, q=q)
        self.conv6 = Conv1DSelfONN(512, 1, kernel_size=4, stride=2, padding=1, q=q)
        
        # Activation
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        """
        Forward pass through discriminator
        
        Args:
            x: Input audio [batch_size, 1, 32000]
            
        Returns:
            validity: Probability that input is real [batch_size, 1, output_length]
        """
        # Progressive downsampling
        x = self.leaky_relu(self.conv1(x))  # [batch, 64, 16000]
        x = self.leaky_relu(self.conv2(x))  # [batch, 128, 8000]
        x = self.leaky_relu(self.conv3(x))  # [batch, 256, 4000]
        x = self.leaky_relu(self.conv4(x))  # [batch, 512, 2000]
        x = self.leaky_relu(self.conv5(x))  # [batch, 512, 2000]
        x = self.conv6(x)                   # [batch, 1, 1000]
        
        return x


class OpGANLoss(nn.Module):
    """
    Composite loss function for Op-GAN training
    Combines adversarial, temporal, and spectral losses
    """
    
    def __init__(self, lambda_temporal=10, lambda_spectral=5, n_fft=256, hop_length=128):
        super(OpGANLoss, self).__init__()
        self.lambda_temporal = lambda_temporal
        self.lambda_spectral = lambda_spectral
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def temporal_loss(self, generated, target):
        """L2 temporal domain loss"""
        return self.mse_loss(generated, target)
    
    def spectral_loss(self, generated, target):
        """STFT magnitude loss in frequency domain"""
        # Compute STFT for both signals
        gen_stft = torch.stft(generated.squeeze(1), n_fft=self.n_fft, 
                             hop_length=self.hop_length, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=self.n_fft, 
                                hop_length=self.hop_length, return_complex=True)
        
        # Use magnitude for loss computation
        gen_mag = torch.abs(gen_stft)
        target_mag = torch.abs(target_stft)
        
        return self.mse_loss(gen_mag, target_mag)
    
    def generator_loss(self, discriminator_output, generated, target):
        """
        Complete generator loss: adversarial + temporal + spectral
        """
        # Adversarial loss (LSGAN variant)
        adversarial_loss = self.mse_loss(discriminator_output, torch.ones_like(discriminator_output))
        
        # Temporal loss  
        temporal_loss = self.temporal_loss(generated, target)
        
        # Spectral loss
        spectral_loss = self.spectral_loss(generated, target)
        
        # Combined loss
        total_loss = (adversarial_loss + 
                     self.lambda_temporal * temporal_loss + 
                     self.lambda_spectral * spectral_loss)
        
        return total_loss, {
            'adversarial': adversarial_loss.item(),
            'temporal': temporal_loss.item(), 
            'spectral': spectral_loss.item(),
            'total': total_loss.item()
        }
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Discriminator loss (LSGAN variant)
        """
        real_loss = self.mse_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.mse_loss(fake_output, torch.zeros_like(fake_output))
        
        total_loss = 0.5 * (real_loss + fake_loss)
        return total_loss


# Test the complete Op-GAN implementation
if __name__ == "__main__":
    print("🎵 Testing Complete 1D Op-GAN Implementation...")
    
    # Test Generator
    generator = OpGANGenerator(input_length=32000, q=5)
    test_input = torch.randn(4, 1, 32000)  # Batch of 4, mono audio, 32k samples
    
    print(f"Generator input shape: {test_input.shape}")
    generated_output = generator(test_input)
    print(f"Generator output shape: {generated_output.shape}")
    
    # Test Discriminator
    discriminator = OpGANDiscriminator(input_length=32000, q=5)
    disc_output = discriminator(generated_output)
    print(f"Discriminator output shape: {disc_output.shape}")
    
    # Test Loss Function
    loss_fn = OpGANLoss(lambda_temporal=10, lambda_spectral=5)
    target_audio = torch.randn(4, 1, 32000)  # Clean target audio
    
    # Generator loss
    gen_loss, loss_dict = loss_fn.generator_loss(disc_output, generated_output, target_audio)
    print(f"Generator loss: {gen_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
    
    # Discriminator loss  
    real_output = discriminator(target_audio)
    disc_loss = loss_fn.discriminator_loss(real_output, disc_output.detach())
    print(f"Discriminator loss: {disc_loss.item():.4f}")
    
    # Model parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nModel Parameters:")
    print(f"Generator: {gen_params:,} parameters")
    print(f"Discriminator: {disc_params:,} parameters")
    print(f"Total: {gen_params + disc_params:,} parameters")
    
    # Operator usage analysis
    print(f"\n🔬 Operator Usage Analysis:")
    usage = generator.get_operator_usage_summary()
    for layer, probs in usage.items():
        print(f"{layer}: {probs.numpy()}")
    
    print("\n✅ 1D Op-GAN implementation complete!")
    print("🚀 Ready for training on your audio restoration dataset!")