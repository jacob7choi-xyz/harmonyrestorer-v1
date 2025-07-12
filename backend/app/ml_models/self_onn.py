import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SelfONN(nn.Module):
    """
    Self-Organized Operational Neural Network Layer
    Based on "Blind Restoration of Real-World Audio by 1D Operational GANs"
    
    Key Innovation: Instead of fixed multiplication, neurons learn to use
    different mathematical operators (sin, cos, exp, etc.) based on input patterns
    """
    
    def __init__(self, input_size, output_size, q=5, bias=True):
        super(SelfONN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.q = q  # Number of nodal operators
        
        # Weights for each operator at each connection
        self.weights = nn.Parameter(torch.randn(output_size, input_size, q))
        
        # Operator selection probabilities (learnable)
        self.operator_probs = nn.Parameter(torch.ones(output_size, q) / q)
        
        # Bias terms
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter('bias', None)
            
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        
        # Initialize operator probabilities to favor multiplication initially
        with torch.no_grad():
            self.operator_probs.data[:, 0] = 0.5  # Multiplication gets higher initial weight
            self.operator_probs.data[:, 1:] = 0.5 / (self.q - 1)
    
    def nodal_operators(self, x, w):
        """
        Applies different mathematical operators to input-weight combinations
        
        Args:
            x: Input tensor [batch_size, input_size]
            w: Weight tensor [output_size, input_size, q]
            
        Returns:
            Tensor of shape [batch_size, output_size, q] with operator results
        """
        batch_size = x.size(0)
        
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(1).unsqueeze(3)  # [batch, 1, input_size, 1]
        w_expanded = w.unsqueeze(0)  # [1, output_size, input_size, q]
        
        # Compute w * x for all connections
        wx = w_expanded * x_expanded  # [batch, output_size, input_size, q]
        
        # Apply different operators for each q
        operators_output = torch.zeros(batch_size, self.output_size, self.input_size, self.q, device=x.device)
        
        # Operator 0: Standard multiplication (baseline)
        operators_output[:, :, :, 0] = wx[:, :, :, 0]
        
        # Operator 1: Sine transformation
        operators_output[:, :, :, 1] = torch.sin(wx[:, :, :, 1])
        
        # Operator 2: Cosine transformation  
        operators_output[:, :, :, 2] = torch.cos(wx[:, :, :, 2])
        
        # Operator 3: Hyperbolic tangent
        operators_output[:, :, :, 3] = torch.tanh(wx[:, :, :, 3])
        
        if self.q > 4:
            # Operator 4: Exponential (with clipping for stability)
            operators_output[:, :, :, 4] = torch.exp(torch.clamp(wx[:, :, :, 4], -10, 10))
        
        # Sum across input dimensions to get final operator outputs
        return torch.sum(operators_output, dim=2)  # [batch, output_size, q]
    
    def forward(self, x):
        """
        Forward pass through Self-ONN layer
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Output tensor [batch_size, output_size]
        """
        # Apply nodal operators
        operator_outputs = self.nodal_operators(x, self.weights)  # [batch, output_size, q]
        
        # Apply softmax to operator probabilities for proper weighting
        operator_weights = F.softmax(self.operator_probs, dim=1)  # [output_size, q]
        
        # Weighted combination of operator outputs
        output = torch.sum(operator_outputs * operator_weights.unsqueeze(0), dim=2)  # [batch, output_size]
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
            
        return output
    
    def get_operator_usage(self):
        """
        Returns the current operator usage probabilities
        Useful for understanding what math the network has learned
        """
        return F.softmax(self.operator_probs, dim=1)


class Conv1DSelfONN(nn.Module):
    """
    1D Convolutional Self-ONN layer for audio processing
    Extends Self-ONN concept to work with time-series audio data
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, q=5):
        super(Conv1DSelfONN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.q = q
        
        # Convolutional weights for each operator
        self.conv_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, q)
        )
        
        # Operator probabilities
        self.operator_probs = nn.Parameter(torch.ones(out_channels, q) / q)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize convolutional weights"""
        n = self.in_channels * self.kernel_size
        stdv = 1. / math.sqrt(n)
        self.conv_weights.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        """
        Forward pass for 1D Convolutional Self-ONN
        
        Args:
            x: Input tensor [batch_size, in_channels, length]
            
        Returns:
            Output tensor [batch_size, out_channels, output_length]
        """
        batch_size, in_channels, length = x.size()
        
        # Calculate output length
        output_length = (length + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.out_channels, output_length, self.q, device=x.device)
        
        # Apply padding if necessary
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding))
        
        # Perform convolution for each operator
        for q_idx in range(self.q):
            # Standard convolution
            conv_out = F.conv1d(x, self.conv_weights[:, :, :, q_idx], stride=self.stride)
            
            # Apply operator transformation
            if q_idx == 0:  # Multiplication (identity)
                outputs[:, :, :, q_idx] = conv_out
            elif q_idx == 1:  # Sine
                outputs[:, :, :, q_idx] = torch.sin(conv_out)
            elif q_idx == 2:  # Cosine
                outputs[:, :, :, q_idx] = torch.cos(conv_out)
            elif q_idx == 3:  # Tanh
                outputs[:, :, :, q_idx] = torch.tanh(conv_out)
            elif q_idx == 4:  # Exponential (clipped)
                outputs[:, :, :, q_idx] = torch.exp(torch.clamp(conv_out, -10, 10))
        
        # Apply operator probabilities
        operator_weights = F.softmax(self.operator_probs, dim=1)  # [out_channels, q]
        operator_weights = operator_weights.unsqueeze(0).unsqueeze(3)  # [1, out_channels, 1, q]
        
        # Weighted combination of operator outputs
        final_output = torch.sum(outputs * operator_weights, dim=3)  # [batch, out_channels, output_length]
        
        return final_output


# Test the implementation
if __name__ == "__main__":
    # Test basic Self-ONN layer
    print("Testing Self-ONN Layer...")
    layer = SelfONN(input_size=64, output_size=32, q=5)
    test_input = torch.randn(16, 64)  # Batch of 16, 64 features
    output = layer(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Operator usage: {layer.get_operator_usage()}")
    
    # Test 1D Convolutional Self-ONN
    print("\nTesting Conv1D Self-ONN Layer...")
    conv_layer = Conv1DSelfONN(in_channels=1, out_channels=16, kernel_size=5, q=5)
    audio_input = torch.randn(8, 1, 1024)  # Batch of 8, mono audio, 1024 samples
    conv_output = conv_layer(audio_input)
    print(f"Audio input shape: {audio_input.shape}")
    print(f"Conv output shape: {conv_output.shape}")
    
    print("\n✅ Self-ONN implementation ready!")
    print("🎵 Next: Build the 1D Op-GAN Generator using these layers!")