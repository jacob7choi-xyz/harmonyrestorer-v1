"""Dataset pipeline for training the OpGAN audio denoiser.

Modules:
    noise           -- Analog noise synthesis (tape hiss, vinyl crackle, hum, etc.)
    acquire         -- Download CC/public domain classical recordings
    preprocess      -- Resample, normalize, slice into fixed-length frames
    generate_pairs  -- Combine clean frames with synthetic noise
    torch_dataset   -- PyTorch Dataset for training
"""
