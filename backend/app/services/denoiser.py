"""
Audio Denoising Service using UVR
"""
import logging
from pathlib import Path
from typing import Optional
from audio_separator.separator import Separator

logger = logging.getLogger(__name__)


class DenoiserService:
    """Simple audio denoising using UVR-DeNoise-Lite"""
    
    def __init__(self, output_dir: Path, model_name: str = "UVR-DeNoise-Lite.pth"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._separator: Optional[Separator] = None
    
    def _get_separator(self) -> Separator:
        """Lazy load the separator"""
        if self._separator is None:
            logger.info("Loading UVR denoising model...")
            self._separator = Separator(
                output_dir=str(self.output_dir),
                output_format="WAV"
            )
            self._separator.load_model(self.model_name)
            logger.info("UVR model loaded")
        return self._separator
    
    def denoise(self, input_path: Path) -> Path:
        """
        Denoise an audio file.
        
        Args:
            input_path: Path to noisy audio file
            
        Returns:
            Path to denoised audio file
        """
        separator = self._get_separator()
        
        logger.info(f"Denoising: {input_path}")
        outputs = separator.separate(str(input_path))
        
        # Find the "No Noise" output
        clean_file = None
        for output in outputs:
            if "No Noise" in output:
                # Output is relative to output_dir
                clean_file = self.output_dir / Path(output).name
                break
        
        if clean_file is None:
            raise RuntimeError(f"Could not find denoised output in {outputs}")
        
        logger.info(f"Denoised output: {clean_file}")
        return clean_file
