"""
HarmonyRestorer Audio Processing Services

This package provides audio restoration services combining:
- Ultimate Vocal Remover (UVR) for professional stem separation
- Breakthrough Op-GANs with solved gradient explosion for restoration

Architecture:
    UVR Separation → Op-GANs Restoration → Enhanced Audio Output
"""

from .separation import UVRSeparationService
from .restoration import AudioRestorationPipeline

# Public API exports
__all__ = [
    'UVRSeparationService',
    'AudioRestorationPipeline',
]

# Package metadata
__version__ = "1.0.0"
__author__ = "HarmonyRestorer"
__description__ = "Audio restoration services with breakthrough gradient stability"

# Service registry for dependency injection
SERVICES = {
    'separation': UVRSeparationService,
    'restoration': AudioRestorationPipeline,
}

def get_service(service_name: str):
    """
    Service factory for clean dependency injection
    
    Args:
        service_name: Name of service ('separation' or 'restoration')
        
    Returns:
        Service class ready for instantiation
        
    Example:
        >>> pipeline_class = get_service('restoration')
        >>> pipeline = pipeline_class()
    """
    if service_name not in SERVICES:
        raise ValueError(f"Unknown service: {service_name}. Available: {list(SERVICES.keys())}")
    
    return SERVICES[service_name]

# Convenience function for quick pipeline creation
def create_full_pipeline():
    """
    Create complete audio restoration pipeline
    
    Returns:
        AudioRestorationPipeline: Ready-to-use pipeline instance
    """
    return AudioRestorationPipeline()

# Package health check
def health_check():
    """
    Verify all services are available and functional
    
    Returns:
        dict: Status of each service component
    """
    try:
        pipeline = AudioRestorationPipeline()
        return pipeline.get_status()
    except Exception as e:
        return {
            "error": str(e),
            "pipeline_ready": False
        }
