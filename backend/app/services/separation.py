"""
UVR Integration Service for HarmonyRestorer
"""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class UVRSeparationService:
    """Integration service for Ultimate Vocal Remover"""
    
    def __init__(self, uvr_path: str = "../external_tools/uvr"):
        self.uvr_path = Path(uvr_path)
        self.separate_script = self.uvr_path / "separate.py"
        
        if not self.separate_script.exists():
            raise FileNotFoundError(f"UVR not found at {self.uvr_path}")
        
        print(f"✅ UVR Integration Service initialized")
        print(f"📁 UVR path: {self.uvr_path}")
        print(f"🔧 Separation script: {self.separate_script}")
    
    def test_connection(self):
        """Test that UVR is accessible"""
        return {
            "uvr_available": self.separate_script.exists(),
            "uvr_path": str(self.uvr_path),
            "separate_script": str(self.separate_script)
        }


# Test the service
if __name__ == "__main__":
    print("🧪 Testing UVR Integration Service...")
    service = UVRSeparationService()
    result = service.test_connection()
    print("📊 Test Results:", result)
    print("🎉 UVR Integration Service ready for development!")
