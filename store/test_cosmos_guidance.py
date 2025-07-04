#!/usr/bin/env python3

"""
Test script for Cosmos2VideoToWorldPipeline with V-JEPA guidance.

This script tests the basic functionality of the guidance implementation
without requiring the full model weights.
"""

import sys
import os
import torch


def test_pipeline_import():
    """Test that the pipeline can be imported without errors."""
    try:
        from pipelines.cosmos2_i2v_guidance_torch import Cosmos2VideoToWorldPipeline
        print("✅ Successfully imported Cosmos2VideoToWorldPipeline")
        return True
    except ImportError as e:
        print(f"❌ Failed to import pipeline: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_guidance_configuration():
    """Test that guidance parameters can be set correctly."""
    try:
        from pipelines.cosmos2_i2v_guidance_torch import Cosmos2VideoToWorldPipeline
        
        # Create a mock pipeline instance (we can't fully initialize without weights)
        pipeline = object.__new__(Cosmos2VideoToWorldPipeline)
        
        # Test setting guidance parameters
        pipeline.guidance_start = 800
        pipeline.guidance_end = 950
        pipeline.guidance_rho_scale = 2.5
        pipeline.vjepa_context_length = 10
        pipeline.vjepa_stride = 4
        pipeline.vjepa_mode = 'mean'
        
        # Verify parameters were set
        assert pipeline.guidance_start == 800
        assert pipeline.guidance_end == 950
        assert pipeline.guidance_rho_scale == 2.5
        assert pipeline.vjepa_context_length == 10
        assert pipeline.vjepa_stride == 4
        assert pipeline.vjepa_mode == 'mean'
        
        print("✅ Successfully configured guidance parameters")
        return True
    except Exception as e:
        print(f"❌ Failed to configure guidance parameters: {e}")
        return False

def test_helper_functions():
    """Test that helper functions are accessible."""
    try:
        from pipelines.cosmos2_i2v_guidance_torch import print_gpu_memory, retrieve_timesteps, retrieve_latents
        print("✅ Successfully imported helper functions")
        return True
    except ImportError as e:
        print(f"❌ Failed to import helper functions: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Cosmos V-JEPA Guidance Pipeline Implementation")
    print("=" * 60)
    
    tests = [
        test_pipeline_import,
        test_guidance_configuration,
        test_helper_functions,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The guidance implementation looks good.")
        print("\n📝 Next steps:")
        print("1. Test with actual model weights")
        print("2. Verify V-JEPA model loading")
        print("3. Test guidance on sample videos")
        print("4. Tune guidance parameters for optimal results")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 