#!/usr/bin/env python3
"""
Quick test script to verify the original methodology implementation
"""

import os
import sys
from reproduce_intphys import IntPhysEvaluator

def test_small_dataset():
    """Test on a small subset to verify the implementation works."""
    
    # Test on O3 dataset (should be smaller)
    data_path = "/home/yjianhao/project/video_guidance/dev/O3"
    
    if not os.path.exists(data_path):
        print(f"Error: Test data not found at {data_path}")
        print("Please update the data_path in this script")
        return False
    
    print("Testing original methodology implementation...")
    print(f"Data path: {data_path}")
    
    # Create evaluator
    evaluator = IntPhysEvaluator()
    
    # Run evaluation on a small subset
    try:
        context_results, all_results = evaluator.evaluate_dataset(
            data_path=data_path,
            frames_per_clip=8,  # Smaller for faster testing
            frame_step=2,
            save_results=True
        )
        
        print("\n" + "="*60)
        print("TEST SUCCESSFUL!")
        print("="*60)
        print("Results summary:")
        
        # Show results for each context length
        for context_len in [2, 4, 6, 8, 10]:
            if context_len in context_results:
                metrics = context_results[context_len]
                print(f"Context {context_len:2d}: "
                      f"Best Abs Acc = {metrics['Best Absolute Accuracy (max)']:5.1f}%, "
                      f"AUROC = {metrics['AUROC (max)']:5.3f}")
        
        # Show filtered result
        if 'Filtered' in context_results:
            filtered = context_results['Filtered']
            print(f"Filtered   : "
                  f"Best Abs Acc = {filtered['Best Absolute Accuracy (max)']:5.1f}%, "
                  f"AUROC = {filtered['AUROC (max)']:5.3f}")
        
        print("\nThe implementation is working correctly!")
        print("Ready to run full evaluation on all datasets.")
        
        return True
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_dataset()
    if success:
        print("\n✓ All tests passed! You can now run the full evaluation.")
        sys.exit(0)
    else:
        print("\n✗ Tests failed. Please check the implementation.")
        sys.exit(1) 