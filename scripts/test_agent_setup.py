#!/usr/bin/env python3
"""
Test Agent Setup - Same Code Path as Training

This script tests the exact same dataset generation code that's used
in the MVQVAEAgent.setup_from_file() method during training.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from motion_vqvae.config_loader import ConfigLoader
from motion_vqvae.agent import MVQVAEAgent


def test_agent_setup(pkl_file: str, config_file: str, motion_id: int = None):
    """
    Test the exact same setup code used in training.
    
    Args:
        pkl_file: Path to PKL file
        config_file: Path to YAML config file
        motion_id: Specific motion ID to test (None for all motions)
    """
    
    print("🧪 Testing Agent Setup (Same as Training)")
    print("=" * 50)
    print(f"PKL file: {pkl_file}")
    print(f"Config file: {config_file}")
    print(f"Motion ID: {motion_id}")
    
    try:
        # Step 1: Load config (same as training)
        print("\n📋 Step 1: Loading Configuration")
        config_loader = ConfigLoader(config_file)
        config = config_loader.to_dict()
        config['device'] = 'cpu'  # Use CPU for testing
        print(f"  ✅ Config loaded successfully")
        
        # Step 2: Create agent (same as training)
        print("\n🤖 Step 2: Creating MVQVAEAgent")
        agent = MVQVAEAgent(config=config, device=torch.device('cpu'))
        print(f"  ✅ Agent created successfully")
        
        # Step 3: Setup from file (same as training)
        print("\n🔧 Step 3: Testing setup_from_file()")
        agent.setup_from_file(pkl_file, motion_id)
        print(f"  ✅ setup_from_file() completed successfully")
        
        # Step 4: Verify setup results
        print("\n📊 Step 4: Verifying Setup Results")
        print(f"  📊 Frame size: {agent.frame_size}")
        print(f"  📊 Mocap data shape: {agent.mocap_data.shape}")
        print(f"  📊 Number of motion sequences: {len(agent.end_indices)}")
        print(f"  📊 Dataset size: {len(agent.train_loader.dataset)}")
        print(f"  📊 Batch size: {agent.train_loader.batch_size}")
        print(f"  📊 Number of batches: {len(agent.train_loader)}")
        
        # Step 5: Test data loading
        print("\n🎲 Step 5: Testing Data Loading")
        batch = next(iter(agent.train_loader))
        print(f"  📊 Batch shape: {batch.shape}")
        print(f"  📊 Expected: [batch_size, window_size, frame_size]")
        print(f"  ✅ Data loading successful")
        
        print(f"\n✅ Agent setup test completed successfully!")
        print(f"🎉 The training dataset generation code is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Agent setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_motion_ids(pkl_file: str, config_file: str):
    """Test with multiple motion IDs to ensure the pipeline works for different IDs."""
    
    print("🎯 Testing Multiple Motion IDs")
    print("=" * 50)
    
    # Test different motion IDs
    test_ids = [0, 1, 2, 50, 100, 299, 300]
    
    for motion_id in test_ids:
        print(f"\n🧪 Testing Motion ID: {motion_id}")
        print("-" * 30)
        
        success = test_agent_setup(pkl_file, config_file, motion_id)
        
        if success:
            print(f"  ✅ Motion ID {motion_id}: PASSED")
        else:
            print(f"  ❌ Motion ID {motion_id}: FAILED")
            return False
    
    print(f"\n🎉 All motion IDs tested successfully!")
    return True


def main():
    """Main function to test agent setup."""
    
    # Your specific parameters
    pkl_file = "/home/dhbaek/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl"
    config_file = "configs/agent.yaml"
    
    print("🧪 Testing Agent Setup (Training Dataset Generation)")
    print("=" * 60)
    print(f"PKL file: {pkl_file}")
    print(f"Config file: {config_file}")
    
    # Check if files exist
    if not Path(pkl_file).exists():
        print(f"\n❌ PKL file not found: {pkl_file}")
        print("Please update the pkl_file path in the script.")
        return
    
    if not Path(config_file).exists():
        print(f"\n❌ Config file not found: {config_file}")
        print("Please update the config_file path in the script.")
        return
    
    try:
        # Test with a single motion ID first
        print("🧪 Testing Single Motion ID")
        success = test_agent_setup(pkl_file, config_file, motion_id=0)
        
        if success:
            # Test with multiple motion IDs
            print("\n🧪 Testing Multiple Motion IDs")
            success = test_multiple_motion_ids(pkl_file, config_file)
        
        if success:
            print(f"\n📋 Test Summary:")
            print(f"  ✅ Config loading: Working")
            print(f"  ✅ Agent creation: Working")
            print(f"  ✅ setup_from_file(): Working")
            print(f"  ✅ Dataset generation: Working")
            print(f"  ✅ Data loading: Working")
            print(f"\n🚀 Training dataset generation is ready!")
            print(f"💡 You can now run training with confidence.")
        else:
            print(f"\n❌ Some tests failed. Please check the errors above.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


'''
python scripts/test_agent_setup.py

'''