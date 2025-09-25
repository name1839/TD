#!/usr/bin/env python3
"""
Test script to verify triplet dataset setup and configurations.
"""

import os
import sys
import traceback

# Add detectron2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_dataset_registration():
    """Test dataset registration."""
    print("Testing dataset registration...")
    
    try:
        from datasets.register_triplet_dataset import register_triplet_datasets
        registered = register_triplet_datasets()
        print(f"✓ Successfully registered {len(registered)} datasets")
        return True
    except Exception as e:
        print(f"✗ Dataset registration failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration file loading."""
    print("\nTesting configuration files...")
    
    from detectron2.config import get_cfg
    
    configs_to_test = [
        'configs/Triplet-Detection/faster_rcnn_R_50_FPN_triplet.yaml',
        'configs/Triplet-Detection/retinanet_R_50_FPN_triplet.yaml',
        'configs/Triplet-Detection/cascade_rcnn_R_50_FPN_triplet.yaml',
        'configs/Triplet-Detection/vitdet_base_triplet.yaml',
    ]
    
    success_count = 0
    
    for config_path in configs_to_test:
        try:
            cfg = get_cfg()
            cfg.merge_from_file(config_path)
            print(f"✓ {os.path.basename(config_path)}: OK")
            success_count += 1
        except Exception as e:
            print(f"✗ {os.path.basename(config_path)}: {e}")
    
    print(f"Config loading: {success_count}/{len(configs_to_test)} successful")
    return success_count == len(configs_to_test)

def test_data_loading():
    """Test actual data loading from registered datasets."""
    print("\nTesting data loading...")
    
    try:
        from detectron2.data import DatasetCatalog, MetadataCatalog
        
        # Test loading a small sample
        dataset_name = "triplet_split1_train"
        
        if dataset_name not in DatasetCatalog:
            print(f"✗ Dataset {dataset_name} not found in catalog")
            return False
        
        # Load first few samples
        dataset_dicts = DatasetCatalog.get(dataset_name)
        
        if len(dataset_dicts) == 0:
            print(f"✗ Dataset {dataset_name} is empty")
            return False
        
        # Check metadata
        metadata = MetadataCatalog.get(dataset_name)
        
        print(f"✓ Dataset {dataset_name}:")
        print(f"  - {len(dataset_dicts)} samples")
        print(f"  - {len(metadata.thing_classes)} classes")
        print(f"  - Sample image: {dataset_dicts[0]['file_name']}")
        print(f"  - Sample annotations: {len(dataset_dicts[0]['annotations'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training setup without actually training."""
    print("\nTesting training setup...")
    
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultTrainer
        from detectron2.data import build_detection_train_loader
        
        # Load config
        cfg = get_cfg()
        cfg.merge_from_file('configs/Triplet-Detection/faster_rcnn_R_50_FPN_triplet.yaml')
        
        # Set minimal config for testing
        cfg.DATASETS.TRAIN = ("triplet_split1_train",)
        cfg.DATASETS.TEST = ("triplet_split1_val",)
        cfg.SOLVER.MAX_ITER = 1  # Just one iteration for testing
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.OUTPUT_DIR = "./output/test_setup"
        
        # Test data loader creation
        train_loader = build_detection_train_loader(cfg)
        print("✓ Training data loader created successfully")
        
        # Test getting one batch
        data_iter = iter(train_loader)
        batch = next(data_iter)
        print(f"✓ Successfully loaded batch with {len(batch)} samples")
        
        return True
        
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TRIPLET DATASET SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Dataset Registration", test_dataset_registration),
        ("Config Loading", test_config_loading),
        ("Data Loading", test_data_loading),
        ("Training Setup", test_training_setup),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        success = test_func()
        results.append((test_name, success))
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:25} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your triplet detection setup is ready!")
        print("\nNext steps:")
        print("1. Run training: python tools/run_triplet_experiments.py --model faster_rcnn --split split1")
        print("2. Or use train_net.py directly: python tools/train_net.py --config-file configs/Triplet-Detection/faster_rcnn_R_50_FPN_triplet.yaml")
    else:
        print(f"\n❌ {len(results) - passed} tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
