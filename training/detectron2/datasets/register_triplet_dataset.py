#!/usr/bin/env python3
"""
Register triplet dataset (Tool-Action-Target) for detectron2.

This script registers all splits of the triplet dataset with detectron2's
DatasetCatalog and MetadataCatalog for easy use in training and evaluation.
"""

import os
import sys
import json

try:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
except ImportError as e:
    print(f"Error importing detectron2: {e}")
    print("Make sure detectron2 is properly installed and you're in the correct environment")
    sys.exit(1)

TRIPLET_DATASET_ROOT = "/ssd/prostate/prostate_track_v2/dataset_coco_triplet"

SPLITS = ["split1", "split2", "split3", "split4", "split5"]

SUBSETS = ["train", "val", "test"]

def get_triplet_categories():
    annotation_file = os.path.join(TRIPLET_DATASET_ROOT, "split1", "train_annotations.json")
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    thing_classes = [cat['name'] for cat in data['categories']]
    
    thing_dataset_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(data['categories'])}
    
    return {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id
    }

def register_triplet_datasets():
    """
    Register all triplet dataset splits and subsets.
    """
    metadata = get_triplet_categories()
    
    print(f"Registering triplet dataset with {len(metadata['thing_classes'])} categories...")
    print(f"Categories: {metadata['thing_classes'][:5]}..." if len(metadata['thing_classes']) > 5 else f"Categories: {metadata['thing_classes']}")
    
    registered_datasets = []
    
    for split in SPLITS:
        for subset in SUBSETS:
            dataset_name = f"triplet_{split}_{subset}"
            
            json_file = os.path.join(TRIPLET_DATASET_ROOT, split, f"{subset}_annotations.json")
            image_root = os.path.join(TRIPLET_DATASET_ROOT, split, subset)
            
            if not os.path.exists(json_file):
                print(f"Warning: Annotation file not found: {json_file}")
                continue
            if not os.path.exists(image_root):
                print(f"Warning: Image directory not found: {image_root}")
                continue
            
            register_coco_instances(
                dataset_name,
                metadata,
                json_file,
                image_root
            )
            
            registered_datasets.append(dataset_name)
            print(f"Registered: {dataset_name}")
    
    print(f"\nSuccessfully registered {len(registered_datasets)} datasets:")
    for dataset in registered_datasets:
        print(f"  - {dataset}")
    
    return registered_datasets

def verify_registration():
    """
    Verify that datasets are properly registered and can be loaded.
    """
    print("\nVerifying dataset registration...")
    
    test_dataset = "triplet_split1_train"
    
    try:
        if test_dataset in DatasetCatalog:
            print(f"✓ {test_dataset} found in DatasetCatalog")
            
            dataset_dicts = DatasetCatalog.get(test_dataset)
            print(f"✓ Loaded {len(dataset_dicts)} samples from {test_dataset}")
            
            metadata = MetadataCatalog.get(test_dataset)
            print(f"✓ Metadata loaded: {len(metadata.thing_classes)} classes")
            
            if len(dataset_dicts) > 0:
                sample = dataset_dicts[0]
                print(f"✓ Sample info: {sample['file_name']}, {len(sample['annotations'])} annotations")
                
        else:
            print(f"✗ {test_dataset} not found in DatasetCatalog")
            
    except Exception as e:
        print(f"✗ Error loading {test_dataset}: {e}")

if __name__ == "__main__":
    registered = register_triplet_datasets()
    
    verify_registration()
    
    print(f"\nDataset registration complete!")
    print(f"You can now use these dataset names in your config files:")
    print(f"DATASETS:")
    print(f"  TRAIN: ('triplet_split1_train',)")
    print(f"  TEST: ('triplet_split1_val',)")
