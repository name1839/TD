#!/usr/bin/env python3
"""
Training script for triplet detection that automatically registers datasets.

This script is a wrapper around the standard train_net.py that ensures
triplet datasets are registered before training starts.
"""

import os
import sys
import json

# Add detectron2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Register triplet datasets first
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def register_triplet_datasets():
    """Register all triplet dataset splits and subsets."""
    # Dataset root path
    TRIPLET_DATASET_ROOT = "/ssd/prostate/prostate_track_v2/dataset_coco_triplet"
    SPLITS = ["split1", "split2", "split3", "split4", "split5"]
    SUBSETS = ["train", "val", "test"]
    
    # Get category information from the first split's train annotations
    annotation_file = os.path.join(TRIPLET_DATASET_ROOT, "split1", "train_annotations.json")
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Extract category names for thing_classes
    thing_classes = [cat['name'] for cat in data['categories']]
    
    # Create category id mapping
    thing_dataset_id_to_contiguous_id = {cat['id']: i for i, cat in enumerate(data['categories'])}
    
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id
    }
    
    print(f"Registering triplet dataset with {len(metadata['thing_classes'])} categories...")
    
    registered_datasets = []
    
    for split in SPLITS:
        for subset in SUBSETS:
            # Dataset name following detectron2 convention
            dataset_name = f"triplet_{split}_{subset}"
            
            # Skip if already registered
            if dataset_name in DatasetCatalog:
                continue
            
            # Paths
            json_file = os.path.join(TRIPLET_DATASET_ROOT, split, f"{subset}_annotations.json")
            image_root = os.path.join(TRIPLET_DATASET_ROOT, split, subset)
            
            # Check if files exist
            if not os.path.exists(json_file):
                print(f"Warning: Annotation file not found: {json_file}")
                continue
            if not os.path.exists(image_root):
                print(f"Warning: Image directory not found: {image_root}")
                continue
            
            # Register the dataset
            register_coco_instances(
                dataset_name,
                metadata,
                json_file,
                image_root
            )
            
            registered_datasets.append(dataset_name)
            print(f"Registered: {dataset_name}")
    
    print(f"Successfully registered {len(registered_datasets)} datasets")
    return registered_datasets

# Register datasets before importing training modules
print("Registering triplet datasets...")
register_triplet_datasets()
print("Dataset registration complete!")

# Now import and run the standard training script
if __name__ == "__main__":
    # Import the standard train_net after datasets are registered
    from tools.train_net import invoke_main

    # Run the standard training
    invoke_main()
