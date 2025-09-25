#!/usr/bin/env python3
"""
Simple script to register triplet datasets.
Place this in the detectron2 root directory and run before training.
"""

import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

TRIPLET_DATASET_ROOT = "/ssd/prostate/prostate_track_v2/dataset_coco_triplet"

SPLITS = ["split1", "split2", "split3", "split4", "split5"]
SUBSETS = ["train", "val", "test"]

def get_triplet_categories():
    """Get category information from the first split's train annotations."""
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
    """Register all triplet dataset splits and subsets."""
    metadata = get_triplet_categories()
    
    print(f"Registering triplet dataset with {len(metadata['thing_classes'])} categories...")
    
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
    
    print(f"\nSuccessfully registered {len(registered_datasets)} datasets")
    return registered_datasets

if __name__ == "__main__":
    register_triplet_datasets()
    print("Dataset registration complete!")
