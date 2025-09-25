#!/usr/bin/env python3
"""
Generate configuration files for different dataset splits.

This script creates separate config files for each split, making it easier
to run experiments without modifying the base configs.
"""

import os
import yaml
from pathlib import Path

# Base configurations
BASE_CONFIGS = {
    'faster_rcnn': 'configs/Triplet-Detection/faster_rcnn_R_50_FPN_triplet.yaml',
    'retinanet': 'configs/Triplet-Detection/retinanet_R_50_FPN_triplet.yaml',
    'cascade_rcnn': 'configs/Triplet-Detection/cascade_rcnn_R_50_FPN_triplet.yaml',
    'vitdet': 'configs/Triplet-Detection/vitdet_base_triplet.yaml',
}

SPLITS = ['split1', 'split2', 'split3', 'split4', 'split5']

def generate_split_config(base_config_path, model_name, split):
    """
    Generate a config file for a specific split.
    """
    # Read base config
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # Create split-specific config content
    # Use relative path from splits directory to parent directory
    relative_base_path = f"../{os.path.basename(base_config_path)}"

    split_config_content = f"""# {model_name.upper()} for Triplet Detection - {split.upper()}
# Auto-generated configuration file

_BASE_: ["{relative_base_path}"]

# Dataset configuration for {split}
DATASETS:
  TRAIN: ("triplet_{split}_train",)
  TEST: ("triplet_{split}_val",)

# Output directory for this split
OUTPUT_DIR: "./output/triplet_{model_name}_{split}"
"""
    
    # Create output directory
    output_dir = f"configs/Triplet-Detection/splits"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write split-specific config
    output_path = os.path.join(output_dir, f"{model_name}_{split}.yaml")
    with open(output_path, 'w') as f:
        f.write(split_config_content)
    
    return output_path

def generate_all_configs():
    """
    Generate all split-specific configuration files.
    """
    print("Generating split-specific configuration files...")
    
    generated_configs = []
    
    for model_name, base_config in BASE_CONFIGS.items():
        print(f"\nGenerating configs for {model_name}:")
        
        for split in SPLITS:
            try:
                output_path = generate_split_config(base_config, model_name, split)
                generated_configs.append(output_path)
                print(f"  ✓ Generated: {output_path}")
            except Exception as e:
                print(f"  ✗ Failed to generate {model_name}_{split}: {e}")
    
    print(f"\n✓ Generated {len(generated_configs)} configuration files")
    
    # Create a summary file
    summary_path = "configs/Triplet-Detection/splits/README.md"
    with open(summary_path, 'w') as f:
        f.write("# Split-Specific Configuration Files\n\n")
        f.write("This directory contains auto-generated configuration files for different dataset splits.\n\n")
        f.write("## Available Configurations\n\n")
        
        for model_name in BASE_CONFIGS.keys():
            f.write(f"### {model_name.upper()}\n")
            for split in SPLITS:
                config_name = f"{model_name}_{split}.yaml"
                f.write(f"- `{config_name}` - {model_name} trained on {split}\n")
            f.write("\n")
        
        f.write("## Usage\n\n")
        f.write("```bash\n")
        f.write("# Train a specific model on a specific split\n")
        f.write("python tools/train_net.py --config-file configs/Triplet-Detection/splits/faster_rcnn_split1.yaml\n")
        f.write("\n")
        f.write("# Or use the experiment runner\n")
        f.write("python tools/run_triplet_experiments.py --model faster_rcnn --split split1\n")
        f.write("```\n")
    
    print(f"✓ Created summary: {summary_path}")
    
    return generated_configs

if __name__ == "__main__":
    generate_all_configs()
