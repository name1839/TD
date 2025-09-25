#!/usr/bin/env python3
"""
Experiment runner for triplet dataset detection.

This script allows you to easily run experiments with different combinations of:
- Dataset splits (split1, split2, split3, split4, split5)
- Models (Faster R-CNN, RetinaNet, Cascade R-CNN, ViT-Det)
- Training configurations

Usage examples:
    # Train Faster R-CNN on split1
    python tools/run_triplet_experiments.py --model faster_rcnn --split split1
    
    # Train all models on split1
    python tools/run_triplet_experiments.py --model all --split split1
    
    # Train Faster R-CNN on all splits
    python tools/run_triplet_experiments.py --model faster_rcnn --split all
    
    # Custom config and output directory
    python tools/run_triplet_experiments.py --model faster_rcnn --split split1 --output-dir ./custom_output
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add detectron2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Register triplet datasets
from datasets.register_triplet_dataset import register_triplet_datasets

# Model configurations mapping
MODEL_CONFIGS = {
    'faster_rcnn': 'configs/Triplet-Detection/faster_rcnn_R_50_FPN_triplet.yaml',
    'retinanet': 'configs/Triplet-Detection/retinanet_R_50_FPN_triplet.yaml',
    'cascade_rcnn': 'configs/Triplet-Detection/cascade_rcnn_R_50_FPN_triplet.yaml',
    'vitdet': 'configs/Triplet-Detection/vitdet_base_triplet.yaml',
}

AVAILABLE_SPLITS = ['split1', 'split2', 'split3', 'split4', 'split5']

def create_experiment_config(base_config, split, output_dir):
    """
    Create a temporary config file with the specified split and output directory.
    """
    import yaml
    from detectron2.config import get_cfg
    
    # Load base config
    cfg = get_cfg()
    cfg.merge_from_file(base_config)
    
    # Update dataset split
    cfg.DATASETS.TRAIN = (f"triplet_{split}_train",)
    cfg.DATASETS.TEST = (f"triplet_{split}_val",)
    
    # Update output directory
    cfg.OUTPUT_DIR = output_dir
    
    # Create temporary config file
    temp_config_path = os.path.join(output_dir, "config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(temp_config_path, 'w') as f:
        f.write(cfg.dump())
    
    return temp_config_path

def run_training(config_file, output_dir, resume=False):
    """
    Run training with the specified configuration.
    """
    cmd = [
        'python', 'tools/train_net.py',
        '--config-file', config_file,
        '--num-gpus', '1',  # Adjust based on your setup
    ]
    
    if resume:
        cmd.append('--resume')
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        print(f"✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with error: {e}")
        return False

def run_evaluation(config_file, model_path, output_dir):
    """
    Run evaluation with the trained model.
    """
    cmd = [
        'python', 'tools/train_net.py',
        '--config-file', config_file,
        '--eval-only',
        'MODEL.WEIGHTS', model_path,
    ]
    
    print(f"Running evaluation: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.dirname(__file__)))
        print(f"✓ Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run triplet detection experiments')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()) + ['all'], 
                       required=True, help='Model to train')
    parser.add_argument('--split', choices=AVAILABLE_SPLITS + ['all'], 
                       required=True, help='Dataset split to use')
    parser.add_argument('--output-dir', default='./output/triplet_experiments',
                       help='Base output directory')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation (requires trained model)')
    parser.add_argument('--model-path', help='Path to trained model for evaluation')
    
    args = parser.parse_args()
    
    # Register datasets
    print("Registering triplet datasets...")
    register_triplet_datasets()
    print("✓ Datasets registered successfully!")
    
    # Determine models and splits to run
    models = list(MODEL_CONFIGS.keys()) if args.model == 'all' else [args.model]
    splits = AVAILABLE_SPLITS if args.split == 'all' else [args.split]
    
    print(f"Running experiments for:")
    print(f"  Models: {models}")
    print(f"  Splits: {splits}")
    
    results = []
    
    for model in models:
        for split in splits:
            experiment_name = f"{model}_{split}"
            output_dir = os.path.join(args.output_dir, experiment_name)
            
            print(f"\n{'='*60}")
            print(f"Running experiment: {experiment_name}")
            print(f"{'='*60}")
            
            # Create experiment-specific config
            base_config = MODEL_CONFIGS[model]
            temp_config = create_experiment_config(base_config, split, output_dir)
            
            success = False
            
            if args.eval_only:
                # Run evaluation only
                model_path = args.model_path or os.path.join(output_dir, "model_final.pth")
                if os.path.exists(model_path):
                    success = run_evaluation(temp_config, model_path, output_dir)
                else:
                    print(f"✗ Model not found: {model_path}")
            else:
                # Run training
                success = run_training(temp_config, output_dir, args.resume)
                
                # Run evaluation if training succeeded
                if success:
                    model_path = os.path.join(output_dir, "model_final.pth")
                    if os.path.exists(model_path):
                        print(f"\nRunning evaluation for {experiment_name}...")
                        run_evaluation(temp_config, model_path, output_dir)
            
            results.append((experiment_name, success))
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for experiment_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{experiment_name:30} {status}")
    
    successful = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nCompleted: {successful}/{total} experiments successful")

if __name__ == "__main__":
    main()
