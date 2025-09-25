#!/usr/bin/env python3
"""
YOLO Self-Distillation Training Script using Ultralytics Framework

This script integrates self-distillation training into the ultralytics framework
using a custom SelfDistillationTrainer.

Usage:
    python train_distill_ultralytics.py --data dataset.yaml --epochs 100
    python train_distill_ultralytics.py --mode test --weights runs/distill/train/weights/best.pt
"""

import argparse
import os
from pathlib import Path
from mtl_detect import MTLDetect, register_mtl_detect
register_mtl_detect()
# Import custom trainer
from distill_trainer import SelfDistillationTrainer
from ultralytics import YOLO, settings

# Enable tensorboard logging
settings['tensorboard'] = True


def train_distillation(
    data="/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml",
    teacher_path="yolo12m_weight/best.pt",
    epochs=30,
    batch_size=16,
    imgsz=640,
    device="0",
    project="runs/distill",
    name="train",
    distill_alpha=0.5,
    distill_temp=4.0,
    target_nc=89,
    **kwargs
):
    print("=" * 80)
    print("YOLO SELF-DISTILLATION TRAINING (Ultralytics Framework)")
    print("=" * 80)
    
    # Verify teacher model exists
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
    
    # Verify dataset exists
    if not os.path.exists(data):
        raise FileNotFoundError(f"Dataset not found: {data}")
    
    print(f"Student model: {teacher_path} (same as teacher for bbox consistency)")
    print(f"Teacher model: {teacher_path}")
    print(f"Dataset: {data}")
    print(f"Target classes: {target_nc}")
    print(f"Distillation alpha: {distill_alpha}")
    print(f"Temperature: {distill_temp}")

    # Load student model (same weights as teacher for bbox consistency)
    model = YOLO(teacher_path)

    # Set custom trainer
    model.trainer = SelfDistillationTrainer
    print(f"Using trainer: {model.trainer.__name__}")

    # Training arguments
    train_args = {
        'data': data,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'project': project,
        'name': name,
        'teacher_path': teacher_path,
    }

    # Add additional kwargs
    train_args.update(kwargs)

    # Debug: Print training arguments
    print("Training arguments:")
    for key, value in train_args.items():
        print(f"   {key}: {value}")

    # Setup freeze strategy using ultralytics native approach
    head_index = 21  # Detection head is layer 21 for YOLOv12m

    # Freeze backbone (layers 0-20)
    freeze_list = [f"{k}" for k in range(head_index)]

    # Freeze detection head CV2 and DFL, keep CV3 trainable
    freeze_list += [f"{head_index}.cv2"]  # Freeze bbox regression
    freeze_list += [f"{head_index}.dfl"]  # Freeze distribution focal loss
    # Note: CV3 (classification) remains trainable for self-distillation

    print(f"🧊 Freeze strategy: {len(freeze_list)} components frozen")
    print(f"   - Backbone: layers 0-{head_index-1}")
    print(f"   - Detection head: CV2, DFL")
    print(f"   - Trainable: CV3 (classification head)")

    # Add freeze parameter to training arguments
    train_args['freeze'] = freeze_list

    # Start training with distillation parameters
    results = model.train(trainer=SelfDistillationTrainer, **train_args)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {results.save_dir}")
    print(f"Best weights: {results.save_dir}/weights/best.pt")
    print(f"Last weights: {results.save_dir}/weights/last.pt")
    
    return results


def test_distillation(
    weights="runs/distill/train/weights/best.pt",
    data="/ssd/prostate/prostate_track_v2/dataset_yolo_triplet",
    imgsz=640,
    device="0",
    **kwargs
):
    """
    Test/validate distilled YOLO model.
    
    Args:
        weights: Path to trained model weights
        data: Path to dataset YAML file
        imgsz: Image size
        device: Device to use
        **kwargs: Additional validation arguments
    """
    print("=" * 80)
    print("YOLO SELF-DISTILLATION TESTING")
    print("=" * 80)
    
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found: {weights}")
    
    print(f"Model weights: {weights}")
    print(f"Dataset: {data}")
    
    # Load trained model
    model = YOLO(weights)
    
    # Run validation
    results = model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        **kwargs
    )
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results


def main():
    """Main function with argument parsing."""
    import sys

    # Save original sys.argv and clear it to avoid ultralytics CLI parsing conflicts
    original_argv = sys.argv.copy()
    sys.argv = ['train_distill.py']  # Keep only script name

    parser = argparse.ArgumentParser(description="YOLO Self-Distillation Training")
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'test'], default='train', 
                       help='Mode: train or test')
    
    # Model and data paths
    parser.add_argument('--data', default='/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml',
                       help='Dataset YAML path')
    parser.add_argument('--teacher', default='yolo12m_weight/best.pt',
                       help='Teacher model path (also used as student base for bbox consistency)')
    parser.add_argument('--weights', default='runs/distill/train/weights/best.pt',
                       help='Model weights for testing')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size')
    parser.add_argument('--device', default='0', 
                       help='Device (e.g., 0, cpu)')
    
    # Distillation parameters
    parser.add_argument('--distill-alpha', type=float, default=0.5,
                       help='Distillation loss weight')
    parser.add_argument('--distill-temp', type=float, default=4.0,
                       help='Temperature for distillation')
    parser.add_argument('--target-nc', type=int, default=89,
                       help='Target number of classes')
    
    # Output parameters
    parser.add_argument('--project', default='runs/distill',
                       help='Project directory')
    parser.add_argument('--name', default='train',
                       help='Experiment name')
    
    # Parse arguments from original argv
    args = parser.parse_args(original_argv[1:])

    if args.mode == 'train':
        # Training mode
        train_distillation(
            data=args.data,
            teacher_path=args.teacher,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            distill_alpha=args.distill_alpha,
            distill_temp=args.distill_temp,
            target_nc=args.target_nc
        )
    else:
        # Testing mode
        test_distillation(
            weights=args.weights,
            data=args.data,
            imgsz=args.imgsz,
            device=args.device
        )


if __name__ == "__main__":
    main()
