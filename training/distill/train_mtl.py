#!/usr/bin/env python3
import argparse
import os
import sys

from mtl_detect import MTLDetect, register_mtl_detect
register_mtl_detect()
from mtl_trainer import MTLTrainer
from ultralytics import YOLO, settings
settings['tensorboard'] = True

def train_triplet_detection(
    data="/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml",
    model_path="yolo12m-mtl.yaml",  
    weights="yolo12m.pt",  
    epochs=100,
    batch_size=16,
    imgsz=640,
    device="0",
    project="runs/mtl",
    name="train",
    target_nc=89,
    pretrained="yolo12m.pt",
    **kwargs
):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model config not found: {model_path}")

    if not os.path.exists(data):
        raise FileNotFoundError(f"Dataset not found: {data}")
    
    model = YOLO(model_path)

    model.trainer = MTLTrainer
    print(f"Using trainer: {model.trainer.__name__}")

    train_args = {
        'data': data,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'project': project,
        'name': name,
        'pretrained': pretrained,
        'val': True,    
        'plots': True,  
    }
    
    results = model.train(trainer=MTLTrainer, **train_args)
    return results


def test_triplet_detection(
    weights="runs/mtl/train/weights/best.pt",
    data="/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml",
    imgsz=640,
    device="0",
    **kwargs
):
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found: {weights}")
    
    model = YOLO(weights)
    
    results = model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        **kwargs
    )
    
    print("TESTING COMPLETED")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results


def main():
    original_argv = sys.argv.copy()
    sys.argv = ['train_mtl.py']
    parser = argparse.ArgumentParser(description="YOLO Multi-Task Learning Training")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', 
                       help='Mode: train or test')
    parser.add_argument('--data', default='/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml',
                       help='Dataset YAML path')
    parser.add_argument('--model', default='yolo12m-mtl.yaml',
                       help='Model config YAML file')
    parser.add_argument('--weights', default = None,
                       help='Pretrained weights path')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='Image size')
    parser.add_argument('--device', default='0', 
                       help='Device (e.g., 0, cpu)')
    parser.add_argument('--target-nc', type=int, default=89,
                       help='Target number of classes (triplets)')
    parser.add_argument('--project', default='runs/mtl',
                       help='Project directory')
    parser.add_argument('--name', default='train',
                       help='Experiment name')
    parser.add_argument('--pretrained', default='yolo12m.pt',
                       help='Pretrained weights path')
    args = parser.parse_args(original_argv[1:])

    if args.mode == 'train':
        # Training mode
        train_triplet_detection(
            data=args.data,
            model_path=args.model,
            weights=args.weights,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            target_nc=args.target_nc,
            pretrained=args.pretrained,
        )
    else:
        test_triplet_detection(
            weights=args.weights,
            data=args.data,
            imgsz=args.imgsz,
            device=args.device
        )

if __name__ == "__main__":
    main()
