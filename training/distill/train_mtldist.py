#!/usr/bin/env python3
import argparse
import os
import sys

from mtl_detect import MTLDetect, register_mtl_detect
register_mtl_detect()

from mtldist_trainer import MTLDistillTrainer
from ultralytics import YOLO, settings

settings['tensorboard'] = True

def train_mtl_distillation(
    data="/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml",
    teacher_path="mtl/split1/weights/best.pt",
    model_path="yolo12m-mtl.yaml",
    epochs=100,
    batch_size=16,
    imgsz=640,
    device="0",
    project="runs/mtldist",
    name="train",
    target_nc=89,
    pretrained= None,
    **kwargs
):
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"MTL teacher model not found: {teacher_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model config not found: {model_path}")
    if not os.path.exists(data):
        raise FileNotFoundError(f"Dataset not found: {data}")
    
    model = YOLO(teacher_path)
    model.trainer = MTLDistillTrainer

    train_args = {
        'data': data,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'device': device,
        'project': project,
        'name': name,
        'teacher_path': teacher_path,  
        'pretrained': pretrained,      
        'val': True,    
        'plots': True,  
    }
    
    train_args.update(kwargs)
    print("Training arguments:")
    for key, value in train_args.items():
        print(f"   {key}: {value}")

    head_index = 21 
    
    freeze_list = [f"{k}" for k in range(head_index)]
    
    freeze_list += [f"{head_index}.cv2"]  
    freeze_list += [f"{head_index}.dfl"]  

    train_args['freeze'] = freeze_list

    results = model.train(trainer=MTLDistillTrainer, **train_args)
    
    print("TRAINING COMPLETED")
    
    return results


def test_mtl_distillation(
    weights="runs/mtldist/train/weights/best.pt",
    data="/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml",
    imgsz=640,
    device="0",
    project="runs/mtldist",
    name="train",
    **kwargs
):

    if not os.path.exists(weights):
        raise FileNotFoundError(f"Model weights not found: {weights}")
    
    model = YOLO(weights)
    
    test_args = {
        'data': data,
        'imgsz': imgsz,
        'device': device,
        'project': project,
        'name': name,
        'val': True, 
        'plots': True,
        'split': 'test',
    }  

    results = model.val(
        **test_args
    )
    
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results


def main():
    original_argv = sys.argv.copy()
    sys.argv = ['train_mtldist.py'] 

    parser = argparse.ArgumentParser(description="Training")
    
    parser.add_argument('--mode', choices=['train', 'test'], default='train', 
                       help='Mode: train or test')
    parser.add_argument('--data', default='/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml',
                       help='Dataset YAML path')
    parser.add_argument('--teacher', default='mtl/split1/weights/best.pt',
                       help='MTL teacher model weights path')
    parser.add_argument('--model', default='yolo12m-mtl.yaml',
                       help='MTL model config YAML file')
    parser.add_argument('--weights', default='runs/mtldist/train/weights/best.pt',
                       help='Model weights for testing')
    parser.add_argument('--pretrained', default='yolo12m.pt',
                       help='Pretrained backbone weights')
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
    parser.add_argument('--project', default='rmtldist',
                       help='Project directory')
    parser.add_argument('--name', default='train',
                       help='Experiment name')
    args = parser.parse_args(original_argv[1:])

    if args.mode == 'train':
        # Training mode
        train_mtl_distillation(
            data=args.data,
            teacher_path=args.teacher,
            model_path=args.model,
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
        test_mtl_distillation(
            weights=args.weights,
            data=args.data,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
        )
if __name__ == "__main__":
    main()
