# Surgical Triplet Detection

A simple training script for surgical action triplet detection using YOLO.

## 🚀 Installation

```bash
# Clone the repository

# yolo series 
cd training/framework/lib
# in python3.8.20, pytorch2.4.1, cuda12.4 
pip install -e ./
cd ..

# detr series (for def-detr, PARIP, MCIT-IG.....)
cd detection/Deformable-DETR
# python3.8.20, pytorch2.4.1, cuda12.4 
pip install -r requirements.txt
cd ./models/ops
sh ./make.sh
cd ..

## detectron2 series 
# in python3.8.20, pytorch2.4.1, cuda12.4 
python -m pip install -e detectron2
cd ..
```

If you want to use our calculation metrics, please follow this [ivtdmetrics](../ivtdmetrics/).

## 📂 YOLO Dataset Structure (5-Fold Cross Validation)

The dataset should be organized with 5-fold cross validation structure:

```
dataset_yolo_triplet/
├── triplet_maps_v2.txt         # Triplet to tool mapping file
├── split1/
│   ├── dataset_triplet.yaml    # Dataset configuration file for split1
│   ├── train/
│   │   ├── images/             # Training images (.jpg)
│   │   └── labels/             # Training labels (.txt)
│   ├── val/
│   │   ├── images/             # Validation images (.jpg)
│   │   └── labels/             # Validation labels (.txt)
│   └── test/
│       ├── images/             # Test images (.jpg)
│       └── labels/             # Test labels (.txt)
├── split2/
│   ├── dataset_triplet.yaml    # Dataset configuration file for split2
│   ├── train/
│   ├── val/
│   └── test/
├── split3/
│   ├── dataset_triplet.yaml
│   ├── train/
│   ├── val/
│   └── test/
├── split4/
│   ├── dataset_triplet.yaml
│   ├── train/
│   ├── val/
│   └── test/
└── split5/
    ├── dataset_triplet.yaml
    ├── train/
    ├── val/
    └── test/
```

## 📁 COCO Dataset Structure (5-Fold Cross Validation)

The COCO format dataset should be organized as follows:

```
dataset_coco_triplet/
├── triplet_maps_v2.txt         # Triplet to tool mapping file
├── split1/
│   ├── train/                  # Training images (.jpg)
│   ├── val/                    # Validation images (.jpg)
│   ├── test/                   # Test images (.jpg)
│   ├── train_annotations.json  # Training annotations in COCO format
│   ├── val_annotations.json    # Validation annotations in COCO format
│   └── test_annotations.json   # Test annotations in COCO format
├── split2/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── train_annotations.json
│   ├── val_annotations.json
│   └── test_annotations.json
├── split3/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── train_annotations.json
│   ├── val_annotations.json
│   └── test_annotations.json
├── split4/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── train_annotations.json
│   ├── val_annotations.json
│   └── test_annotations.json
└── split5/
    ├── train/
    ├── val/
    ├── test/
    ├── train_annotations.json
    ├── val_annotations.json
    └── test_annotations.json
```

### 🏷️ Label Format

#### YOLO Format
Each label file contains annotations in YOLO format:

```
class_id center_x center_y width height
```

Where:
- `class_id`: Triplet class ID (0-88)
- `center_x, center_y`: Normalized center coordinates (0-1)
- `width, height`: Normalized bounding box dimensions (0-1)

#### COCO Format
Annotations are stored in JSON files following COCO format:

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "1.jpg",
            "width": 1920,
            "height": 1080
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": ["x", "y", "width", "height"],
            "area": 12345,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "instrument_verb_target",
            "supercategory": "triplet"
        }
    ]
}
```

Where:
- `bbox`: [x, y, width, height] in absolute pixel coordinates 
- `category_id`: Triplet class ID (1-89, COCO format starts from 1)
- `area`: Bounding box area in pixels

<a id="bench_yolo"></a>
### 🏋️ Yolo Training Example

```bash
python train_yolo.py \
    --data /path/to/dataset.yaml \
    --model yolo12m.pt \
    --epochs 150 \
    --batch 16 \
    --mapping-file /path/to/triplet_maps_v2.txt \
    --apply-ivt-metrics true
```

### 🧪 Yolo Testing Example

```bash
python train_yolo.py \
    --data /path/to/dataset.yaml \
    --test-only \
    --model yolo12m.pt \
    --name yolov12 \
    --weights /path/to/best.pt \
    --mapping-file /path/to/triplet_maps_v2.txt
```

### ⚙️ Yolo Parameters

- **`--agnostic-nms`**: Use class-agnostic NMS (default: False)
- **`--tool-nms`**: Apply tool-based NMS patch
- **`--mapping-file`**: Path to triplet→tool mapping file
- **`--apply-ivt-metrics`**: Apply IVT Detection metrics patch for mAP calculation (default: True)

<a id="bench_detr"></a>
### 🤖 DETR Training and Testing Examples

```bash
bash detr/train_triplet_detr.sh    # training
```

```bash
bash detr/run_triplet_inference.sh # testing
```

```bash
python framework/train_rtdetr.py # training and testing
```

Make sure to modify the paths in the script according to your setup.


All these results above can calculate the mAP using following command: 
```bash
python calculate_ivtd.py # based on ivtdmetrics
```

<a id="bench_tdnet"></a>
### 🧠 TDnet Training and Testing Workflow

```bash
# Navigate to the Working Directory
cd distill

# Train the Teacher Model
python train_mtl.py \
--data "/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml" \
--batch "16" \
--project "mtl" \
--name "split1" \
--weights "./yolo12m_weight/yolo12m.pt" \
--model "yolo12m-mtl.yaml" \
--epochs "120" \
--pretrained "./yolo12m_weight/yolo12m.pt"

# Train and inference the Student Model
python train_mtldist.py \
--data "/ssd/prostate/prostate_track_v2/dataset_yolo_triplet/split1/dataset_triplet.yaml" \
--batch "16" \
--project "mtl_dist" \
--name "split1" \
--teacher "mtl/split1/weights/best.pt" \
--epochs "50"

# Calculate final metrics from the inference results
python calculate_ivtd.py \
--inference_path "/path/to/runs/detect/inference_run" \
--ground_truth_path "/path/to/your/ground_truth_labels"

```

### 💻 Other Training and Testing Examples
please check our [detectron2](./detectron2) folder


<a id="postprocess"></a>
### 🖼️ Postprocess (Visualization in Triplet-labelme or SurgLabel)

To visualize the original image shape of predictions in our custom annotation tool using LabelMe format, please follow the steps below.

```bash
# Navigate to the postprocess directory:
cd postprocess

# Convert predictions to LabelMe format:
python convert_yolo_to_labelme.py

# Add transformation parameters to the converted LabelMe JSON:
# This script applies the shape size parameters (e.g., 640x640) from the conversion step to the prediction results in LabelMe format.
python add_transform_info_yolo_predict_json.py

# Restore predictions to the original image shape:
python resize_640.py --restore

# Open our triplet-labelme or SurgLabel to visualize the annotation
```

## 🙏 Acknowledgments

This code is based on the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR).
