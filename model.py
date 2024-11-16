# model.py

import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
    'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
    'teddy bear', 'hair drier', 'toothbrush'
]

transform = T.Compose([T.ToTensor()])  # Define transform globally to avoid redundant calls

def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def get_predictions(model, img, threshold=0.7):
    try:
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)

        pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in outputs[0]['labels'].numpy()]
        pred_scores = outputs[0]['scores'].detach().numpy()
        pred_boxes = outputs[0]['boxes'].detach().numpy()

        pred_boxes = pred_boxes[pred_scores >= threshold].astype(int)
        pred_classes = [pred_classes[i] for i in range(len(pred_scores)) if pred_scores[i] >= threshold]
        pred_scores = pred_scores[pred_scores >= threshold]

        return pred_boxes, pred_classes, pred_scores
    except Exception as e:
        print(f"Error in prediction: {e}")
        return [], [], []
