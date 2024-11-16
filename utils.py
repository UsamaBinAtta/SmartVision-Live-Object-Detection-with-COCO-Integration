import cv2

def transform_image(img):
    try:
        return cv2.resize(img, (640, 480))
    except Exception as e:
        print(f"Error resizing image: {e}")
        return img

COLOR_DICT = {
    'person': (0, 255, 0),
    'bicycle': (72, 61, 139),
    'car': (0, 0, 255),
    'motorcycle': (0, 255, 255),
    'airplane': (255, 0, 0),
    'bus': (255, 215, 0),
    'train': (0, 255, 0),
    'truck': (139, 69, 19),
    'boat': (255, 192, 203),
    'traffic light': (255, 105, 180),
    'fire hydrant': (255, 69, 0),
    'stop sign': (255, 0, 255),
    'parking meter': (0, 128, 128),
    'bench': (0, 0, 139),
    'bird': (0, 255, 127),
    'cat': (255, 0, 0),
    'dog': (0, 0, 255),
    'horse': (0, 255, 255),
    'sheep': (255, 20, 147),
    'cow': (255, 140, 0),
    'elephant': (128, 128, 128),
    'bear': (128, 0, 128),
    'zebra': (0, 128, 0),
    'giraffe': (255, 255, 0),
    'backpack': (0, 139, 139),
    'umbrella': (0, 255, 255),
    'handbag': (139, 69, 19),
    'tie': (255, 20, 147),
    'suitcase': (0, 128, 128),
    'frisbee': (255, 99, 71),
    'skis': (210, 105, 30),
    'snowboard': (0, 128, 128),
    'sports ball': (255, 69, 0),
    'kite': (255, 215, 0),
    'baseball bat': (128, 0, 128),
    'baseball glove': (255, 69, 0),
    'skateboard': (255, 20, 147),
    'surfboard': (0, 139, 139),
    'tennis racket': (0, 255, 255),
    'bottle': (255, 215, 0),
    'wine glass': (255, 69, 0),
    'cup': (255, 0, 255),
    'fork': (0, 255, 0),
    'knife': (255, 105, 180),
    'spoon': (139, 69, 19),
    'bowl': (255, 20, 147),
    'banana': (255, 255, 0),
    'apple': (255, 0, 0),
    'sandwich': (0, 255, 255),
    'orange': (255, 140, 0),
    'broccoli': (0, 128, 0),
    'carrot': (255, 165, 0),
    'hot dog': (255, 105, 180),
    'pizza': (255, 69, 0),
    'donut': (255, 192, 203),
    'cake': (139, 69, 19),
    'chair': (255, 215, 0),
    'couch': (0, 128, 128),
    'potted plant': (0, 255, 127),
    'bed': (139, 69, 19),
    'dining table': (255, 140, 0),
    'toilet': (255, 105, 180),
    'TV': (0, 255, 255),
    'laptop': (255, 69, 0),
    'mouse': (0, 255, 0),
    'remote': (0, 128, 128),
    'keyboard': (255, 20, 147),
    'cell phone': (255, 0, 255),
    'microwave': (255, 192, 203),
    'oven': (255, 105, 180),
    'toaster': (0, 255, 255),
    'sink': (128, 128, 128),
    'refrigerator': (0, 128, 128),
    'book': (255, 20, 147),
    'clock': (255, 0, 255),
    'vase': (0, 255, 127),
    'scissors': (0, 0, 255),
    'teddy bear': (255, 140, 0),
    'hair drier': (128, 0, 128),
    'toothbrush': (255, 105, 180),
    '__default__': (255, 0, 255)  # Default color for unknown labels
}

def draw_boxes(frame, boxes, classes, scores, color_dict):
    try:
        for box, label, score in zip(boxes, classes, scores):
            color = color_dict.get(label, color_dict['__default__'])
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    except Exception as e:
        print(f"Error drawing boxes: {e}")
        return frame