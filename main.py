import cv2
from model import load_model, get_predictions
from utils import draw_boxes, COLOR_DICT

def live_object_detection():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")

        model = load_model()
        if model is None:
            raise RuntimeError("Failed to load the model.")

        # Resize the frame once and keep the size consistent for performance
        width, height = 640, 480

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            # Resize the frame once here
            frame_resized = cv2.resize(frame, (width, height))

            # Run predictions on the resized frame
            boxes, classes, scores = get_predictions(model, frame_resized)

            # Draw boxes on the frame
            frame_with_boxes = draw_boxes(frame_resized, boxes, classes, scores, COLOR_DICT)

            # Display the frame with the detected objects
            cv2.imshow('Live Object Detection', frame_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error during live object detection: {e}")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    live_object_detection()