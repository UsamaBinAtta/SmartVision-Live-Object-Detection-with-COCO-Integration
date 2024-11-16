# Object Detection Project

## Overview

This project demonstrates an object detection system using a pre-trained Faster R-CNN model. The system is capable of identifying and localizing objects in real-time from a camera feed.

## Project Structure

- `model.py`: Contains functions for loading the Faster R-CNN model and making predictions.
- `utils.py`: Includes helper functions for drawing bounding boxes, mapping colors, and other utilities.
- `main.py`: Handles the live object detection and manages the camera feed.

## Installation

To set up this project, you'll need to install the required dependencies. Follow these steps:

1. **Create a Virtual Environment** (if you haven't already):

   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

3. **Install Required Packages**:

   ```bash
   pip install torch torchvision opencv-python
   ```

## Usage

1. **Run the Object Detection Script**:

   ```bash
   python main.py
   ```

2. **View the Results**:

   The script will open a camera feed window where detected objects will be highlighted with bounding boxes.

## Error Handling

The project includes error handling to manage potential issues with camera operations and model inference. Ensure that your camera is connected and accessible.

## Contribution

Feel free to contribute to the project by creating issues or submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Faster R-CNN model is provided by [TorchVision](https://pytorch.org/vision/stable/index.html).
- Special thanks to the [COCO dataset](https://cocodataset.org/) for the class labels.
