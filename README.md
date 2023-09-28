# Facial Emotion Recognizer

This Python code provides a facial emotion recognition system capable of detecting and recognizing emotions in images or video frames. It uses a pre-trained deep learning model for emotion classification on faces. This README will guide you through the code, its functionalities, and how to use it.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Example Usage](#example-usage)
- [License](#license)

## Requirements

To use this code, you need the following dependencies:

- Python 3.x
- OpenCV (cv2)
- NumPy
- PyTorch
- torchvision
- imutils
- Pillow (PIL)

You can install these dependencies using `pip`:

```bash
pip install opencv-python numpy torch torchvision imutils pillow
```

## Usage

To use the Facial Emotion Recognizer, follow these steps:

1. Clone or download this repository.

2. Download the necessary model files:

- res10_300x300_ssd_iter_140000.prototxt
- res10_300x300_ssd_iter_140000.caffemodel
- FER_trained_model.pt

3. Place the model files in the models directory within the project folder.

4. You can then use the FacialEmotionRecognizer class from the provided code to perform facial emotion recognition on images or video frames.

## Code Overview

Here's an overview of the code and its main components:

- `FacialEmotionRecognizer`: This class encapsulates the functionality for facial emotion recognition. It loads the necessary models, processes images or video frames, and recognizes emotions.

- `EMOTION_DICT`: A dictionary mapping emotion labels (integers) to their corresponding names.

- `__init__`: The constructor of FacialEmotionRecognizer initializes the logger, loads the face detection model, sets up the device (CPU or GPU), and loads the pre-trained emotion recognition model.

- `setup_device`: This method configures the preferred device (CPU or GPU) for computation based on user settings.

- `load_trained_model`: This method loads the pre-trained emotion recognition model from a file.

- `recognize_emotion`: Given a face image, this method predicts the emotion label.

- `process_image`: Processes a single image, detects faces, and displays the recognized emotions.

- `process_video`: Processes a video file, detecting faces and recognizing emotions in each frame.

- `process_frame`: Processes a single frame from a video, including face detection and emotion recognition.

## Example Usage

Here's an example of how to use the `FacialEmotionRecognizer` class to recognize emotions in an image:

```python
from facial_emotion_recognizer import FacialEmotionRecognizer

# Create a FacialEmotionRecognizer instance
recognizer = FacialEmotionRecognizer()

# Process an image and display the result
recognizer.process_image("path/to/your/image.jpg")
```

To process a video file:

```python
# Process a video file and display the result
recognizer.process_video("path/to/your/video.mp4")
```

You can also use the webcam or a RealSense camera by passing `"realsense"` or `0` as the `video_path` parameter.