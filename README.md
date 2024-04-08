# Project README

## Lucas Boscatti's Final Project Nero

This project is a facial emotion detection system developed by Lucas Boscatti. It utilizes deep learning techniques to recognize and classify emotions in real-time from images or video sources. The project provides a user-friendly interface for running the emotion detection system with various options and configurations.

### Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy

### Installation

1. Clone this repository to your local machine:

   ```bash
   git clone <repository_url>

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

You can run the emotion detection system with different configurations using command-line arguments.

    ```bash
    python3 main.py [--model_name MODEL_NAME] [--model_option MODEL_OPTION] [--backend_option BACKEND_OPTION] [--providers PROVIDERS] [--file FILE] [--display_window DISPLAY_WINDOW] [--game_mode GAME_MODE]
    ```

Arguments
- --model_name: Specify the name of the model file (default: "resnet18.onnx").
- --model_option: Specify the model option (default: "onnx").
- --backend_option: Choose the backend-target pair to run the demo (default: based on GPU availability). Options: 1 - OpenCV implementation + CPU, 2 - CUDA + GPU (CUDA), 3 - CUDA + GPU (CUDA FP16).
- --providers: Choose the execution provider (default: based on GPU availability). Options: 1 - CPUExecutionProvider, 2 - CUDAExecutionProvider (CUDA), 3 - TensorrtExecutionProvider (CUDA FP16).
- --file: Specify the input source. Options: '0' - default webcam, 'realsense' - Realsense camera, 'path_to/video' - video or image file (default: "0").
- --display_window: Specify whether to display the window (True or False, default: True).
- --game_mode: Specify the game mode. If True, only two faces will be detected, and the output of the winner will be displayed (True or False, default: False).

### Examples

### Basic usage

    ```bash
    python3 main.py --file "path_to/video.mp4"
    ```

### Game mode

    ```bash
    python3 main.py --file "0" --game_mode True
    ```

### Complex usage

    ```bash
    python main.py --model_name "emotion_model.onnx" --file "path_to/video.mp4" --display_window False --game_mode False
    ```
