import argparse

import torch
from flask import Flask, Response, render_template

from emotion_detector import EmotionDetector

CUDA = torch.cuda.is_available()
app = Flask(__name__, template_folder="template")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Lucas Boscatti's Final Project Nero")
    parser.add_argument("--model_name", default="resnet18.onnx", type=str)
    parser.add_argument(
        "--model_option",
        default="onnx",
        type=str,
        choices=["pytorch", "onnx", "tensorrt"],
        help="""Choose the model to run:
        1: pytorch
        2: onnx
        3: tensorrt""",
    )
    parser.add_argument(
        "--backend_option",
        default=2 if CUDA else 1,
        type=int,
        choices=[1, 2, 3],
        help="""Choose the backend-target pair to run this demo:
        1: OpenCV implementation + CPU,
        2: CUDA + GPU (CUDA)
        3: CUDA + GPU (CUDA FP16)""",
    )
    parser.add_argument(
        "--providers",
        default=2 if CUDA else 1,
        type=int,
        choices=[1, 2, 3],
        help="""Choose the backend-target pair to run this demo:
        1: CPUExecutionProvider
        2: CUDAExecutionProvider (CUDA)
        3: TensorrtExecutionProvider (CUDA FP16)""",
    )
    parser.add_argument(
        "--file",
        default="0",
        type=str,
        choices=["0", "realsense", "path_to/video"],
        help="""Specify the input source: 
        '0' to default webcam
        'realsense' to use the realsense camera. 
        'path_to/video' to use a video or a image file""",
    )
    parser.add_argument(
        "--game_mode",
        default=False,
        type=bool,
        help="Specify the game mode: True or False. If True, only two faces will be detected and a output of the winner will be displayed",
    )

    return parser.parse_args()


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        detector.start_inference(args.file),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


args = parse_arguments()

detector = EmotionDetector(
    model_name=args.model_name,
    model_option=args.model_option,
    backend_option=args.backend_option,
    providers=args.providers,
    game_mode=args.game_mode,
)

app.run(debug=True)
