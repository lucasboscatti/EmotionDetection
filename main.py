import argparse

import torch

from emotion_detector import EmotionDetector

CUDA = torch.cuda.is_available()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Lucas Boscatti's Final Project Nero")
    parser.add_argument("--model_name", default="resnet18.onnx", type=str)
    parser.add_argument("--model_option", default="onnx", type=str)
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
        help="""Specify the input source: 
        '0' to default webcam
        'realsense' to use the realsense camera
        'path_to/video' to use a video or a image file""",
    )
    parser.add_argument(
        "--display_window",
        default=True,
        type=bool,
        help="Specify whether to display the window (True or False)",
    )
    parser.add_argument(
        "--game_mode",
        default=False,
        type=bool,
        help="Specify the game mode: True or False. If True, only two faces will be detected and a output of the winner will be displayed",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    detector = EmotionDetector(
        model_name=args.model_name,
        model_option=args.model_option,
        backend_option=args.backend_option,
        providers=args.providers,
        game_mode=args.game_mode,
    )

    detector.start_inference(args.file, display_window=args.display_window)


if __name__ == "__main__":
    main()
