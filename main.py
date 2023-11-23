import argparse

from emotion_detector import EmotionDetector

parser = argparse.ArgumentParser(description="Lucas Boscatti's Final Project Nero")
parser.add_argument("--arch", default="ResNet18", type=str)
parser.add_argument("--accelerator", default="cuda", type=str, help="[cuda, cpu]")
parser.add_argument("--model_name", default="resnet18.onnx", type=str)
parser.add_argument("--backend_option", default=1, type=int, help="""Choose one of the backend-target pair to run this demo:
                        1: (default) OpenCV implementation + CPU,
                        2: CUDA + GPU (CUDA),
                        3: CUDA + GPU (CUDA FP16)""",
)
parser.add_argument("--providers", default=None, type=int, help="""Choose one of the backend-target pair to run this demo:
                        1: (default) CPUExecutionProvider,
                        2: CUDAExecutionProvider (CUDA),
                        3: TensorrtExecutionProvider (CUDA FP16)""",
)
parser.add_argument("--file", default=0, type=str, help=["0", "realsense", "path/to/video or image"])
parser.add_argument("--display_window", default=True, type=bool, help="True or False")
parser.add_argument("--option", default="video", type=str, help=["video", "image"])


if __name__ == "__main__":
    args = parser.parse_args()

    detector = EmotionDetector(
        accelerator=args.accelerator,
        backend_option=args.backend_option,
        model_name=args.model_name,
        providers=args.providers,
    )

    if args.option == "image":
        detector.process_image(args.file)

    detector.process_video(args.file, display_window=args.display_window)