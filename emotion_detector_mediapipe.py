import logging
import sys
from time import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from imutils.video import FPS
from PIL import Image

sys.path.append("../")

from train.models import resnet


class EmotionDetectorMediapipe:
    """
    A class for detecting and recognizing emotions in images or video frames.

    Args:
        use_cuda (bool, optional): Whether to use CUDA for faster processing if a GPU is available. Default is True.
        backend_option (int, optional): Backend option for OpenCV's DNN module. Default is 1.

    Attributes:
        EMOTION_DICT (dict): A dictionary mapping emotion labels to their corresponding names.
    """

    EMOTION_DICT = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    }

    def __init__(
        self,
        accelerator: str = "cuda" if torch.cuda.is_available() else "cpu",
        backend_option: int = 0 if torch.cuda.is_available() else 1,
        model_name: str = "vgg.pt",
    ):
        """
        Initializes the Detector object.

        Args:
            use_cuda (bool, optional): Whether to use CUDA for faster processing if a GPU is available. Default is cuda if CUDA is available, otherwise cpu.
            backend_option (int, optional): Backend option for OpenCV's DNN module. Default is 0 if CUDA is available, otherwise 1.
        """
        self.logger = self.setup_logger()
        self.device = self.setup_device(accelerator, backend_option)
        self.emotion_model = self.load_trained_model(f"train/models/{model_name}")

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def setup_device(self, accelerator, backend_option):
        backend_target_pairs = [
            [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
        ]

        if accelerator == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("CUDA is not available. Using CPU instead.")

        return torch.device("cpu")

    def load_trained_model(self, model_path: str) -> nn.Module:
        """
        Loads a pre-trained emotion recognition model from the specified path.

        Args:
            model_path (str): The path to the pre-trained model file.

        Returns:
            Face_Emotion_CNN: The loaded pre-trained model.
        """
        model = resnet.ResNet18()
        model.load_state_dict(
            torch.load(model_path, map_location=self.device)["model_state_dict"]
        )
        model.to(self.device)
        model.eval()
        return model

    def recognize_emotion(self, face: np.ndarray) -> str:
        try:
            transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.TenCrop(40),
                    transforms.Lambda(
                        lambda crops: torch.stack(
                            [transforms.ToTensor()(crop) for crop in crops]
                        )
                    ),
                    transforms.Lambda(
                        lambda tensors: torch.stack(
                            [
                                transforms.Normalize(mean=(0,), std=(255,))(t)
                                for t in tensors
                            ]
                        )
                    ),
                ]
            )
            resize_frame = cv2.resize(face, (48, 48))
            gray_frame = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
            inputs = Image.fromarray(gray_frame)
            inputs = transform(inputs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

                # forward pass
                outputs = self.emotion_model(inputs)

                # combine results across the crops
                outputs = outputs.view(bs, ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops

                _, preds = torch.max(outputs.data, 1)
                preds = preds.cpu().numpy()[0]

                print(preds)

                print("-" * 15)
                predicted_emotion_label = EmotionDetectorMediapipe.EMOTION_DICT[preds]

            return predicted_emotion_label
        except cv2.error as e:
            self.logger.error("No emotion detected")

    def process_image(self, img_name: str) -> None:
        """
        Processes and displays an image with emotion recognition.

        Args:
            img_name (str): The path to the input image file.
        """
        self.img = cv2.imread(img_name)
        self.height, self.width = self.img.shape[:2]
        self.process_frame()
        cv2.imshow("Output", self.img)
        cv2.waitKey(0)

    def process_video(self, video_path: str, display_window: bool = True) -> None:
        """
        Processes a video file, performing emotion recognition on each frame.

        Args:
            video_path (str): The path to the input video file.
                if video_path == "realsense", then the video is captured from the realsense camera.
                if video_path == 0, then the video is captured from the webcam.
                or else, the video is captured from the specified path.
            display_window (bool, optional): Whether to display the processed image using cv2.imshow.
                Defaults to True.
        """
        if video_path == "realsense":
            video_path = "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"

        self.logger.info("Video path: %s", video_path)

        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Error opening video stream or file")
            return

        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as face_detection:
            success, self.img = cap.read()
            self.height, self.width = self.img.shape[:2]

            fps = FPS().start()

            while success:
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                self.img.flags.writeable = False
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                results = face_detection.process(self.img)

                # Draw the face detection annotations on the image.
                self.img.flags.writeable = True
                self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(self.img, detection)
                        face = detection.location_data.relative_bounding_box

                        try:
                            self.process_frame(face)
                            if display_window:
                                cv2.imshow("Output", self.img)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                            fps.update()
                            success, self.img = cap.read()
                        except KeyboardInterrupt:
                            break

            fps.stop()
            self.logger.info("Elapsed time: %.2f", fps.elapsed())
            self.logger.info("Approx. FPS: %.2f", fps.fps())

            cap.release()
            cv2.destroyAllWindows()

    def process_frame(self, face) -> None:
        """
        Processes the current frame, detects faces, and recognizes emotions.
        """
        h, w, _ = self.img.shape
        x_min, y_min = int(face.xmin * w), int(face.ymin * h)
        x_max, y_max = int(face.width * w + x_min), int(face.height * h + y_min)

        face = self.img[y_min:y_max, x_min:x_max]

        emotion = self.recognize_emotion(face)

        cv2.putText(
            self.img,
            emotion,
            (x_min + 5, y_min - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )