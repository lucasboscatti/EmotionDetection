import logging
import sys

import cv2
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from imutils.video import FPS
from PIL import Image

sys.path.append("../")


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def calculate_winner(bbox_predictions):
    classes_weights = {
        0: -1,
        1: 1,
        2: 0,
        None: 0,
    }

    competitor_1 = bbox_predictions["bbox_left"]
    competitor_2 = bbox_predictions["bbox_right"]

    def calculate_score(competitor):
        score = [classes_weights[num] for num in competitor]
        print(score)
        return sum(score)

    score_1 = calculate_score(competitor_1)
    score_2 = calculate_score(competitor_2)

    if score_1 > score_2:
        print("Winner: left")
        return "left" # 0

    print("Winner: right")
    return "right" # 1


class EmotionDetector:
    """
    A class for detecting and recognizing emotions in images or video frames.

    Args:
        use_cuda (bool, optional): Whether to use CUDA for faster processing if a GPU is available. Default is True.
        backend_option (int, optional): Backend option for OpenCV's DNN module. Default is 1.

    Attributes:
        EMOTION_DICT (dict): A dictionary mapping emotion labels to their corresponding names.
    """

    EMOTION_DICT = {0: "BAD", 1: "GOOD", 2: "NEUTRAL"}

    def __init__(
        self,
        accelerator: str = "cuda" if torch.cuda.is_available() else "cpu",
        backend_option: int = 0 if torch.cuda.is_available() else 1,
        model_name: str = "resnet18.onnx",
        providers=1,
        num_faces=None,
    ):
        """
        Initializes the Detector object.

        Args:
            use_cuda (bool, optional): Whether to use CUDA for faster processing if a GPU is available. Default is cuda if CUDA is available, otherwise cpu.
            backend_option (int, optional): Backend option for OpenCV's DNN module. Default is 0 if CUDA is available, otherwise 1.
        """
        self.logger = self.setup_logger()
        self.face_model = self.load_face_model(backend_option)
        self.device = self.setup_device(accelerator)
        self.emotion_model = self.load_trained_model(
            f"train/models/{model_name}", providers
        )
        self.num_faces = num_faces
        self.bbox_predictions = {
            "bbox_left": [],
            "bbox_right": [],
        }

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def load_face_model(self, backend_option: int = 1):
        face_model = cv2.dnn.readNetFromCaffe(
            "train/models/face_detector/res10_300x300_ssd_iter_140000.prototxt",
            "train/models/face_detector/res10_300x300_ssd_iter_140000.caffemodel",
        )
        backend_target_pairs = [
            [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
        ]

        face_model.setPreferableBackend(backend_target_pairs[backend_option][0])
        face_model.setPreferableTarget(backend_target_pairs[backend_option][1])
        return face_model

    def setup_device(self, accelerator):
        if accelerator == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("CUDA is not available. Using CPU instead.")

        return torch.device("cpu")

    def load_trained_model(
        self, model_name: str, providers
    ) -> onnxruntime.InferenceSession:
        """
        Loads a pre-trained emotion recognition model from the specified path.

        Args:
            model_path (str): The path to the pre-trained model file.
            providers (list, optional): The list of providers to use for inference. Defaults to None.

        Returns:
            Face_Emotion_CNN: The loaded pre-trained model.
        """
        providers_options = {
            1: ["CPUExecutionProvider"],
            2: ["CUDAExecutionProvider"],
            3: ["TensorrtExecutionProvider"],
        }

        return onnxruntime.InferenceSession(
            model_name, providers=["CPUExecutionProvider"]
        )

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
            inputs = Image.fromarray(resize_frame)
            inputs = transform(inputs).unsqueeze(0).to(self.device)

            with torch.no_grad():
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

                inputs = {self.emotion_model.get_inputs()[0].name: to_numpy(inputs)}
                outputs = self.emotion_model.run(
                    [self.emotion_model.get_outputs()[0].name], inputs
                )
                outputs = torch.from_numpy(outputs[0])

                # combine results across the crops
                outputs = outputs.view(bs, ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops

                _, preds = torch.max(outputs.data, 1)
                preds = preds.cpu().numpy()[0]

            return preds
        except cv2.error as e:
            self.logger.error("No emotion detected: ", e)

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

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("Error opening video stream or file")
            return

        success, self.img = cap.read()
        self.height, self.width = self.img.shape[:2]

        fps = FPS().start()

        while success:
            try:
                self.process_frame()
                if display_window:
                    cv2.imshow("Output", self.img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    if self.num_faces == 2:
                        flag = calculate_winner(self.bbox_predictions)
                        self.logger.info("O robô deve andar para: %s", flag)
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

    def process_frame(self) -> None:
        """
        Processes the current frame, detects faces, and recognizes emotions.
        """
        blob = cv2.dnn.blobFromImage(
            cv2.resize(self.img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        self.face_model.setInput(blob)
        predictions = self.face_model.forward()

        if self.num_faces == 2:
            try:
                prediction_1 = predictions[0, 0, 0, 2]
                prediction_2 = predictions[0, 0, 1, 2]

                if prediction_1 > 0.5 and prediction_2 > 0.5:
                    bbox_1 = predictions[0, 0, 0, 3:7] * np.array(
                        [self.width, self.height, self.width, self.height]
                    )
                    bbox_2 = predictions[0, 0, 1, 3:7] * np.array(
                        [self.width, self.height, self.width, self.height]
                    )
                    (x_min_1, y_min_1, x_max_1, y_max_1) = bbox_1.astype("int")
                    (x_min_2, y_min_2, x_max_2, y_max_2) = bbox_2.astype("int")
                    cv2.rectangle(
                        self.img, (x_min_1, y_min_1), (x_max_1, y_max_1), (0, 0, 255), 2
                    )
                    cv2.rectangle(
                        self.img, (x_min_2, y_min_2), (x_max_2, y_max_2), (0, 0, 255), 2
                    )

                    face_1 = self.img[y_min_1:y_max_1, x_min_1:x_max_1]
                    face_2 = self.img[y_min_2:y_max_2, x_min_2:x_max_2]

                    emotion_1 = self.recognize_emotion(face_1)
                    emotion_2 = self.recognize_emotion(face_2)

                    if x_min_1 < x_min_2:
                        self.bbox_predictions["bbox_left"].append(emotion_1)
                        self.bbox_predictions["bbox_right"].append(emotion_2)
                        cv2.putText(
                            self.img,
                            EmotionDetector.EMOTION_DICT[emotion_1],
                            (x_min_1 + 5, y_min_1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            self.img,
                            EmotionDetector.EMOTION_DICT[emotion_2],
                            (x_min_2 + 5, y_min_2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        self.bbox_predictions["bbox_left"].append(emotion_2)
                        self.bbox_predictions["bbox_right"].append(emotion_1)
                        cv2.putText(
                            self.img,
                            EmotionDetector.EMOTION_DICT[emotion_2],
                            (x_min_2 + 5, y_min_2 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            self.img,
                            EmotionDetector.EMOTION_DICT[emotion_1],
                            (x_min_1 + 5, y_min_1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                else:
                    self.logger.error("Only one face detected.")
            except:
                self.logger.error("Only one face detected.")

        else:
            for i in range(predictions.shape[2]):
                if predictions[0, 0, i, 2] > 0.5:
                    bbox = predictions[0, 0, i, 3:7] * np.array(
                        [self.width, self.height, self.width, self.height]
                    )
                    (x_min, y_min, x_max, y_max) = bbox.astype("int")
                    cv2.rectangle(
                        self.img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2
                    )

                    face = self.img[y_min:y_max, x_min:x_max]

                    emotion = self.recognize_emotion(face)

                    cv2.putText(
                        self.img,
                        EmotionDetector.EMOTION_DICT[emotion],
                        (x_min + 5, y_min - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
