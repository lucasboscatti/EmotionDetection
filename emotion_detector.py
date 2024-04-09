import logging
import mimetypes
import sys

import cv2
import numpy as np
import onnxruntime
import torch
import torchvision.transforms as transforms
from imutils.video import FPS
from PIL import Image

from train.models import resnet
from utils.utils import calculate_winner, to_numpy

logger = logging.getLogger()
logger.setLevel(logging.INFO)

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorRT as trt

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    batch = 1
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []
except ImportError:
    TensorRT = False
    logger.info("TensorRTInference not found.")

sys.path.append("../")

CUDA = torch.cuda.is_available()


class EmotionDetector:
    """
    A class for detecting and recognizing emotions in images or video frames.

    Args:
        use_cuda (bool, optional): Whether to use CUDA for faster processing if a GPU is available. Default is True.
        backend_option (int, optional): Backend option for OpenCV's DNN module. Default is 1.

    Attributes:
        EMOTIONS (dict): A dictionary mapping emotion labels to their corresponding names.
    """

    EMOTIONS = {0: "BAD", 1: "GOOD", 2: "NEUTRAL"}

    def __init__(
        self,
        model_name: str,
        model_option: str,
        backend_option: int,
        providers: int,
        fp16: bool = False,
        game_mode: bool = False,
    ):
        """
        Initializes the Detector object.

        Args:
            use_cuda (bool, optional): Whether to use CUDA for faster processing if a GPU is available. Default is cuda if CUDA is available, otherwise cpu.
            backend_option (int, optional): Backend option for OpenCV's DNN module. Default is 0 if CUDA is available, otherwise 1.
        """

        self.device = torch.device("cuda" if CUDA else "cpu")
        self.model_option = model_option
        self.game_mode = game_mode
        self.bbox_predictions = {
            "bbox_left": [],
            "bbox_right": [],
        }

        self.face_model = self.load_face_model(backend_option)
        self.emotion_model = self.load_trained_model(
            model_name,
            providers=providers,
            fp16=fp16,
        )

    def load_face_model(self, backend_option: int) -> cv2.dnn_Net:
        """
        Load the face model for face detection.

        Parameters:
            backend_option (int): Backend option for OpenCV's DNN module.

        Returns:
            cv2.dnn_Net: The loaded face model for face detection.
        """
        face_model = cv2.dnn.readNetFromCaffe(
            "ready_to_use_models/face_model/res10_300x300_ssd_iter_140000.prototxt",
            "ready_to_use_models/face_model/res10_300x300_ssd_iter_140000.caffemodel",
        )

        backend_target_pairs = [
            [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
            [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
        ]

        face_model.setPreferableBackend(backend_target_pairs[backend_option][0])
        face_model.setPreferableTarget(backend_target_pairs[backend_option][1])
        return face_model

    def load_trained_model(self, model_name: str, providers: int, fp16: bool):
        """
        Load a trained model.

        Args:
            model_name (str): The path to the model file or to the checkpoint file.
            model_option (str): The option for loading the model.

        Returns:
            model: The loaded model.
        """
        model_path = f"ready_to_use_models/emotion_model/{model_name}"

        if self.model_option == "pytorch":
            model = resnet.ResNet18()
            model.load_state_dict(
                torch.load(model_path, map_location=self.device)["model_state_dict"]
            )
            model.to(self.device)
            model.eval()

        elif self.model_option == "onnx":
            providers_options = {
                1: ["CPUExecutionProvider"],
                2: ["CUDAExecutionProvider"],
                3: ["TensorrtExecutionProvider"],
            }
            model = onnxruntime.InferenceSession(
                model_path, providers=providers_options[providers]
            )

        elif self.model_option == "tensorrt":
            if not TensorRT:
                logger.error("TensorRTInference not found.")
                return

            with open(model_path, "rb") as f:
                serialized_engine = f.read()

            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(serialized_engine)

            # create buffer
            for binding in engine:
                size = trt.volume(engine.get_tensor_shape(binding)) * batch
                host_mem = cuda.pagelocked_empty(shape=[size], dtype=np.float32)
                cuda_mem = cuda.mem_alloc(host_mem.nbytes)

                bindings.append(int(cuda_mem))
                if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    host_inputs.append(host_mem)
                    cuda_inputs.append(cuda_mem)
                else:
                    host_outputs.append(host_mem)
                    cuda_outputs.append(cuda_mem)

            return engine

        return model

    def start_inference(self, file, display_window=True):
        if file in ["0", "realsense"]:
            self.process_video(file, display_window=display_window)

        mimetypes.init()
        mimestart = mimetypes.guess_type(file)[0]

        if mimestart != None:
            mimestart = mimestart.split("/")[0]

            if mimestart == "video":
                self.process_video(file, display_window=display_window)

            elif mimestart == "image":
                self.process_image(file)

        logger.error("Invalid file type.")

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocesses an image.

        Args:
            img_name (str): The path to the input image file.
        """
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
        try:
            inputs = Image.fromarray(image).resize((48, 48))
            return transform(inputs).unsqueeze(0).to(self.device)
        except ValueError as e:
            logger.error("Error preprocessing image: ", e)

    def recognize_emotion(self, face: np.ndarray) -> str:
        inputs = self.preprocess_image(face)
        if inputs is not None:
            with torch.no_grad():
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

                if self.model_option in ["pytorch", "tensorrt"]:
                    outputs = self.emotion_model(inputs)

                elif self.model_option == "onnx":
                    inputs = {self.emotion_model.get_inputs()[0].name: to_numpy(inputs)}
                    outputs = self.emotion_model.run(
                        [self.emotion_model.get_outputs()[0].name], inputs
                    )
                    outputs = torch.from_numpy(outputs[0])

                elif self.model_option == "tensorrt":
                    stream = cuda.Stream()
                    context = self.emotion_model.create_execution_context()
                    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
                    context.execute_v2(bindings)
                    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
                    stream.synchronize()
                    output = host_outputs[0]
                    return np.argmax(output)

                outputs = outputs.view(bs, ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops
                _, preds = torch.max(outputs.data, 1)
                preds = preds.cpu().numpy()[0]

            return preds

        logger.error("No face detected.")

    def process_image(self, img_name: str) -> None:
        """
        Processes and displays an image with emotion recognition.

        Args:
            img_name (str): The path to the input image file.
        """
        image = cv2.imread(img_name)
        self.process_frame(image)
        cv2.imshow("Output", image)
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

        video_path = (
            "v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"
            if video_path == "realsense"
            else 0
        )

        logger.info("Video path: %s", video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Error opening video stream or file")

        success, image = cap.read()

        fps = FPS().start()

        while success:
            try:
                self.process_frame(image)
                cv2.imshow("Output", image) if display_window else None

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    if self.game_mode:
                        flag = calculate_winner(self.bbox_predictions)
                        logger.info("O robô deve andar para: %s", flag)
                    break
                fps.update()
                success, image = cap.read()
            except KeyboardInterrupt:
                break

        fps.stop()
        logger.info("Elapsed time: %.2f", fps.elapsed())
        logger.info("Approx. FPS: %.2f", fps.fps())

        cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, image: np.ndarray) -> None:
        """
        Processes the current frame, detects faces, and recognizes emotions.
        """
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        self.face_model.setInput(blob)
        predictions = self.face_model.forward()

        if self.game_mode:
            self.game_mode_process(predictions, image)

        self.default_mode_process(predictions, image)

    def game_mode_process(self, predictions: np.ndarray, image: np.ndarray) -> None:
        """
        Processes and displays an image with emotion recognition in game mode.

        Args:
            predictions (np.ndarray): The predictions from the face model.
            image (np.ndarray): The input image.
        """
        height, width = image.shape[:2]
        prediction_1 = predictions[0, 0, 0, 2]
        prediction_2 = predictions[0, 0, 1, 2]

        if prediction_1 > 0.5 and prediction_2 > 0.5:
            bbox_1 = predictions[0, 0, 0, 3:7] * np.array(
                [width, height, width, height]
            )
            bbox_2 = predictions[0, 0, 1, 3:7] * np.array(
                [width, height, width, height]
            )
            (x_min_1, y_min_1, x_max_1, y_max_1) = bbox_1.astype("int")
            (x_min_2, y_min_2, x_max_2, y_max_2) = bbox_2.astype("int")
            cv2.rectangle(image, (x_min_1, y_min_1), (x_max_1, y_max_1), (0, 0, 255), 2)
            cv2.rectangle(image, (x_min_2, y_min_2), (x_max_2, y_max_2), (0, 0, 255), 2)

            face_1 = image[y_min_1:y_max_1, x_min_1:x_max_1]
            face_2 = image[y_min_2:y_max_2, x_min_2:x_max_2]

            emotion_1 = self.recognize_emotion(face_1)
            emotion_2 = self.recognize_emotion(face_2)

            if emotion_1 is not None and emotion_2 is not None:
                if x_min_1 < x_min_2:
                    self.bbox_predictions["bbox_left"].append(emotion_1)
                    self.bbox_predictions["bbox_right"].append(emotion_2)
                    cv2.putText(
                        image,
                        EmotionDetector.EMOTIONS[emotion_1],
                        (x_min_1 + 5, y_min_1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        EmotionDetector.EMOTIONS[emotion_2],
                        (x_min_2 + 5, y_min_2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    return
                else:
                    self.bbox_predictions["bbox_left"].append(emotion_2)
                    self.bbox_predictions["bbox_right"].append(emotion_1)
                    cv2.putText(
                        image,
                        EmotionDetector.EMOTIONS[emotion_2],
                        (x_min_2 + 5, y_min_2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        EmotionDetector.EMOTIONS[emotion_1],
                        (x_min_1 + 5, y_min_1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
        else:
            logger.warning("No faces detected OR only one face detected.")

    def default_mode_process(self, predictions, image):
        height, width = image.shape[:2]
        for i in range(predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array(
                    [width, height, width, height]
                )
                (x_min, y_min, x_max, y_max) = bbox.astype("int")
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                face = image[y_min:y_max, x_min:x_max]

                emotion = self.recognize_emotion(face)

                if emotion:
                    cv2.putText(
                        image,
                        EmotionDetector.EMOTIONS[emotion],
                        (x_min + 5, y_min - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
