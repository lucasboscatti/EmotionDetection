import logging
import mimetypes
import sys

import cv2
import numpy as np

# import onnxruntime
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
    import tensorrt as trt

    TensorRT = True

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    batch = 1
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    bindings = []

except ImportError as e:
    TensorRT = False
    logger.info("TensorRTInference not found.")

sys.path.append("../")

CUDA = torch.cuda.is_available()

device = torch.device("cuda" if CUDA else "cpu")


def prepare_engine(model_name):
    if not TensorRT:
        logger.error("TensorRTInference not found.")
        return

    model_path = f"ready_to_use_models/emotion_model/{model_name}"
    with open(model_path, "rb") as f:
        serialized_engine = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    # create buffer
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch
        host_mem = cuda.pagelocked_empty(shape=[size], dtype=np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)

    return engine


def load_trained_model(model_name: str, providers: int, fp16: bool):
    """
    Load a trained model.

    Args:
        model_name (str): The path to the model file or to the checkpoint file.
        model_option (str): The option for loading the model.

    Returns:
        model: The loaded model.
    """
    model_path = f"ready_to_use_models/emotion_model/{model_name}"

    if model_option == "pytorch":
        model = resnet.ResNet18()
        model.load_state_dict(
            torch.load(model_path, map_location=device)["model_state_dict"]
        )
        model.to(device)
        model.eval()

        return model


def preprocess_image(image: np.ndarray):
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
                    [transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors]
                )
            ),
        ]
    )
    try:
        inputs = Image.fromarray(image).resize((48, 48))
        return transform(inputs).unsqueeze(0).to(device)
    except ValueError as e:
        logger.error("Error preprocessing image: ", e)


def recognize_emotion(face: np.ndarray) -> str:
    inputs = preprocess_image(face)
    if inputs is not None:
        with torch.no_grad():
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            if model_option == "pytorch":
                outputs = emotion_model(inputs)

            elif model_option == "tensorrt":
                stream = cuda.Stream()
                context = engine.create_execution_context()

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


model_name = "resnet18.onnx"
model_option = "onnx"
backend_option = 2 if CUDA else 1
providers = 2 if CUDA else 1
fp16 = False
game_mode = False

if model_option == "tensorrt":
    engine = prepare_engine(model_name)
else:
    emotion_model = load_trained_model(
        model_name,
        providers=providers,
        fp16=fp16,
    )

img = cv2.imread("happy.jpeg")

result = recognize_emotion(img)
print(result)

