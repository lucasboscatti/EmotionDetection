import logging
import os

import gdown
import torch
from flask import Flask, Response, redirect, render_template, request, url_for

from emotion_detector import EmotionDetector

CUDA = torch.cuda.is_available()
app = Flask(__name__, template_folder="template")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class DeviceWindows:
    def __init__(
        self,
        model_name="resnet18.onnx",
        model_option="onnx",
        backend_option=0,
        providers=1,
    ):
        self.model_name = model_name
        self.model_option = model_option
        self.backend_option = backend_option
        self.providers = providers


class DeviceJetsonNano:
    def __init__(
        self,
        model_name="resnet18_engine.trt",
        model_option="tensorrt",
        backend_option=1,
        providers=1,
    ):
        self.model_name = model_name
        self.model_option = model_option
        self.backend_option = backend_option
        self.providers = providers


class DeviceLinux:
    def __init__(
        self,
        model_name="resnet18.onnx",
        model_option="onnx",
        backend_option=0,
        providers=1,
    ):
        self.model_name = model_name
        self.model_option = model_option
        self.backend_option = backend_option
        self.providers = providers


class DeviceCustom:
    def __init__(
        self,
        model_name,
        model_option,
        backend_option,
        providers,
    ):
        self.model_name = model_name
        self.model_option = model_option
        self.backend_option = backend_option
        self.providers = providers


@app.route("/", methods=["GET", "POST"])
def index():
    """Video streaming home page."""
    if request.method == "POST":
        selected_device = request.form["device_option"]
        video_option = request.form["video_option"]
        mode_option = request.form["mode_option"]

        if selected_device == "windows":
            device = DeviceWindows()
        elif selected_device == "jetson_nano":
            device = DeviceJetsonNano()
        elif selected_device == "linux":
            device = DeviceLinux()
        elif selected_device == "custom":
            return redirect(
                url_for(
                    "select_params", video_option=video_option, mode_option=mode_option
                )
            )
        if mode_option == "game":
            return redirect(
                url_for(
                    "game_mode",
                    model_name=device.model_name,
                    model_option=device.model_option,
                    backend_option=device.backend_option,
                    providers=device.providers,
                    video_option=video_option,
                    mode_option=mode_option,
                )
            )
        elif mode_option == "normal":
            return redirect(
                url_for(
                    "normal_mode",
                    model_name=device.model_name,
                    model_option=device.model_option,
                    backend_option=device.backend_option,
                    providers=device.providers,
                    video_option=video_option,
                    mode_option=mode_option,
                )
            )
    return render_template("index.html")


@app.route("/select_params/<video_option>/<mode_option>", methods=["GET", "POST"])
def select_params(video_option, mode_option):
    if request.method == "POST":
        model_option = request.form["model_option"]
        backend_option = request.form["backend_option"]
        providers = request.form["providers"]

        model_names = {
            "pytorch": "best_checkpoint.tar",
            "onnx": "resnet18.onnx",
            "engine": "resnet18_engine.trt",
        }

        return redirect(
            url_for(
                "game_mode",
                model_name=model_names[model_option],
                model_option=model_option,
                backend_option=backend_option,
                providers=providers,
                video_option=video_option,
                mode_option=mode_option,
            )
        )
    return render_template("select_params.html")


@app.route("/game_mode", methods=["GET", "POST"])
def game_mode():
    args = request.args.to_dict()
    if request.method == "POST":
        if request.form["submit_button"] == "Calcular o vencedor":
            detector.game_mode_result()
            return redirect(url_for("index"))
    return render_template("game_mode.html", **args)


@app.route("/normal_mode", methods=["GET", "POST"])
def normal_mode():
    args = request.args.to_dict()
    return render_template("normal_mode.html", **args)


@app.route("/video_feed", methods=["GET", "POST"])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    model_name = request.args.get("model_name")
    model_option = request.args.get("model_option")
    backend_option = int(request.args.get("backend_option"))
    providers = int(request.args.get("providers"))
    video_option = request.args.get("video_option")

    modes = {"normal": False, "game": True}
    mode_option = modes[request.args.get("mode_option")]

    detector.model_name = model_name
    detector.model_option = model_option
    detector.backend_option = backend_option
    detector.providers = providers
    detector.video_option = video_option
    detector.game_mode = mode_option

    return Response(
        detector.start_inference(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def download_model_files():
    model_files = [
        (
            "ready_to_use_models/emotion_model/resnet18.onnx",
            "1XmW8hCprZ6ZHGlU493Hp7jTl9iLXXI0F",
        ),
        (
            "ready_to_use_models/emotion_model/resnet18_engine.trt",
            "1oMQj-x9p7XwYV-nRuLiw_mrFv2MEuUH1",
        ),
        (
            "ready_to_use_models/emotion_model/best_checkpoint.tar",
            "1HSdcnLKckmVAhB5yaqYnI-wmonbN7-BX",
        ),
        (
            "ready_to_use_models/face_model/",
            "https://drive.google.com/drive/folders/1Su2UpXX2Ub0iL3IYQwMgNfQfkKMe7pNr",
        ),
    ]
    for file_path, id in model_files:
        if not os.path.exists(file_path):
            logging.info(f"{file_path} not found. Downloading...")
            try:
                gdown.download(id=id, output=file_path)
            except Exception as e:
                gdown.download_folder(id=id, output="ready_to_use_models/")


if __name__ == "__main__":
    download_model_files()
    detector = EmotionDetector()
    app.run("0.0.0.0", 5000)
