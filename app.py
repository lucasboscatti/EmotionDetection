import torch
from flask import Flask, Response, redirect, render_template, request, url_for

from emotion_detector import EmotionDetector

CUDA = torch.cuda.is_available()
app = Flask(__name__, template_folder="template")


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
        model_name="resnet18.engine",
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
        model_name="resnet18.engine",
        model_option="tensorrt",
        backend_option=3,
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
        return redirect(
            url_for(
                "video_feed",
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
            "engine": "resnet18.engine",
        }

        return redirect(
            url_for(
                "video_feed",
                model_name=model_names[model_option],
                model_option=model_option,
                backend_option=backend_option,
                providers=providers,
                video_option=video_option,
                mode_option=mode_option,
            )
        )
    return render_template("select_params.html")


@app.route("/video_feed", methods=["GET", "POST"])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    args = request.args.to_dict()

    modes = {"normal": False, "game": True}

    detector = EmotionDetector(
        model_name=args["model_name"],
        model_option=args["model_option"],
        backend_option=int(args["backend_option"]),
        providers=int(args["providers"]),
        video_option=args["video_option"],
        game_mode=modes[args["mode_option"]],
    )
    return Response(
        detector.start_inference(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )