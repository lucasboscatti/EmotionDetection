from emotion_detector_mediapipe import EmotionDetectorMediapipe

detector = EmotionDetectorMediapipe(
    accelerator="cuda", backend_option=1, model_name="best_checkpoint.tar"
)

detector.process_video(0, display_window=True)
