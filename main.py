from emotion_detector import EmotionDetector

detector = EmotionDetector(accelerator="cuda", backend_option=1)

detector.process_video(0, display_window=True)

#detector.process_image("happy.webp")