import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from imutils.video import FPS
from PIL import Image

from train.models import resnet

EMOTION_DICT = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet.ResNet18()
model.load_state_dict(
    torch.load("train/models/best_checkpoint.tar", map_location=device)[
        "model_state_dict"
    ]
)
model.to(device)
model.eval()


def recognize_emotion(face: np.ndarray) -> str:
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

        inputs = Image.fromarray(face)
        inputs = transform(inputs).unsqueeze(0).to(device)

        with torch.no_grad():
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            # forward pass
            outputs = model(inputs)

            # combine results across the crops
            outputs = outputs.view(bs, ncrops, -1)
            outputs = torch.sum(outputs, dim=1) / ncrops

            _, preds = torch.max(outputs.data, 1)
            preds = preds.cpu().numpy()[0]

            print(preds)

            print("-" * 15)
            predicted_emotion_label = EMOTION_DICT[preds]

        return predicted_emotion_label
    except cv2.error as e:
        print("No emotion detected")


# Load the cascade
face_cascade = cv2.CascadeClassifier(
    "opencv_models/cuda/haarcascade_frontalface_default.xml"
)

# To capture video from webcam.
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    fps = FPS().start()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray_frame = gray[y : y + h, x : x + w]
        face = cv2.resize(roi_gray_frame, (48, 48))

        emotion = recognize_emotion(roi_gray_frame)

        cv2.putText(
            img,
            emotion,
            (int(x) + 5, int(y) - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Display
    cv2.imshow("img", img)
    fps.update()
    # Stop if escape key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release the VideoCapture object
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
