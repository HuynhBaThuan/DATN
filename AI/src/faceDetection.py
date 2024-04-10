import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
#     image_rgb = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)

# # Phát hiện khuôn mặt trong ảnh
#     faces = detector.detect_faces(image_rgb)
    for (x, y, w, h) in faces:
        # cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.rectangle(vid, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
    return faces

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "My Face Detection Project", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    # cv2.imshow('Video', cv2.resize(video_frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()