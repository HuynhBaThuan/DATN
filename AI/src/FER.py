import cv2
from deepface import DeepFace

def main():
    # Load pre-trained model
    model = DeepFace.build_model('FacialExpressionModel')

    # Load image
    image = cv2.imread('happy.jpg')

    # Detect faces
    faces = DeepFace.detectFace(image, detector_backend='opencv')

    # For each face, predict emotion
    for face in faces:
        emotions = DeepFace.analyze(image, actions=['emotion'], models={'emotion': model})
        emotion = max(emotions['emotion'], key=emotions['emotion'].get)
        print("Emotion:", emotion)

        # Draw rectangle around face
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display emotion label
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display result
    cv2.imshow('Emotion Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
