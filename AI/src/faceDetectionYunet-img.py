import os
import numpy as np
import cv2

def main():
    directory = os.path.dirname(__file__)
    image_files = [os.path.join(directory, 'image.jpg'), os.path.join(directory, 'image2.jpg')]  # Danh sách các đường dẫn đến các tệp hình ảnh

    weights = os.path.join(directory, "yunet_n_320_320.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

    for image_file in image_files:
        image = cv2.imread(image_file)  # Đọc hình ảnh từ tệp
        if image is None:
            print(f"Cannot read image file: {image_file}")
            continue
        
        channels = 1 if len(image.shape) == 2 else image.shape[2]

        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)
                
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)
        
        cv2.imshow("Image", image)  # Hiển thị hình ảnh
        cv2.waitKey(0)  # Chờ một phím nhấn để chuyển sang hình ảnh tiếp theo hoặc thoát

    cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ hiển thị

if __name__ == '__main__':
    main()