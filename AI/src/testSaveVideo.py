import os
import numpy as np
import cv2

def main():
    directory = os.path.dirname(__file__)
    capture = cv2.VideoCapture(0) # Camera
    if not capture.isOpened():
        exit()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # Codec để ghi video
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480)) # Kích thước khung hình (chiều rộng, chiều cao)
    
    weights = os.path.join(directory, "yunet_n_320_320.onnx")
    face_detector = cv2.FaceDetectorYN_create(weights, "", (0, 0))

    while True:
        result, image = capture.read() # Đọc khung hình từ camera
        if not result:
            break
        
        # Ở đây bạn có thể thực hiện xử lý hình ảnh nếu cần

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

            # ランドマーク（右目、左目、鼻、右口角、左口角）
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            for landmark in landmarks:
                radius = 5
                thickness = -1
                cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)
                
            # 信頼度
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)
        
        out.write(image) # Ghi khung hình vào video
        
        cv2.imshow("Camera", image) # Hiển thị khung hình từ camera
        key = cv2.waitKey(1) # Chờ một phím nhấn
        if key == ord('q'):
            break
    
    capture.release() # Giải phóng tài nguyên của camera
    out.release() # Giải phóng tài nguyên của VideoWriter
    cv2.destroyAllWindows() # Đóng tất cả các cửa sổ hiển thị

if __name__ == '__main__':
    main()
