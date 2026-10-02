import cv2
from ultralytics import YOLO

# 새로 학습한 모델
model = YOLO("models/best_plate_yolo26.pt")

cap = cv2.VideoCapture("plate_test.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO 번호판 검출
    results = model(
        frame,
        imgsz=512,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    for box in result.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        print(
            "CLASS:", cls,
            "CONF:", round(conf, 3),
            "BOX:", x1, y1, x2, y2
        )

        # 번호판 영역만 자르기
        plate = frame[y1:y2, x1:x2]

        # 원본 영상에 Bounding Box 표시
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        if plate.size > 0:
            cv2.imshow("Plate Crop", plate)

    cv2.imshow("YOLO Detection", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()