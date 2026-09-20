from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

results = model("test_car.jpg", conf=0.1)
result = results[0]

print("탐지 개수:", len(result.boxes))

image = cv2.imread("test_car.jpg")

if len(result.boxes) > 0:
    box = result.boxes[0]

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = float(box.conf[0])

    print("confidence:", confidence)

    plate = image[y1:y2, x1:x2]

    cv2.imshow("Plate", plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("번호판 탐지 실패")