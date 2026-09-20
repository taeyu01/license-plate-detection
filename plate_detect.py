from ultralytics import YOLO
import cv2
import easyocr
import re

model = YOLO("best.pt")
reader = easyocr.Reader(['ko', 'en'], gpu=False)

results = model("test_car.jpg", conf=0.1)
result = results[0]

image = cv2.imread("test_car.jpg")

if len(result.boxes) > 0:
    box = result.boxes[0]

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = float(box.conf[0])

    print("YOLO confidence:", confidence)

    # 1. 번호판 Crop
    plate = image[y1:y2, x1:x2]

    # 2. Gray
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # 3. Blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 4. Adaptive Threshold
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 5. OCR
    ocr_result = reader.readtext(binary)

    print("OCR 결과:", ocr_result)

    if len(ocr_result) > 0:
        text = ocr_result[0][1]
        text = text.replace(" ", "")
        text = re.sub(r'[^0-9가-힣]', '', text)

        print("번호판:", text)

    cv2.imshow("Plate", plate)
    cv2.imshow("OCR Image", binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("번호판 탐지 실패")