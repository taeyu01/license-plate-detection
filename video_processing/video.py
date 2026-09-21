import cv2
import re
import easyocr
from ultralytics import YOLO
from collections import Counter

model = YOLO("best.pt")
reader = easyocr.Reader(["ko", "en"], gpu=False)

cap = cv2.VideoCapture("plate_test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

counter = Counter()
final_plate = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.1, imgsz=1280, verbose=False)
    result = results[0]

    if len(result.boxes) > 0:
        best_box = max(result.boxes, key=lambda box: float(box.conf[0]))

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        conf = float(best_box.conf[0])

        plate = frame[y1:y2, x1:x2].copy()

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        ocr_result = reader.readtext(binary)

        if len(ocr_result) > 0:
            text = ocr_result[0][1]
            ocr_conf = ocr_result[0][2]

            text = text.replace(" ", "")
            text = re.sub(r"[^0-9가-힣]", "", text)

            if ocr_conf >= 0.5:
                if re.fullmatch(r"[0-9]{2,3}[가-힣][0-9]{4}", text):
                    counter[text] += 1
                    print(text, "→", counter[text], "회")

                    if counter[text] >= 3:
                        final_plate = text
                        print("최종 번호판:", final_plate)
                        break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Plate", plate)
        cv2.imshow("Binary", binary)

    display_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("License Plate Detection", display_frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()