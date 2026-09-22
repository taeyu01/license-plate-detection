import cv2
import re
import easyocr
from ultralytics import YOLO
from collections import Counter
import mysql.connector

model = YOLO("best.pt")
reader = easyocr.Reader(["ko", "en"], gpu=False)
db = mysql.connector.connect(host="localhost", user="root", password="mysql123", database="parking")
cursor = db.cursor()

def is_registered_vehicle(plate):
    cursor.execute("SELECT * FROM cars WHERE plate_number = %s", (plate,))
    car = cursor.fetchone()
    return car is not None

def recognize_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    ocr_result = reader.readtext(binary)

    if len(ocr_result) == 0:
        return None, 0, binary

    text = ocr_result[0][1]
    ocr_conf = ocr_result[0][2]

    text = text.replace(" ", "")
    text = re.sub(r"[^0-9가-힣]", "", text)

    return text, ocr_conf, binary

def is_valid_plate(text, ocr_conf):
    if text is None:
        return False
    if ocr_conf < 0.5:
        return False
    return re.fullmatch(r"[0-9]{2,3}[가-힣][0-9]{4}", text) is not None

cap = cv2.VideoCapture("plate_test.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

counter = Counter()
final_plate = None
gate_state = "WAITING"
parking_state = "EMPTY"


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 차단기 시스템
    if gate_state == "WAITING":
        results = model(frame, conf=0.1, imgsz=1280, verbose=False)
        result = results[0]

        if len(result.boxes) > 0:
            best_box = max(result.boxes, key=lambda box: float(box.conf[0]))

            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            plate = frame[y1:y2, x1:x2].copy()

            text, ocr_conf, binary = recognize_plate(plate)

            if is_valid_plate(text, ocr_conf):
                counter[text] += 1
                print(text, "→", counter[text], "회")

                if counter[text] >= 3:
                    final_plate = text
                    print("최종 번호판:", final_plate)

                    if is_registered_vehicle(final_plate):
                        print("등록 차량")
                        gate_state = "OPEN"
                    else:
                        print("미등록 차량")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Plate", plate)
            cv2.imshow("Binary", binary)

    elif gate_state == "OPEN":
        print("차단기 OPEN 상태")

        passage_detected = True

        if passage_detected:
            print("차량 통과 감지")
            print("차단기 CLOSE")

            gate_state = "WAITING"
            counter.clear()
            final_plate = None

            break # 테스트용

    # 주차 공간 시스템
    if parking_state == "EMPTY":
        parking_detected = True # 테스트용, 실제 IR 센서로 교체 예정

        if parking_detected:
            parking_state = "PARKED"
            print("주차 완료")
            # 나중에 앱에 주차 완료 전송
        
    elif parking_state == "PARKED":
        parking_detected = False # 임시 : 차량이 빠졌다고 가정

        if not parking_detected:
            parking_state = "EMPTY"
            print("차량 출차, 빈자리")
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()