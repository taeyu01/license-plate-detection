import cv2
import re
import time
import easyocr
from ultralytics import YOLO
from collections import Counter

# 모델 / OCR
model = YOLO("models/best_plate_yolo26.pt")
reader = easyocr.Reader(["ko", "en"], gpu=False)

cap = cv2.VideoCapture("plate_test.mp4")

# 번호판별 인식 횟수
counter = Counter()

# 최종 번호판
final_plate = None


def is_valid_plate(text):
    """
    일반적인 번호판 형태:
    숫자 2~3자리 + 한글 1글자 + 숫자 4자리
    """
    return re.fullmatch(
        r"[0-9]{2,3}[가-힣][0-9]{4}",
        text
    ) is not None


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        imgsz=512,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    if len(result.boxes) > 0:

        # YOLO confidence가 가장 높은 box
        best_box = max(
            result.boxes,
            key=lambda box: float(box.conf[0])
        )

        x1, y1, x2, y2 = map(
            int,
            best_box.xyxy[0]
        )

        plate = frame[y1:y2, x1:x2].copy()

        if plate.size > 0:

            # -------------------------
            # OpenCV 전처리
            # -------------------------

            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                None,
                fx=2,
                fy=2,
                interpolation=cv2.INTER_CUBIC
            )

            # -------------------------
            # OCR
            # -------------------------

            start = time.perf_counter()

            ocr_result = reader.recognize(
                gray,
                detail=1
            )

            end = time.perf_counter()

            print(
                "OCR TIME:",
                round((end - start) * 1000, 1),
                "ms"
            )

            if len(ocr_result) > 0:

                text = ocr_result[0][1]

                # 공백 / 특수문자 제거
                text = text.replace(" ", "")
                text = re.sub(
                    r"[^0-9가-힣]",
                    "",
                    text
                )

                print("OCR:", text)

                # -------------------------
                # 번호판 형식 검사
                # -------------------------

                if is_valid_plate(text):

                    counter[text] += 1

                    print(
                        "VALID:",
                        text,
                        "→",
                        counter[text],
                        "회"
                    )

                    # -------------------------
                    # 3회 이상 → 최종 확정
                    # -------------------------

                    if counter[text] >= 3:

                        final_plate = text

                        print()
                        print("====================")
                        print("최종 번호판:", final_plate)
                        print("====================")

                        break

                else:
                    print("INVALID:", text)

            # 화면 표시
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.imshow("Plate Crop", plate)
            cv2.imshow("OCR Input", gray)

    cv2.imshow("Detection", frame)

    if final_plate is not None:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("프로그램 종료")