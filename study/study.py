import cv2
import numpy as np
import easyocr


# 1. 이미지 읽기
image = cv2.imread("car.jpg")

# 번호판 위치 찾기용 전처리

# 2. 전체 이미지 Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Noise 감소
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Edge 검출
edges = cv2.Canny(blur, 100, 200)

# 5. 끊어진 Edge를 조금 연결
kernel = np.ones((3, 3), np.uint8)

closed = cv2.morphologyEx(
    edges,
    cv2.MORPH_CLOSE,
    kernel
)
# 번호판 후보 찾기
# 6. Contour 찾기
contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

plate = None
best_area = 0

# 7. 찾은 Contour들을 하나씩 검사
for contour in contours:

    # 수동으로 잡던 Crop 좌표를 countour로 부터 자동으로 만들어주는 역할
    x, y, w, h = cv2.boundingRect(contour)

    # 사각형 넓이
    area = w * h

    # 가로 / 세로 비율
    ratio = w / h

    # 번호판처럼 가로로 긴 사각형만 후보로 선택
    if 2.5 < ratio < 6.0 and area > 500:

        # 후보가 여러 개라면 가장 큰 것을 사용
        if area > best_area:
            best_area = area
            plate = image[y:y+h, x:x+w]

            cv2.rectangle(
        image,
        (x, y),
        (x+w, y+h),
        (0, 255, 0),
        2
    )

# 번호판 OCR용 전처리
if plate is not None:

    plate_gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    plate_blur = cv2.GaussianBlur(
        plate_gray,
        (3, 3),
        0
    )

    plate_binary = cv2.adaptiveThreshold(
        plate_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    # OCR
    reader = easyocr.Reader(
        ['ko', 'en'],
        gpu=False
    )

    result = reader.readtext(plate_binary)

    print("OCR 결과:", result)

    if len(result) > 0:

        text = result[0][1]

        # 공백 제거
        text = text.replace(" ", "")

        print("번호판:", text)

        # 등록 차량 확인

        registered_cars = [
            "180호3682",
            "12가3456",
            "34나5678"
        ]

        if text in registered_cars:
            print("등록 차량")
        else:
            print("미등록 차량")


    cv2.imshow("Plate", plate)
    cv2.imshow("Plate Binary", plate_binary)

else:
    print("번호판 후보를 찾지 못했습니다.")


cv2.imshow("Original", image)
cv2.imshow("Edges", edges)
cv2.imshow("Closed", closed)

cv2.waitKey(0)
cv2.destroyAllWindows()