import cv2          # 이미지 처리: 색상 변환, 확대, 저장
import re           # 문자열 정리 및 번호판 형식 검사
import easyocr      # 번호판 이미지에서 문자 인식
import requests     # Flask 서버에 HTTP 요청 전송
import time         # 시간 측정 및 대기
from ultralytics import YOLO                # 번호판 위치 검출
from collections import Counter, deque      # 득표수 계산, 최근 OCR 결과 보관
from picamera2 import Picamera2, Preview    # Pi 카메라 제어 및 미리보기


model = YOLO("models/best_plate_yolo26.pt")
reader = easyocr.Reader(["ko", "en"], gpu=False)

def is_registered_vehicle(plate):
    response = requests.post(   # Flask 서버로 POST 요청을 보내고 응답 저장
        "http://127.0.0.1:5000/check_plate",    # 같은 장치에서 실행 중인 서버 주소
        json={"plate_number": plate},       # 번호판 문자열을
        timeout=5)
    response.raise_for_status()  # HTTP 오류가 있으면 예외 발생
    data = response.json()  # JSON 응답을 Python 데이터로 변환

    registered = data.get("registered")  # 등록 여부 추출, 키가 없으면 None
    if not isinstance(registered, bool):  # True/False 타입인지 확인
        raise ValueError("서버 응답에 등록 여부가 없거나 형식이 잘못됐습니다")

    return registered  # 등록이면 True, 미등록이면 False 반환

def recognize_plate(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # 부분별 대비 개선 설정
    gray = clahe.apply(gray)    # 대비 개선 적용
    gray = cv2.resize(gray, None, fx=3, fy=3, 
                      interpolation=cv2.INTER_CUBIC) # 보간법

    start = time.perf_counter() # OCR 시작 시간 기록
    ocr_result = reader.recognize(gray, detail=1)   # 문자와 신뢰도 등을 반환
    print("OCR:", (time.perf_counter() - start) * 1000, "ms")   # OCR 소요 시간 출력

    if len(ocr_result) == 0:
        return None, 0

    text = ocr_result[0][1].replace(" ", "")
    text = re.sub(r"[^0-9가-힣]", "", text)
    ocr_conf = ocr_result[0][2]

    print("OCR TEXT:", text)
    print("OCR CONF:", ocr_conf)
    return text, ocr_conf

# OCR로 읽은 문자열이 번호판 형식에 맞는지 검사
def is_valid_plate(text):
    if text is None:
        return False
    return re.fullmatch(r"[0-9]{2,3}[가-힣][0-9]{4}", text) is not None

# 카메라를 제어할 객체 생성, 이후 picam2를 통해 촬영, 설정, 종료 수행
picam2 = Picamera2()

# 카메라 설정 
camera_config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "XRGB8888"})

picam2.configure(camera_config) # 만든 설정을 카메라에 적용
picam2.start_preview(Preview.QTGL) # 카메라 영상을 화면에 보여주는 미리보기 창 준비
picam2.start()
time.sleep(1)

# 최근 유효한 OCR 결과를 최대 5개 저장, 6번째가 들어오면 가장 오래된 결과가 자동 삭제됨
recent_plates = deque(maxlen=5)  

final_plate = None  
gate_state = "WAITING"
recognition_start = None
rejected_time = None
last_detected_time = None
absence_timeout = 3.0

try:
    while True:
        if gate_state == "REJECTED":
            if time.monotonic() - rejected_time >= 5:
                print("재인식 대기 상태로 복귀")
                gate_state = "WAITING"
                recent_plates.clear()
                final_plate = None
                recognition_start = None
                rejected_time = None
            else:
                time.sleep(0.1)
            continue

        if gate_state == "OPEN":
            print("차단기 OPEN 상태")
            passage_detected = True  # 테스트용. 실제 센서 감지 아님.

            if passage_detected:
                print("차량 통과 감지 (테스트)")
                print("차단기 CLOSE (테스트)")
                gate_state = "WAITING"
                recent_plates.clear()
                final_plate = None
                recognition_start = None
                break
            continue

        frame = picam2.capture_array() # 카메라에서 영상 한 프레임을 NumPy 배열로 가져옴
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) # 4채널 -> BGR 3채널로 변환

        start = time.perf_counter()
        results = model(frame, conf=0.5, imgsz=640, classes=[1], verbose=False)
        print("YOLO:", (time.perf_counter() - start) * 1000, "ms")

        result = results[0]

        if len(result.boxes) == 0:
            if last_detected_time is not None:
                if time.monotonic() - last_detected_time >= absence_timeout:
                    recent_plates.clear()
                    recognition_start = None
                    last_detected_time = None
                    print("번호판 미검출 3초 → Voting 초기화")
            continue

        last_detected_time = time.monotonic()

        best_box = max(result.boxes, key=lambda box: float(box.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])

        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        print("CONF:", float(best_box.conf[0]))
        print("CLASS:", int(best_box.cls[0]))
        print("BOX:", x1, y1, x2, y2)

        plate = frame[y1:y2, x1:x2].copy()
        text, ocr_conf, processed = recognize_plate(plate)
      
        if is_valid_plate(text):
            if recognition_start is None:
                recognition_start = time.perf_counter()

            recent_plates.append(text)
            votes = Counter(recent_plates)
            candidate, count = votes.most_common(1)[0]

            print("최근 OCR:", list(recent_plates))
            print(candidate, "→", count, "표")

            if count >= 3:
                final_plate = candidate
                recognition_time = time.perf_counter() - recognition_start

                print("최종 번호판:", final_plate)
                print(f"번호판 확정 시간: {recognition_time:.2f}초")

                recent_plates.clear()
                recognition_start = None

                try:
                    registered = is_registered_vehicle(final_plate)

                except (requests.exceptions.RequestException, ValueError) as error:
                    print("API 조회 실패:", error)
                    print("승인 확인 불가 → 차단기 유지")
                    gate_state = "REJECTED"
                    rejected_time = time.monotonic()

                else:
                    if registered:
                        print("등록 차량")
                        gate_state = "OPEN"
                    else:
                        print("미등록 차량")
                        gate_state = "REJECTED"
                        rejected_time = time.monotonic()

except KeyboardInterrupt:
    print("\n프로그램 종료")

finally:
    picam2.stop()
    picam2.close()
    print("정상 종료")