# License Plate Detection

YOLO 기반 자동차 번호판 탐지 프로젝트입니다.

## Features
- Custom YOLO model for license plate detection
- Bounding box extraction
- Automatic license plate cropping
- OpenCV image processing

## Pipeline

Vehicle Image
→ YOLO Detection
→ Bounding Box
→ Plate Crop
→ OCR (planned)

## Files

- `plate_detect.py` : 번호판 탐지 및 crop 코드
- `best.pt` : 학습된 모델 파일 (GitHub에는 업로드하지 않음)

## Next Step

- OpenCV preprocessing
- EasyOCR
- Vehicle database lookup
