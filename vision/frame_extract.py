import cv2
import os

VIDEO_PATH = "plate_test.mp4"
OUTPUT_DIR = "new_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#영상 전체에서 균등하게 30장 추출
num_images = 30
interval = max(total_frames // num_images, 1)

print("전체 프레임:", total_frames)
print("추출 간격:", interval)

saved = 0
frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_idx % interval == 0 and saved < num_images:
        filename = os.path.join(OUTPUT_DIR, f"plate_{saved:03d}.jpg")

        cv2.imwrite(filename, frame)
        print("저장:", filename)

        saved += 1
    frame_idx += 1

cap.release()

print("완료:", saved, "장")