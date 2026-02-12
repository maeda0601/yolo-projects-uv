from ultralytics import YOLO
import cv2
import time

interval = 1.0  # 秒
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# 👇 ここで解像度を変更
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Webカメラが開けませんでした。")

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 👇 指定間隔まで待つ
    elapsed = time.time() - start
    if elapsed < interval:
        time.sleep(interval - elapsed)

cap.release()
cv2.destroyAllWindows()