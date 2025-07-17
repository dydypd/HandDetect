import cv2
import json
import os

VIDEO_PATH = "demi.mp4"
OUTPUT_LABELS = "labels.json"

# Lưu kết quả: danh sách {frame_idx, key}
labels = []
frame_idx = 0

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_frame(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None

print("[INFO] Hướng dẫn:")
print("← / →: lùi/tiến 1 frame")
print("A–Z: Gán phím được nhấn tại frame hiện tại")
print("S: Lưu label & thoát")

while True:
    frame = get_frame(frame_idx)
    if frame is None:
        print("Hết video.")
        break

    # Hiển thị frame số
    disp = frame.copy()
    cv2.putText(disp, f"Frame: {frame_idx}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Labeling", disp)
    key = cv2.waitKeyEx(0)
    print(key)
    if key == 96:
        print("Đã lưu.")
        break
    elif key == 2424832:  # ← left
        frame_idx = max(0, frame_idx - 1)
    elif key == 2555904:  # → right
        frame_idx = min(total_frames - 1, frame_idx + 1)
    elif ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
        char = chr(key).upper()
        labels.append({"frame": frame_idx, "key": char})
        print(f"✔️ Đã gán nhãn: {char} tại frame {frame_idx}")
        frame_idx = min(total_frames - 1, frame_idx + 1)  # tự tiến 1 frame

cap.release()
cv2.destroyAllWindows()

with open(OUTPUT_LABELS, "w") as f:
    json.dump(labels, f, indent=2)
print(f"📁 Đã lưu nhãn tại: {OUTPUT_LABELS}")
