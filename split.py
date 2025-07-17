import os
import cv2
import json
import shutil
from glob import glob

VIDEO_PATH = "demi.mp4"
LABEL_FILE = "labels.json"
FRAME_DIR = "data/frames"
SEQ_DIR = "data/sequences"
T = 6  # số frame trong mỗi chuỗi
OFFSET_BEFORE = 3  # số frame trước frame nhấn


# Step 1: Trích frame từ video
def extract_frames():
    os.makedirs(FRAME_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[1/3] Trích xuất {total} frame (cắt 20% dưới + phải)...")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # CROP: giữ 80% trên + 80% bên trái
        h, w, _ = frame.shape
        crop_h = int(h * 0.8)
        crop_w = int(w * 0.8)
        frame_cropped = frame[:crop_h, :crop_w, :]  # giữ 80% trên và trái

        # Lưu
        path = os.path.join(FRAME_DIR, f"frame_{idx:05d}.jpg")
        cv2.imwrite(
            path,
            frame_cropped,
            [cv2.IMWRITE_JPEG_QUALITY, 75]  # Giảm nhẹ chất lượng
        )
        idx += 1

    cap.release()
    print(f"✅ Đã lưu {idx} frame (cropped) vào {FRAME_DIR}")




# Step 2: Tạo map frame index → file path
def get_frame_map():
    frames = sorted(glob(os.path.join(FRAME_DIR, "frame_*.jpg")))
    return {int(os.path.basename(f).split("_")[1].split(".")[0]): f for f in frames}


# Step 3: Tạo chuỗi frame quanh nhãn
def create_sequences():
    with open(LABEL_FILE, "r") as f:
        labels = json.load(f)

    os.makedirs(SEQ_DIR, exist_ok=True)
    frame_map = get_frame_map()
    max_idx = max(frame_map.keys())
    seq_id = 0

    print(f"[2/3] Tạo chuỗi frame từ {len(labels)} nhãn...")

    for item in labels:
        center = item["frame"]
        key = item["key"]

        start = center - OFFSET_BEFORE
        end = start + T

        if start < 0 or end > max_idx + 1:
            print(f"[⚠️] Bỏ qua frame {center} (gần biên)")
            continue

        seq_path = os.path.join(SEQ_DIR, f"seq_{seq_id:04d}")
        os.makedirs(seq_path, exist_ok=True)

        for i in range(T):
            frame_idx = start + i
            src = frame_map[frame_idx]
            dst = os.path.join(seq_path, f"frame_{i:02d}.jpg")
            shutil.copy(src, dst)

        with open(os.path.join(seq_path, "label.json"), "w") as f:
            json.dump({"pressed_key": key}, f)

        print(f"✔️ seq_{seq_id:04d} | frame {center} | key = {key}")
        seq_id += 1

    print(f"[✅] Tạo xong {seq_id} chuỗi trong {SEQ_DIR}")


# Main runner
if __name__ == "__main__":
    extract_frames()
    create_sequences()
