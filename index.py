import os
import cv2
import torch
import numpy as np
import faiss
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import open_clip

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = Path("images")
ARTIFACT_DIR = Path("embeddings")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
BOX_CONF = 0.25
KP_CONF = 0.3

# Keypoint indices for COCO/YOLOv8-pose
SHOULDERS = [5, 6]
HIPS = [11, 12]

# -----------------------------
# SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Models
print("Loading models...")
pose_model = YOLO("yolov8n-pose.pt")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-16", pretrained="laion2b_s34b_b88k"
)
clip_model = clip_model.to(device).eval()
print("Models loaded.")

# -----------------------------
# HELPERS
# -----------------------------
def load_existing_paths(name):
    path_file = ARTIFACT_DIR / f"{name}_paths.txt"
    if not path_file.exists():
        return set()
    return set(line.strip() for line in path_file.read_text().splitlines())

def save_or_append_index(name, new_embs, new_paths):
    if new_embs.size == 0:
        return

    index_path = ARTIFACT_DIR / f"{name}.index"
    paths_path = ARTIFACT_DIR / f"{name}_paths.txt"

    faiss.normalize_L2(new_embs)

    if index_path.exists():
        idx = faiss.read_index(str(index_path))
        idx.add(new_embs)
        mode = "a"
    else:
        idx = faiss.IndexFlatIP(new_embs.shape[1])
        idx.add(new_embs)
        mode = "w"

    faiss.write_index(idx, str(index_path))
    with open(paths_path, mode) as f:
        for p in new_paths:
            f.write(f"{p}\n")
    print(f"[OK] {name}: +{len(new_embs)} (Total {idx.ntotal})")

def split_person(img):
    h, w = img.shape[:2]
    results = pose_model(img, conf=BOX_CONF, verbose=False)[0]

    if not results.boxes or len(results.boxes) == 0:
        return None, None, None

    # Pick biggest person by box area
    boxes = results.boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = np.argmax(areas)

    x1, y1, x2, y2 = boxes[idx].astype(int)
    person_h = y2 - y1

    kps = results.keypoints.data[idx].cpu().numpy()
    conf = results.keypoints.conf[idx].cpu().numpy()

    has_hips = conf[HIPS[0]] > KP_CONF and conf[HIPS[1]] > KP_CONF
    has_shoulders = conf[SHOULDERS[0]] > KP_CONF and conf[SHOULDERS[1]] > KP_CONF

    # Exit early for headshots/partial views
    if not has_hips:
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return crop, None, None

    hip_y = max(int(kps[HIPS[0]][1]), int(kps[HIPS[1]][1]))
    shoulder_y = min(int(kps[SHOULDERS[0]][1]), int(kps[SHOULDERS[1]][1])) if has_shoulders else None
    torso_len = (hip_y - shoulder_y) if shoulder_y else (person_h * 0.4)

    # Torso-relative cropping logic
    upper_end = hip_y - int(0.20 * torso_len)
    lower_start = hip_y - int(0.27 * torso_len)

    person = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    upper = img[max(0, y1):max(y1, int(upper_end)), max(0, x1):min(w, x2)]
    lower = img[max(y1, int(lower_start)):min(h, y2), max(0, x1):min(w, x2)]

    return person, upper, lower

# -----------------------------
# MAIN PIPELINE
# -----------------------------
start_time = time.perf_counter()

# Filter for images
image_paths = [str(p) for p in IMAGE_DIR.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

# Trackers for incremental loading
categories = ["clip_full", "clip_person", "clip_upper", "clip_lower"]
existing_data = {cat: load_existing_paths(cat) for cat in categories}

# Buffers for new data
buffers = {cat: {"imgs": [], "paths": []} for cat in categories}

for path in tqdm(image_paths, desc="Processing Images"):
    if path in existing_data["clip_full"]:
        continue

    img = cv2.imread(path)
    if img is None:
        continue

    # Helper: Preprocess for CLIP
    def to_clip(cv_img):
        return clip_preprocess(Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)))

    # 1. Full Image
    buffers["clip_full"]["imgs"].append(to_clip(img))
    buffers["clip_full"]["paths"].append(path)

    # 2. Split Crops
    person, upper, lower = split_person(img)
    
    crops = [person, upper, lower]
    crop_cats = ["clip_person", "clip_upper", "clip_lower"]

    for crop, cat in zip(crops, crop_cats):
        if crop is not None and crop.size > 0:
            buffers[cat]["imgs"].append(to_clip(crop))
            buffers[cat]["paths"].append(path)

# -----------------------------
# ENCODING & SAVING
# -----------------------------
for cat in categories:
    imgs = buffers[cat]["imgs"]
    paths = buffers[cat]["paths"]
    
    if not imgs:
        print(f"[SKIP] {cat}: No new images.")
        continue

    embs = []
    for i in tqdm(range(0, len(imgs), BATCH_SIZE), desc=f"Encoding {cat}"):
        batch = torch.stack(imgs[i : i + BATCH_SIZE]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(batch)
            emb /= emb.norm(dim=-1, keepdim=True)
        embs.append(emb.cpu().numpy())

    final_embs = np.vstack(embs).astype("float32")
    save_or_append_index(cat, final_embs, paths)

total_time = time.perf_counter() - start_time
print(f"\nIndexing complete in {total_time:.2f} seconds.")