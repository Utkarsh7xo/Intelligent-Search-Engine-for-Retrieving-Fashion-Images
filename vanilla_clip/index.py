import os
import cv2
import torch
import numpy as np
import faiss
import time
from PIL import Image
from tqdm import tqdm
import open_clip

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_DIR = "../images"
ARTIFACT_DIR = "embeddings_vanilla"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

INDEX_PATH = os.path.join(ARTIFACT_DIR, "vanilla_clip.index")
PATHS_FILE = os.path.join(ARTIFACT_DIR, "vanilla_paths.txt")

BATCH_SIZE = 32
MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"

# -----------------------------
# SETUP
# -----------------------------
start_time = time.perf_counter() # Start the clock

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(device).eval()

# -----------------------------
# INCREMENTAL CHECK
# -----------------------------
existing_paths = set()
if os.path.exists(PATHS_FILE):
    with open(PATHS_FILE, "r") as f:
        existing_paths = set(line.strip() for line in f)

# Filter out images we have already indexed
all_image_paths = [
    os.path.join(IMAGE_DIR, f) 
    for f in os.listdir(IMAGE_DIR) 
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_paths = [p for p in all_image_paths if p not in existing_paths]

if not image_paths:
    print("No new images to index. Everything is up to date.")
    exit()

print(f"Found {len(image_paths)} new images (skipping {len(existing_paths)} already indexed).")

# -----------------------------
# PROCESSING
# -----------------------------
all_embs = []
new_valid_paths = []

with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        batch_tensors = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_tensors.append(preprocess(img))
                new_valid_paths.append(path)
            except Exception as e:
                print(f"\nError loading {path}: {e}")
                continue

        if not batch_tensors:
            continue

        input_stack = torch.stack(batch_tensors).to(device)
        embeddings = model.encode_image(input_stack)
        
        # Normalize for Cosine Similarity
        embeddings /= embeddings.norm(dim=-1, keepdim=True)
        all_embs.append(embeddings.cpu().numpy())

# -----------------------------
# SAVE / APPEND INDEX
# -----------------------------
if all_embs:
    new_embs = np.vstack(all_embs).astype("float32")
    
    if os.path.exists(INDEX_PATH):
        # Load existing index and add new vectors
        index = faiss.read_index(INDEX_PATH)
        index.add(new_embs)
        write_mode = "a" # Append to the paths file
    else:
        # Create new index
        dim = new_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(new_embs)
        write_mode = "w" # Create new paths file

    faiss.write_index(index, INDEX_PATH)
    
    with open(PATHS_FILE, write_mode) as f:
        for p in new_valid_paths:
            f.write(p + "\n")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print("-" * 30)
    print(f"Indexing complete!")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"New images added: {len(new_valid_paths)}")
    print(f"Total images in index: {index.ntotal}")
    print("-" * 30)
else:
    print("No new embeddings were generated.")