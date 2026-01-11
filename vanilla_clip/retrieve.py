# retrieve.py
import faiss
import torch
import numpy as np
from PIL import Image
import open_clip
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
ARTIFACT_DIR = "embeddings_vanilla"
TOP_K = 5

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD CLIP
# -----------------------------
clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-16",
    pretrained="laion2b_s34b_b88k"
)
clip_model = clip_model.to(device).eval()

# -----------------------------
# LOAD INDEX
# -----------------------------
index = faiss.read_index(f"{ARTIFACT_DIR}/vanilla_clip.index")

with open(f"{ARTIFACT_DIR}/vanilla_paths.txt") as f:
    paths = [l.strip() for l in f]

# -----------------------------
# EMBED TEXT
# -----------------------------
def embed_text(text):
    tokens = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        e = clip_model.encode_text(tokens)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy().astype("float32")

# -----------------------------
# SEARCH
# -----------------------------
def retrieve(query):
    q = embed_text(query)
    sims, idxs = index.search(q, TOP_K)

    return [
        (paths[i], sims[0][j])
        for j, i in enumerate(idxs[0])
    ]

# -----------------------------
# DISPLAY
# -----------------------------
def display(results):
    n = len(results)
    plt.figure(figsize=(4 * n, 4))

    for i, (path, score) in enumerate(results):
        img = Image.open(path).convert("RGB")
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{score:.3f}")

    plt.tight_layout()
    plt.show()

# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    #query = "Someone wearing a blue shirt sitting on a park bench."
    #query = "Someone wearing a blue shirt sitting on a park bench"
    query = "A red tie and a white shirt in a formal setting."
    #query = "red shirt with blue pants"
    results = retrieve(query)

    for p, s in results:
        print(f"{s:.3f} | {p}")

    display(results)
