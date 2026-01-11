import os
import json
import faiss
import torch
import numpy as np
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import open_clip
from dotenv import load_dotenv
from google import genai

load_dotenv()

# =====================================================
# CONFIG & GLOBALS
# =====================================================
ARTIFACT_DIR = Path("embeddings")
FINAL_TOP_K = 5
GEMINI_MODEL = "gemini-2.5-flash-lite" #"gemini-2.0-flash"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Clients
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =====================================================
# SYSTEM PROMPT
# =====================================================
SYSTEM_PROMPT = """
**Role:** You are a Fashion Query Parser specialized in image retrieval systems. Your task is to decompose a user's natural language fashion query into four specific visual categories: upper body, lower body, full body, and complete image.

**Definitions:**
1. **upper_body**: Extracts specific tops, outerwear, and accessories worn on the torso (e.g., shirts, jackets, ties, scarves, sweaters).
2. **lower_body**: Extracts specific garments worn on the legs (e.g., jeans, trousers, skirts, shorts).
3. **full_body**: Includes one-piece garments (e.g., dresses, jumpsuits, suits) AND stylistic descriptors (e.g., formal, casual, business attire, streetwear).
4. **complete_image**: Includes environmental context, locations, background elements, and person actions (e.g., "in a park", "inside an office", "sitting on a bench", "walking").

**Output Format:**
Return ONLY a valid JSON object with the following structure:
{
  "upper_body": [],
  "lower_body": [],
  "full_body": [],
  "complete_image": []
}

**Strict Rules:**
- Do not add conversational filler or explanations.
- Don't change the query words or add anything new just segregate them into the four categories.
- If a category has no relevant information, return an empty list `[]`.
- Adjectives (colors, patterns, textures) must stay attached to their respective items
- For multitple things in a category, separate them into distinct prompts within the list.
- For vibe of clothes (e.g., "formal", "casual"), place them under full_body only.
"""

# =====================================================
# CORE FUNCTIONS
# =====================================================

def parse_query_with_gemini(query: str):
    """Uses Gemini to segment the natural language query into fashion categories."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=query,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️ Gemini Parsing Failed: {e}")
        return {cat: [] for cat in ["upper_body", "lower_body", "full_body", "complete_image"]}

def load_clip_model():
    """Loads and returns the CLIP model and tokenizer."""
    print(f"Loading CLIP on {device}...")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    return model.to(device).eval()

def embed_text(text: str, model):
    """Generates a normalized text embedding for a given string."""
    tokens = open_clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype("float32")

def load_all_indices():
    """Reads all FAISS indices and path files from the artifacts directory."""
    index_data = {}
    categories = {
        "upper_body": "clip_upper",
        "lower_body": "clip_lower",
        "full_body": "clip_person",
        "complete_image": "clip_full"
    }
    
    for cat, file_prefix in categories.items():
        idx_path = ARTIFACT_DIR / f"{file_prefix}.index"
        txt_path = ARTIFACT_DIR / f"{file_prefix}_paths.txt"
        
        if idx_path.exists() and txt_path.exists():
            idx = faiss.read_index(str(idx_path))
            paths = txt_path.read_text().splitlines()
            # Extract all vectors for dot-product similarity calculation
            vectors = idx.reconstruct_n(0, idx.ntotal)
            index_data[cat] = (vectors, paths)
        else:
            print(f"⚠️ Missing artifacts for {cat}")
            
    return index_data

def calculate_custom_score(similarities):
    """
    Applies the custom formula: f(x) = [prod(xi)^2 * n] / sum(xi^n)
    where xi are similarity scores.
    """
    x = np.array(similarities)
    n = len(x)
    if n > 1:
        prod_xi = np.prod(x)
        sum_pow_xi = np.sum(x**n)
        return ((10*prod_xi)**2 * n) /(10* (sum_pow_xi + 1e-9))
    return x[0] if n == 1 else 0.0

# =====================================================
# SEARCH PIPELINE
# =====================================================

def search(query: str, model, index_data):
    parsed = parse_query_with_gemini(query)
    print("\n[Parsed Query]:", json.dumps(parsed, indent=2))

    active_segments = [seg for seg, items in parsed.items() if items]
    if not active_segments:
        return []

    # Map to track scores and details per image path
    all_image_paths = set().union(*(data[1] for data in index_data.values()))
    score_vectors = {path: [] for path in all_image_paths}
    breakdown = {path: {seg: {} for seg in active_segments} for path in all_image_paths}

    # Calculate similarities for each segment
    for segment in active_segments:
        vectors, paths = index_data[segment]
        
        for prompt in parsed[segment]:
            q_emb = embed_text(prompt, model)
            # Dot product (Cosine Similarity on normalized vectors)
            sims = np.dot(q_emb, vectors.T).flatten()
            sims = np.clip(sims, 1e-5, None) # Prevent zeros for product formula
            
            path_sim_map = dict(zip(paths, sims))
            for path in all_image_paths:
                val = path_sim_map.get(path, 1e-5)
                score_vectors[path].append(val)
                if path in path_sim_map:
                    breakdown[path][segment][prompt] = float(val)

    # Compute final rankings
    final_rankings = []
    for path, scores in score_vectors.items():
        final_score = calculate_custom_score(scores)
        final_rankings.append((path, final_score))

    final_rankings.sort(key=lambda x: x[1], reverse=True)
    top_results = final_rankings[:FINAL_TOP_K]

    # Print Results Table
    print("\n" + "="*80)
    print(f"{'IMAGE PATH':<55} | {'SCORE':<10}")
    print("="*80)
    for path, score in top_results:
        print(f"{Path(path).name:<55} | {score:.4e}")
        for seg in active_segments:
            for prompt, sim in breakdown[path][seg].items():
                print(f"  └─ {seg[:5]} | {prompt[:30]:<30} | sim: {sim:.3f}")
    print("="*80)

    return top_results

def display_results(results):
    if not results: return
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1: axes = [axes]
    
    for i, (path, score) in enumerate(results):
        img = Image.open(path).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(f"Score: {score:.3f}", fontsize=10)
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

# =====================================================
# MAIN RUNNER
# =====================================================
if __name__ == "__main__":
    clip_model_inst = load_clip_model()
    index_storage = load_all_indices()
    
    test_query = "A red tie and a white shirt in a formal setting"
    
    results = search(test_query, clip_model_inst, index_storage)
    display_results(results)