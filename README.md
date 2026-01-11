# Intelligent Search Engine for Retrieving Fashion Images

An intelligent **fashion image retrieval search engine** that finds relevant fashion images from a dataset using **natural language descriptions**. This project enhances traditional image search (e.g., CLIP) by combining **pre-segmentation** and **LLM parsing** to overcome limitations of zero-shot similarity models while maintaining their flexibility.

---

## 🚀 Features

- 🔍 **Natural Language Image Search:** Retrieve fashion images by describing what you want in plain text.
- 🧠 **Pre-Segmentation + LLM Parsing:** Uses segmentation to isolate key features and Large Language Models to interpret the text query.
- ⚡ **Zero-Shot Capabilities:** Leveraging CLIP-like models for generalization without training on exact categories.
- 📦 Ready to run with minimal setup.

---

## 📁 Repository Structure

```

.
├── index.py                   # Indexing and setup script
├── retrieve.py                # Retrieval engine for running queries
├── requirements.txt           # Python dependencies
├── vanilla_clip/              # Baseline CLIP implementation for comparison                 
└── results/                   # Example outputs and evaluation results
````

---

## 🧠 How It Works

1. **Image Preprocessing & Segmentation:** Break input images into meaningful regions before embedding.
2. **Embedding & Parsing:** Feed segmented images and description text to an LLM/embedding model.
3. **Similarity Search:** Compare embeddings using nearest-neighbor or semantic similarity techniques.
4. **Return Results:** Outputs images most closely matching the description.

> *This approach improves image retrieval accuracy over vanilla CLIP-based search by understanding the query at a deeper semantic level.*

---

## 🛠️ Getting Started

### 🔧 Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
````

### 📦 Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/Utkarsh7xo/Intelligent-Search-Engine-for-Retrieving-Fashion-Images.git
   cd Intelligent-Search-Engine-for-Retrieving-Fashion-Images
   ```

2. Add your **environment variables** in `.env` (e.g., model API keys, paths).

---

## ▶️ Run

### Index Images

```bash
python index.py
```

### Query the Engine

```bash
python retrieve.py
```

Provide a natural language text prompt in the 'test_query' variable to get similar fashion images from the database.

---


## 🧩 Dependencies

torch
torchvision
open-clip-torch 
faiss-cpu
numpy
Pillow
tqdm
opencv-python
matplotlib
ultralytics
google-genai
python-dotenv
---

## 📄 License

This project is open source — feel free to use and modify it.

---
