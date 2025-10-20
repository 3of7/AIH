## Setup and Reproducibility

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd brain-tumor-classification
```

2. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

3. Update the `DATA_DIR` path in the training script to point to your dataset

4. Run training:
```bash
python brain_tumor_vit.py
```
```
