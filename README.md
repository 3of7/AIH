## Classification of Brain Tumors using the high resolution MRI images
This project aims to develop an ML model to help classifying three types 
of brain tumors, viz. glioma, meningioma, and pituitary adenoma.
### Clinical Importance: 
The brain tumors glioma, meningioma, and pituitary adenoma differ in origin 
and treatment, making accurate classification vital for surgical and
therapeutic decisions. 
### Imaging Modality: 
T1-weighted contrast-enhanced MRI highlights tumor boundaries and tissue
characteristics, providing crucial input for diagnostic models. 
### Tumor Differences: 
#### Gliomas
 Originate within brain tissue and often infiltrate diffusely; 
#### Meningiomas 
 Arise from meninges and are usually well-circumscribed; 
#### Pituitary adenomas
 Occur near the optic chiasm, affecting hormonal balance.
### Diagnostic Challenge: 
Visual overlap among tumor types makes manual diagnosis error-prone,
motivating automated deep learning approaches. 
### AI Objective: 
Train Vision Transformers to learn discriminative spatial and texture
patterns from MRI data, enabling rapid, reproducible, and accurate
classification for clinical support.
Vision Transformers excel at medical image classification, and 
brain tumor classification is a well-established use case. The 
Dataset size of 3000 images (~1000 per class) is reasonable for
fine-tuning a pre-trained ViT model. While not huge, it's 
workable with proper augmentation and transfer learning


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
