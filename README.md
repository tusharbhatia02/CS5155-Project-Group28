# CS5155-Project-Group28
CsI5155-Machine Learning (Fall 2025)-Project

# Cross-Modal Recipe Retrieval with Nutrition-Aware Re-ranking

This repository contains the implementation for our CSI 5155 Machine Learning project: **Cross-Modal Retrieval for Personalized and Nutritionally-Aware Recipe Recommendation**.

The project trains a dual-encoder model (ResNet-50 for images + DistilBERT for text) using contrastive learning on three progressively larger subsets of the Recipe1M+ dataset. A lightweight nutritional re-ranking module is supported in a separate interface.

## Repository Structure
```
├── 12k_training.ipynb        # Training on ~7,100 matched pairs
├── 50k_training.ipynb        # Training on ~23,127 matched pairs
├── 100k_training.ipynb       # Training on ~48,982 matched pairs
├── README.md                 # Project documentation
```

Each notebook includes preprocessing, model definition, training, evaluation, and model saving.

## Experimental Setup / Runtime Environment

This project was executed on **Google Colab** and **Kaggle Notebooks** with GPU acceleration.

### Google Colab Configuration
- **GPU:** NVIDIA T4
- **GPU Memory:** 15.0 GB
- **System RAM:** 12.7 GB (2.7 GB used during training)
- **Disk:** 112.6 GB available (38.1 GB used at peak)
- **Python Environment:** Python 3.10+, PyTorch 2.x, Torchvision, Transformers

### Kaggle Notebook Configuration
- **GPU:** NVIDIA T4
- **GPU Memory:** 15.0 GB
- **System RAM:** 32 GB
- **Disk:** ~200 GB available
- **Runtime:** GPU Accelerator enabled

## Dependencies

Install required libraries:
```bash
pip install torch torchvision transformers tqdm numpy pandas pillow
```

The notebooks automatically download DistilBERT weights.

## Dataset (Recipe1M+)

The Recipe1M+ dataset is not included in this repository due to size and licensing.

You must manually download:
- `layer1.json` — recipe text
- `layer2.json` — image–recipe mapping
- Image directory (12k, 50k, or 100k downloaded images)

## Dependencies

Install required libraries:
```bash
pip install torch torchvision transformers tqdm numpy pandas pillow
```

Update the paths inside each notebook:
```python
LAYER1_JSON = "/path/to/layer1.json"
LAYER2_JSON = "/path/to/layer2.json"
IMG_DIR = "/path/to/images/"
```


## How to Use

1. Download Recipe1M+ images + JSON files
2. Update dataset paths in each notebook
3. Run preprocessing cells
4. Train the model (GPU recommended)
5. Evaluate retrieval performance
6. Save and load the best model for inference or demos
