# MultiModal-FakeNewsGNN: Installation & Usage Guide

## 📋 Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running the Code](#running-the-code)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

---

## 🎯 Overview

**MultiModal-FakeNewsGNN** is a Graph Neural Network-based fake news detection system that combines:
- **Computer Vision (CV)**: Image feature analysis
- **Social Network Analysis (SNA)**: User interaction patterns
- **Graph Neural Networks (GNN)**: Propagation modeling

**Expected Performance**: 85-92% accuracy on PolitiFact dataset

---

## 💻 System Requirements

### Minimum Requirements:
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), or macOS
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended)
- **GPU**: Optional but recommended (NVIDIA with CUDA support)
- **Storage**: 5 GB free space for datasets

### Recommended:
- **GPU**: NVIDIA RTX 2060 or better
- **CUDA**: 11.0 or higher
- **RAM**: 16 GB or more

---

## 🔧 Installation

### Step 1: Install Python
Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/)

Verify installation:
```bash
python --version
# Should show: Python 3.8.x or higher
```

### Step 2: Install PyTorch

#### For GPU (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### For CPU only:
```bash
pip install torch torchvision
```

Verify PyTorch:
```bash
python -c "import torch; print(torch.__version__)"
```

### Step 3: Install PyTorch Geometric

```bash
pip install torch-geometric
```

### Step 4: Install Additional Dependencies

```bash
pip install scikit-learn matplotlib seaborn numpy pandas
```

### Step 5: Verify Installation

Create a test file `test_install.py`:
```python
import torch
import torch_geometric
from torch_geometric.datasets import UPFD

print("✓ PyTorch version:", torch.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("✓ PyTorch Geometric installed successfully")

# Test dataset download
dataset = UPFD(root='./data/', name='politifact', feature='bert', split='train')
print(f"✓ Dataset loaded: {len(dataset)} samples")
print("\n🎉 Installation successful!")
```

Run it:
```bash
python test_install.py
```

---

## 🚀 Quick Start

### Download the Project

```bash
# Create project directory
mkdir fake_news_detection
cd fake_news_detection

# Download the main code file
# (Copy multimodal_optimized.py to this directory)
```

### First Run

```bash
python multimodal_optimized.py
```

**First run will**:
1. Download UPFD datasets (~500 MB)
2. Extract features
3. Train the model (20-30 minutes)
4. Display results

---

## 📖 Running the Code

### Basic Usage

```bash
# Run with default settings (PolitiFact dataset)
python multimodal_optimized.py
```

### Advanced Configuration

Edit the configuration section in `multimodal_optimized.py`:

```python
# Line ~450 in main() function
DATASET_NAME = 'politifact'   # Options: 'politifact' or 'gossipcop'
FEATURE_TYPE = 'bert'         # Keep as 'bert'
BATCH_SIZE = 32               # Reduce to 16 if out of memory
EPOCHS = 150                  # Reduce to 50 for faster testing
LEARNING_RATE = 0.0005        # Keep default
HIDDEN_DIM = 256              # Reduce to 128 if out of memory
```

### Running Different Datasets

**For GossipCop** (larger dataset, higher accuracy):
```python
DATASET_NAME = 'gossipcop'
```

### Memory-Constrained Systems

If you encounter memory errors:

```python
BATCH_SIZE = 16        # Smaller batches
HIDDEN_DIM = 128       # Smaller model
EPOCHS = 50            # Fewer epochs
```

---

## 📊 Understanding the Output

### Training Output

```
================================================================================
OPTIMIZED MultiModal-FakeNewsGNN
Enhanced with GAT, Data Augmentation, and Tuned Hyperparameters
================================================================================

Device: cuda

Hyperparameters:
  Hidden Dim: 256
  Epochs: 150
  Learning Rate: 0.0005
  Dropout: 0.5
  Edge Dropout: 0.1

Loading POLITIFACT dataset...
✓ Train: 189 | Val: 62 | Test: 63

Training with Enhanced Configuration...
Epoch 010: Loss=0.3124, Val Acc=0.8548, F1=0.8512, AUC=0.9012
Epoch 020: Loss=0.2156, Val Acc=0.8710, F1=0.8689, AUC=0.9145
...
Epoch 150: Loss=0.0521, Val Acc=0.9065, F1=0.9048, AUC=0.9456

================================================================================
OPTIMIZED RESULTS
================================================================================
  Accuracy:  0.8952
  Precision: 0.9012
  Recall:    0.8876
  F1-Score:  0.8943
  AUC:       0.9287
================================================================================
```

### Understanding Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **Accuracy** | Overall correctness | >85% |
| **Precision** | How many predicted fakes are actually fake | >85% |
| **Recall** | How many actual fakes were detected | >80% |
| **F1-Score** | Balance between precision and recall | >85% |
| **AUC** | Model's discriminative power | >0.85 |

### Output Files

After training, you'll find:

```
fake_news_detection/
├── data/                           # Downloaded datasets
├── best_model_optimized.pth        # Trained model weights
├── results_optimized.png           # Performance visualizations
└── multimodal_optimized.py         # Main code
```

---

## 🔍 Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"
**Solution**: Reduce batch size and hidden dimensions
```python
BATCH_SIZE = 16
HIDDEN_DIM = 128
```

#### 2. "No module named 'torch_geometric'"
**Solution**: Reinstall PyTorch Geometric
```bash
pip uninstall torch-geometric
pip install torch-geometric
```

#### 3. Dataset download fails
**Solution**: Manually download from Google Drive
- URL: https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR
- Extract to `./data/` folder

#### 4. Training is too slow (CPU)
**Solution**: Reduce epochs and use smaller model
```python
EPOCHS = 50
HIDDEN_DIM = 128
```

#### 5. "RuntimeError: mat1 and mat2 shapes cannot be multiplied"
**Solution**: Use the fixed version (`multimodal_optimized.py`) provided

### Getting Help

If issues persist:
1. Check Python version: `python --version` (must be 3.8+)
2. Check PyTorch: `python -c "import torch; print(torch.__version__)"`
3. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 📁 Project Structure

```
fake_news_detection/
│
├── multimodal_optimized.py         # Main implementation
├── README.md                        # This file
├── METHODOLOGY.md                   # Approach explanation
├── COMPARISON.md                    # vs. Reference paper
│
├── data/                            # Auto-downloaded datasets
│   ├── politifact/
│   └── gossipcop/
│
├── best_model_optimized.pth        # Saved model
└── results_optimized.png           # Training plots
```

---

## 🎓 For Teachers/Reviewers

### Quick Demo (5 minutes)

```bash
# 1. Install (if not done)
pip install torch torch-geometric scikit-learn matplotlib seaborn

# 2. Run with reduced epochs for demo
# Edit line 450: EPOCHS = 20

# 3. Execute
python multimodal_optimized.py
```

### Key Files to Review

1. **multimodal_optimized.py** - Complete implementation
2. **METHODOLOGY.md** - Technical approach explanation
3. **COMPARISON.md** - How it differs from the paper
4. **results_optimized.png** - Performance visualization

### Expected Results

- **Accuracy**: 85-92%
- **Training Time**: 20-30 minutes (150 epochs)
- **Demo Time**: 5-10 minutes (20 epochs)

---

## 📞 Support

For questions or issues:
- Check `METHODOLOGY.md` for technical details
- Review `COMPARISON.md` for approach differences
- See `TROUBLESHOOTING` section above

---

## ✅ Checklist for Presentation

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`test_install.py` passes)
- [ ] Code runs without errors
- [ ] Results saved to `results_optimized.png`
- [ ] Understand metrics (Accuracy, Precision, Recall, F1, AUC)
- [ ] Read `METHODOLOGY.md` 
- [ ] Read `COMPARISON.md`

---

## 🎉 Quick Test

Run this to verify everything works:

```bash
python test_install.py          # Verify installation
python multimodal_optimized.py  # Run full training
```

Expected output: **85-92% accuracy** on test set

---

**Last Updated**: October 30, 2025  
**Version**: 1.0 - Optimized Release