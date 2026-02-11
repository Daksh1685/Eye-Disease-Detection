#  Eye Disease Detection System

A deep learning-based web application for detecting eye diseases from OCT (Optical Coherence Tomography) images using MobileNetV2 architecture with DRUSEN class weight boost.

## Features

- **Real-time Disease Detection**: Upload OCT images and get instant predictions
- **4 Disease Classes**: CNV, DME, DRUSEN, NORMAL
- **93.44% Accuracy**: High-performance MobileNetV2 model with DRUSEN boost
- **User-Friendly Interface**: Built with Streamlit for easy interaction
- **Fast Inference**: Lightweight and optimized model
- **Detailed Metrics**: View per-class performance metrics

## Supported Diseases

| Disease | Description | Abbreviation |
|---------|-------------|--------------|
| **CNV** | Choroidal Neovascularization | CNV |
| **DME** | Diabetic Macular Edema | DME |
| **DRUSEN** | Age-related Macular Degeneration | DRUSEN |
| **NORMAL** | Healthy Eyes | NORMAL |

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Daksh1685/Eye-Disease-Detection.git
cd Eye-Disease-Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the model**
- Download `eye_disease_model_boosted_drusen.h5` from the repository releases
- Place it in the same directory as `streamlit_app.py`

### Running the Application

**On Windows:**
```bash
run_streamlit.bat
```

**On Mac/Linux:**
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Project Structure

```
Eye-Disease-Detection/
├── streamlit_app.py                    # Main Streamlit application
├── eye_disease_training_drusen.py      # Training code for the model
├── requirements.txt                    # Python dependencies
├── run_streamlit.bat                   # Windows launcher script
├── README.md                           # This file
└── .gitignore                          # Files to exclude from git
```

## Model Details

**Architecture:** MobileNetV2
- Pre-trained on ImageNet
- Fine-tuned on 109,309 OCT images
- DRUSEN class weight boost (+35% for minority class balancing)

**Performance:**
- Test Accuracy: **93.44%**
- Training Epochs: 75
- Batch Size: 32
- Input Size: 224×224×3 (RGB)

**Per-Class Metrics:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| CNV | 99.52% | 87.72% | 93.25% |
| DME | 82.18% | 96.12% | 88.61% |
| DRUSEN | 65.43% | 89.40% | 75.56% |
| NORMAL | 97.93% | 96.44% | 97.18% |

## Using the Application

1. **Upload an Image**
   - Click on the upload area
   - Select a JPG, PNG, or BMP OCT image
   - Image should be at least 224×224 pixels

2. **Get Predictions**
   - Click "Analyze Image"
   - Wait 1-3 seconds for inference
   - View prediction with confidence score

3. **View Details**
   - See predicted disease class
   - Check confidence percentage
   - Review detailed per-class metrics

4. **Navigate Tabs**
   - **Prediction**: Upload and analyze images
   - **Model Info**: View architecture and metrics
   - **Help**: FAQs and usage guide


## Technical Stack

- **Frontend**: Streamlit 1.28.0+
- **Deep Learning**: TensorFlow 2.13.0+, Keras 3.0.0+
- **Image Processing**: Pillow 9.5.0+
- **Data Processing**: NumPy 1.24.0+
- **Visualization**: Matplotlib 3.7.0+, Seaborn 0.12.0+
- **ML Tools**: scikit-learn 1.3.0+

## Requirements

```
streamlit>=1.28.0
tensorflow>=2.13.0
keras>=3.0.0
numpy>=1.24.0
pillow>=9.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
```

## Training

To retrain the model with your own data:

```bash
python eye_disease_training_drusen.py
```

Ensure your dataset follows this structure:
```
dataset/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
├── val/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── test/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

## Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 Base (ImageNet pre-trained)
    ↓
Global Average Pooling
    ↓
Dense(128, ReLU, Dropout=0.5)
    ↓
Dense(4, Softmax)
    ↓
Output (4 classes)
```

## Results

- **Trained on**: 109,309 OCT images
- **Training Approach**: 3-phase transfer learning
- **Best Accuracy**: 93.44% on test set
- **Training Time**: ~2-3 hours on GPU

> *Remember: This tool is for research only. Always consult medical professionals for clinical decisions.*
