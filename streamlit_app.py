"""
Eye Disease Detection Streamlit App
====================================
Deploy eye_disease_model_boosted_drusen.h5 model for inference on unseen data
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Eye Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [role="tablist"] button {
        font-size: 16px;
        font-weight: bold;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 10px;
        color: #333;
    }
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 10px;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# =====================================================================
# SIDEBAR CONFIGURATION
# =====================================================================
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    st.markdown("### 📝 About")
    st.markdown("""
    This application uses a deep learning model trained on **109,309** OCT images
    to detect eye diseases:
    - **CNV** - Choroidal Neovascularization
    - **DME** - Diabetic Macular Edema
    - **DRUSEN** - Age-related Macular Degeneration
    - **NORMAL** - Healthy eyes
    """)

# =====================================================================
# MAIN INTERFACE
# =====================================================================
st.title(" Eye Disease Detection")
st.markdown("Detect eye diseases from OCT images using Deep Learning")
st.markdown("---")

# =====================================================================
# LOAD MODEL
# =====================================================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = "eye_disease_model_boosted_drusen.h5"
    try:
        model = keras.models.load_model(model_path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure the model file is in the same directory as this script.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

if model is None:
    st.stop()

# Class names and colors
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
class_colors = {
    'CNV': '#FF6B6B',
    'DME': '#4ECDC4',
    'DRUSEN': '#FFE66D',
    'NORMAL': '#95E1D3'
}

class_descriptions = {
    'CNV': 'Choroidal Neovascularization - Abnormal blood vessel growth',
    'DME': 'Diabetic Macular Edema - Fluid accumulation in macula',
    'DRUSEN': 'Age-related Macular Degeneration - Drusen deposits',
    'NORMAL': 'Healthy - No disease detected'
}

# =====================================================================
# PREPROCESSING FUNCTION
# =====================================================================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model inference"""
    # Convert to RGB if grayscale
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array
    image_array = np.array(image, dtype='float32')
    
    # Normalize (ImageNet normalization)
    image_array = image_array / 255.0
    
    return image_array

# =====================================================================
# PREDICTION FUNCTION
# =====================================================================
def predict(image_array):
    """Make prediction on image"""
    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Make prediction
    predictions = model.predict(image_batch, verbose=0)
    
    # Get probabilities
    probs = predictions[0]
    
    # Get class index and name
    class_idx = np.argmax(probs)
    class_name = class_names[class_idx]
    confidence = float(probs[class_idx])
    
    return probs, class_name, confidence, class_idx

# =====================================================================
# MAIN APP TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Info", "❓ Help"])

# =====================================================================
# TAB 1: PREDICTION
# =====================================================================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an OCT image (JPG, PNG, BMP)",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a retinal OCT scan image"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Get image info
            img_size = image.size
            st.info(f"📐 Image Size: {img_size[0]}×{img_size[1]} pixels")
            
            # Make prediction
            st.markdown("---")
            if st.button("🚀 Analyze Image", key="predict_button", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess
                    processed_image = preprocess_image(image)
                    
                    # Predict
                    probs, pred_class, confidence, class_idx = predict(processed_image)
                    
                    # Store results
                    st.session_state.probs = probs
                    st.session_state.pred_class = pred_class
                    st.session_state.confidence = confidence
                    st.session_state.class_idx = class_idx
                    st.session_state.image = image
                    st.session_state.prediction_made = True
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if 'prediction_made' in st.session_state and st.session_state.prediction_made:
            probs = st.session_state.probs
            pred_class = st.session_state.pred_class
            confidence = st.session_state.confidence
            
            # Main prediction box
            col_result = st.container()
            with col_result:
                # Color based on prediction
                color = class_colors.get(pred_class, '#667eea')
                st.markdown(f"""
                    <div style="background: {color}; padding: 20px; border-radius: 10px; text-align: center;">
                        <h2 style="color: white; margin: 0;">{pred_class}</h2>
                        <p style="color: white; font-size: 18px; margin: 10px 0;">
                            Confidence: <b>{confidence*100:.2f}%</b>
                        </p>
                        <p style="color: white; font-size: 14px; margin: 0;">
                            {class_descriptions[pred_class]}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.markdown("### 📈 Detailed Metrics")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            for i, (col, class_name, prob) in enumerate(zip(
                [metrics_col1, metrics_col2, metrics_col3, metrics_col4],
                class_names,
                probs
            )):
                with col:
                    st.metric(
                        class_name,
                        f"{prob*100:.1f}%",
                        delta=None,
                        delta_color="normal"
                    )
        else:
            st.info("📤 Upload an image and click 'Analyze Image' to see predictions")

# =====================================================================
# TAB 2: MODEL INFORMATION
# =====================================================================
with tab2:
    st.subheader("📊 Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Architecture Details")
        st.info("""
        **Base Model:** MobileNetV2
        - Pre-trained on ImageNet
        - Parameters: ~4.3M
        - Input: 224×224×3
        
        **Custom Layers:**
        - Global Average Pooling
        - Dense (512) + ReLU
        - Dropout (0.5)
        - Dense (256) + ReLU
        - Dropout (0.3)
        - Output Dense (4) + Softmax
        """)
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        st.success("""
        **Overall Accuracy:** 93.44%
        
        **Class-wise Performance:**
        - CNV: 93.25% F1
        - DME: 88.61% F1
        - DRUSEN: 75.56% F1
        - NORMAL: 97.18% F1
        
        **Training:**
        - Optimizer: Adam
        - Loss: Categorical Crossentropy
        - Epochs: 75
        - Batch Size: 32
        """)
    
    st.markdown("---")
    st.subheader("🎯 Class Distribution in Training Data")
    
    distribution = {
        'CNV': 34.3,
        'DME': 10.6,
        'DRUSEN': 8.1,
        'NORMAL': 47.0
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    colors_list = [class_colors[name] for name in distribution.keys()]
    ax1.pie(distribution.values(), labels=distribution.keys(), autopct='%1.1f%%',
            colors=colors_list, startangle=90, textprops={'fontweight': 'bold', 'fontsize': 11})
    ax1.set_title('Training Data Distribution', fontsize=13, fontweight='bold')
    
    # Bar chart
    ax2.bar(distribution.keys(), distribution.values(), color=colors_list, alpha=0.8, 
            edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Class Distribution Breakdown', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (name, val) in enumerate(distribution.items()):
        ax2.text(i, val + 1, f'{val}%', ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# =====================================================================
# TAB 3: HELP
# =====================================================================
with tab3:
    st.subheader("❓ Frequently Asked Questions")
    
    with st.expander("📤 How do I upload an image?", expanded=False):
        st.write("""
        1. Go to the **Prediction** tab
        2. Click on the upload area or drag & drop
        3. Select a retinal OCT scan image (JPG, PNG, or BMP)
        4. The image will be displayed for preview
        5. Click the **"Analyze Image"** button to get predictions
        """)
    
    with st.expander("🎯 What do the disease classes mean?", expanded=False):
        for class_name, description in class_descriptions.items():
            st.markdown(f"**{class_name}:** {description}")
    
    with st.expander("📊 How is accuracy calculated?", expanded=False):
        st.write("""
        The model's accuracy of 93.44% was calculated on the test set containing 15,963 images.
        
        This means that when given an unseen OCT image, the model correctly identifies the disease 
        in approximately 93 out of 100 cases.
        """)
    
    with st.expander("⚙️ What is the confidence threshold?", expanded=False):
        st.write("""
        The confidence threshold (in the sidebar) is the minimum probability required to display 
        a prediction result. For example:
        - If threshold is 0.7 and prediction confidence is 0.65, a warning will be shown
        - If threshold is 0.7 and prediction confidence is 0.85, the result is displayed normally
        
        Higher thresholds = more conservative predictions
        """)
    
    with st.expander("🖼️ What image formats are supported?", expanded=False):
        st.write("""
        The following image formats are supported:
        - **JPG/JPEG** - Most common format for medical images
        - **PNG** - Lossless format
        - **BMP** - Bitmap format
        
        **Recommended:** Use high-quality OCT scans (224×224 or larger)
        """)
    
    with st.expander("⚠️ Important Disclaimer", expanded=False):
        st.warning("""
        ⚠️ **MEDICAL DISCLAIMER:**
        
        This application is for **educational and research purposes only**.
        
        - **NOT** a substitute for professional medical diagnosis
        - Should **NOT** be used for clinical decision-making
        - Always consult with qualified ophthalmologists
        - Results should be validated by medical professionals
        
        The model's predictions are based on deep learning patterns and may not 
        be 100% accurate in all cases.
        """)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p><b>Eye Disease Detection</b> | By Daksh Chaurasia</p>
</div>
""", unsafe_allow_html=True)
