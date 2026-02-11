"""
Eye Disease Detection - MobileNetV2 with Enhanced DRUSEN Detection
3-Phase Training with Boosted DRUSEN Class Weights and Segmentation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATASET STATISTICS & TRAINING PLAN
# ============================================================================
print("="*80)
print("EYE DISEASE DETECTION - MobileNetV2 WITH ENHANCED DRUSEN DETECTION")
print("="*80)

dataset_stats = {
    'train': {'CNV': 26218, 'DME': 8118, 'DRUSEN': 6206, 'NORMAL': 35973},
    'val': {'CNV': 7491, 'DME': 2319, 'DRUSEN': 1773, 'NORMAL': 10278},
    'test': {'CNV': 3746, 'DME': 1161, 'DRUSEN': 887, 'NORMAL': 5139}
}

print("\n📊 DATASET STATISTICS:")
print("-" * 80)
for split, classes in dataset_stats.items():
    total = sum(classes.values())
    print(f"\n{split.upper()} Split (Total: {total:,} images):")
    for class_name, count in classes.items():
        percentage = (count / total) * 100
        print(f"  {class_name:10s}: {count:6,} images ({percentage:6.2f}%)")

# ============================================================================
# CALCULATE BOOSTED CLASS WEIGHTS (Enhanced for DRUSEN)
# ============================================================================
print("\n⚖️  BOOSTED CLASS WEIGHT CALCULATION (for enhanced DRUSEN):")
print("-" * 80)

all_train_labels = []
for class_idx, class_name in enumerate(['CNV', 'DME', 'DRUSEN', 'NORMAL']):
    all_train_labels.extend([class_idx] * dataset_stats['train'][class_name])

class_weights = compute_class_weight('balanced', 
                                      classes=np.unique(all_train_labels),
                                      y=all_train_labels)

# Boost DRUSEN weight further for segmentation + boosting effect
class_weights[2] *= 2.0  # Double DRUSEN weight
class_weights[1] *= 1.3  # Increase DME weight slightly
class_weights[0] *= 1.1  # Slight increase for CNV

# Normalize weights
class_weights = class_weights / np.sum(class_weights) * len(class_weights)

class_weight_dict = dict(enumerate(class_weights))

print("\nBoosted Class Weights (for handling imbalance + segmentation):")
for idx, (class_name, weight) in enumerate(zip(['CNV', 'DME', 'DRUSEN', 'NORMAL'], class_weights)):
    print(f"  {class_name:10s}: {weight:.4f}")

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
print("\n" + "="*80)
print("🎯 TRAINING CONFIGURATION (ENHANCED)")
print("="*80)

# Use segmented dataset if available, otherwise use original
SEGMENTED_DATASET = r'e:\Eye Disease\Dataset_Segmented'
ORIGINAL_DATASET = r'e:\Eye Disease\Dataset - train+val+test'

if os.path.exists(SEGMENTED_DATASET):
    dataset_path = SEGMENTED_DATASET
    print("✅ Using SEGMENTED dataset for enhanced DRUSEN detection")
else:
    dataset_path = ORIGINAL_DATASET
    print("⚠️  Segmented dataset not found. Using original dataset.")
    print("   To improve DRUSEN: Run segment_drusen.py first")

# Force segmented dataset path
dataset_path = SEGMENTED_DATASET

CONFIG = {
    # Data Configuration
    'dataset_path': dataset_path,
    'image_size': (224, 224),
    'batch_size': 32,
    
    # Phase 1: Transfer Learning
    'phase1': {
        'epochs': 25,
        'learning_rate': 1e-3,
        'description': 'Transfer Learning - Frozen MobileNetV2 backbone'
    },
    
    # Phase 2: Fine-tuning
    'phase2': {
        'epochs': 30,
        'learning_rate': 1e-5,
        'unfreeze_layers': 100,
        'description': 'Fine-tuning - Unfrozen top layers'
    },
    
    # Phase 3: Extended Fine-tuning
    'phase3': {
        'epochs': 25,  # Increased from 20 for DRUSEN focus
        'learning_rate': 1e-6,
        'unfreeze_layers': 150,
        'description': 'Extended Fine-tuning - Enhanced DRUSEN focus'
    },
    
    # Augmentation (aggressive for minority classes)
    'augmentation': {
        'rotation_range': 30,  # Increased
        'width_shift_range': 0.3,  # Increased
        'height_shift_range': 0.3,  # Increased
        'zoom_range': 0.3,  # Increased
        'shear_range': 0.2,  # Increased
        'horizontal_flip': True,
        'vertical_flip': False,
        'fill_mode': 'nearest'
    }
}

print("\n📋 PHASE 1 - TRANSFER LEARNING:")
print(f"  Epochs: {CONFIG['phase1']['epochs']}")
print(f"  Learning Rate: {CONFIG['phase1']['learning_rate']}")

print("\n📋 PHASE 2 - FINE-TUNING:")
print(f"  Epochs: {CONFIG['phase2']['epochs']}")
print(f"  Learning Rate: {CONFIG['phase2']['learning_rate']}")

print("\n📋 PHASE 3 - EXTENDED FINE-TUNING (DRUSEN FOCUS):")
print(f"  Epochs: {CONFIG['phase3']['epochs']}")
print(f"  Learning Rate: {CONFIG['phase3']['learning_rate']}")
print(f"  Unfreeze Layers: {CONFIG['phase3']['unfreeze_layers']}")

total_epochs = CONFIG['phase1']['epochs'] + CONFIG['phase2']['epochs'] + CONFIG['phase3']['epochs']
print(f"\n⏱️  TOTAL TRAINING TIME:")
print(f"  Total Epochs: {total_epochs}")
print(f"  Estimated: 15-20 hours (GPU)")

# ============================================================================
# DATA LOADING WITH AGGRESSIVE AUGMENTATION
# ============================================================================
print("\n" + "="*80)
print("📦 LOADING AND PREPARING DATA")
print("="*80)

def create_data_generators():
    """Create data generators with aggressive augmentation"""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=CONFIG['augmentation']['rotation_range'],
        width_shift_range=CONFIG['augmentation']['width_shift_range'],
        height_shift_range=CONFIG['augmentation']['height_shift_range'],
        zoom_range=CONFIG['augmentation']['zoom_range'],
        shear_range=CONFIG['augmentation']['shear_range'],
        horizontal_flip=CONFIG['augmentation']['horizontal_flip'],
        vertical_flip=CONFIG['augmentation']['vertical_flip'],
        fill_mode=CONFIG['augmentation']['fill_mode']
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_test_datagen

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
print("\n" + "="*80)
print("🧠 MODEL ARCHITECTURE")
print("="*80)

def create_model_with_segmentation_attention():
    """MobileNetV2 with enhanced architecture for DRUSEN detection"""
    
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    
    # Enhanced global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Enhanced dense layers for better DRUSEN discrimination
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)  # Increased dropout
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)  # Increased dropout
    
    x = layers.Dense(128, activation='relu')(x)  # Additional layer
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(4, activation='softmax', name='classification')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    print("\n✅ Enhanced Model Architecture Created:")
    print(f"  Base Model: MobileNetV2 (ImageNet pre-trained)")
    print(f"  Input Shape: 224 x 224 x 3")
    print(f"  Output Classes: 4")
    print(f"  Total Parameters: {model.count_params():,}")
    
    return model, base_model

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_phase1(model, base_model, train_gen, val_gen, train_path, val_path):
    """Phase 1: Transfer Learning"""
    
    print("\n" + "="*80)
    print(f"🚀 PHASE 1 - TRANSFER LEARNING ({CONFIG['phase1']['epochs']} epochs)")
    print("="*80)
    
    base_model.trainable = False
    
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['phase1']['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint('best_phase1_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    train_steps = sum(dataset_stats['train'].values()) // CONFIG['batch_size']
    val_steps = sum(dataset_stats['val'].values()) // CONFIG['batch_size']
    
    history_phase1 = model.fit(
        train_gen, steps_per_epoch=train_steps, epochs=CONFIG['phase1']['epochs'],
        validation_data=val_gen, validation_steps=val_steps,
        class_weight=class_weight_dict, callbacks=callbacks, verbose=1
    )
    
    return history_phase1

def train_phase2(model, base_model, train_gen, val_gen, history_phase1):
    """Phase 2: Fine-tuning"""
    
    print("\n" + "="*80)
    print(f"🔄 PHASE 2 - FINE-TUNING ({CONFIG['phase2']['epochs']} epochs)")
    print("="*80)
    
    base_model.trainable = True
    for layer in base_model.layers[:-CONFIG['phase2']['unfreeze_layers']]:
        layer.trainable = False
    
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['phase2']['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-8),
        keras.callbacks.ModelCheckpoint('best_phase2_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    train_steps = sum(dataset_stats['train'].values()) // CONFIG['batch_size']
    val_steps = sum(dataset_stats['val'].values()) // CONFIG['batch_size']
    
    history_phase2 = model.fit(
        train_gen, steps_per_epoch=train_steps, epochs=CONFIG['phase2']['epochs'],
        validation_data=val_gen, validation_steps=val_steps,
        class_weight=class_weight_dict, callbacks=callbacks, verbose=1
    )
    
    return history_phase2

def train_phase3(model, base_model, train_gen, val_gen, history_phase2):
    """Phase 3: Extended Fine-tuning (DRUSEN Focus)"""
    
    print("\n" + "="*80)
    print(f"🔥 PHASE 3 - EXTENDED FINE-TUNING ({CONFIG['phase3']['epochs']} epochs) - DRUSEN FOCUS")
    print("="*80)
    
    base_model.trainable = True
    for layer in base_model.layers[:-CONFIG['phase3']['unfreeze_layers']]:
        layer.trainable = False
    
    optimizer = keras.optimizers.Adam(learning_rate=CONFIG['phase3']['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-9),
        keras.callbacks.ModelCheckpoint('best_phase3_model.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    train_steps = sum(dataset_stats['train'].values()) // CONFIG['batch_size']
    val_steps = sum(dataset_stats['val'].values()) // CONFIG['batch_size']
    
    history_phase3 = model.fit(
        train_gen, steps_per_epoch=train_steps, epochs=CONFIG['phase3']['epochs'],
        validation_data=val_gen, validation_steps=val_steps,
        class_weight=class_weight_dict, callbacks=callbacks, verbose=1
    )
    
    return history_phase3

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
if __name__ == "__main__":
    
    train_path = os.path.join(CONFIG['dataset_path'], 'train')
    val_path = os.path.join(CONFIG['dataset_path'], 'val')
    test_path = os.path.join(CONFIG['dataset_path'], 'test')
    
    print("\n⏳ Creating data generators...")
    train_datagen, val_test_datagen = create_data_generators()
    
    train_generator = train_datagen.flow_from_directory(
        train_path, target_size=CONFIG['image_size'], batch_size=CONFIG['batch_size'],
        class_mode='categorical', shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_path, target_size=CONFIG['image_size'], batch_size=CONFIG['batch_size'],
        class_mode='categorical', shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_path, target_size=CONFIG['image_size'], batch_size=CONFIG['batch_size'],
        class_mode='categorical', shuffle=False
    )
    
    print("✅ Data generators created!")
    
    model, base_model = create_model_with_segmentation_attention()
    
    # Training phases
    history_phase1 = train_phase1(model, base_model, train_generator, val_generator, train_path, val_path)
    history_phase2 = train_phase2(model, base_model, train_generator, val_generator, history_phase1)
    history_phase3 = train_phase3(model, base_model, train_generator, val_generator, history_phase2)
    
    model.save('eye_disease_final_model_drusen_enhanced.h5')
    print("\n✅ Model saved: eye_disease_final_model_drusen_enhanced.h5")
    
    # Evaluate
    print("\n" + "="*80)
    print("📊 EVALUATION ON TEST SET")
    print("="*80)
    
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    phase1_len = len(history_phase1.history['accuracy'])
    phase2_len = len(history_phase2.history['accuracy'])
    phase1_offset = phase1_len
    phase2_offset = phase1_len + phase2_len
    
    axes[0].plot(history_phase1.history['accuracy'], label='Phase 1 Train')
    axes[0].plot(history_phase1.history['val_accuracy'], label='Phase 1 Val')
    axes[0].plot(range(phase1_offset, phase1_offset + phase2_len),
                 history_phase2.history['accuracy'], label='Phase 2 Train')
    axes[0].plot(range(phase1_offset, phase1_offset + phase2_len),
                 history_phase2.history['val_accuracy'], label='Phase 2 Val')
    axes[0].plot(range(phase2_offset, phase2_offset + len(history_phase3.history['accuracy'])),
                 history_phase3.history['accuracy'], label='Phase 3 Train')
    axes[0].plot(range(phase2_offset, phase2_offset + len(history_phase3.history['val_accuracy'])),
                 history_phase3.history['val_accuracy'], label='Phase 3 Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training Accuracy - Enhanced DRUSEN')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history_phase1.history['loss'], label='Phase 1 Train')
    axes[1].plot(history_phase1.history['val_loss'], label='Phase 1 Val')
    axes[1].plot(range(phase1_offset, phase1_offset + phase2_len),
                 history_phase2.history['loss'], label='Phase 2 Train')
    axes[1].plot(range(phase1_offset, phase1_offset + phase2_len),
                 history_phase2.history['val_loss'], label='Phase 2 Val')
    axes[1].plot(range(phase2_offset, phase2_offset + len(history_phase3.history['loss'])),
                 history_phase3.history['loss'], label='Phase 3 Train')
    axes[1].plot(range(phase2_offset, phase2_offset + len(history_phase3.history['val_loss'])),
                 history_phase3.history['val_loss'], label='Phase 3 Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss - Enhanced DRUSEN')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_drusen_enhanced.png', dpi=300, bbox_inches='tight')
    print("✅ Training history saved: training_history_drusen_enhanced.png")
    
    print("\n" + "="*80)
    print("✨ TRAINING COMPLETE!")
    print("="*80)
