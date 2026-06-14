import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================
# KONFIGURASI
# =============================================
DATASET_DIR = 'dataset'
MODEL_DIR   = 'model'
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 20

os.makedirs(MODEL_DIR, exist_ok=True)

# =============================================
# DATA AUGMENTATION & PREPROCESSING
# =============================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("\n✅ Kelas yang ditemukan:", train_generator.class_indices)
NUM_CLASSES = len(train_generator.class_indices)

# =============================================
# ARSITEKTUR MODEL CNN
# =============================================
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    # Fully Connected
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================================
# CALLBACKS
# =============================================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'weather_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# =============================================
# TRAINING
# =============================================
print("\n🚀 Mulai Training...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =============================================
# SIMPAN LABEL KELAS
# =============================================
import json
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}
with open(os.path.join(MODEL_DIR, 'class_labels.json'), 'w') as f:
    json.dump(class_labels, f)
print("\n✅ Label kelas disimpan di model/class_labels.json")

# =============================================
# PLOT HASIL TRAINING
# =============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
plt.show()
print("\n✅ Grafik training disimpan di model/training_history.png")
print("\n🎉 Training selesai! Model disimpan di model/weather_model.h5")