import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from Load_dataset import load_dataSet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
img_train, img_test, labels_train, labels_test, label_encoder = load_dataSet()

# Constants
num_classes = 5
input_shape = (200, 200, 3)
epochs = 10
batch_size = 128

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

# Flatten and dense layers
flatten_layer = layers.Flatten()
dense_layer_1 = layers.Dense(128, activation='relu')
dense_layer_2 = layers.Dense(194, activation='relu')
dense_layer_3 = layers.Dense(194, activation='relu')
# dense_layer_4 = layers.Dense(179, activation='relu')
# dense_layer_5 = layers.Dense(179, activation='relu')
# dense_layer_6 = layers.Dense(179, activation='relu')
prediction_layer = layers.Dense(num_classes, activation='softmax')

# Build the model
model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_layer_3,
    # dense_layer_4,
    # dense_layer_5,
    # dense_layer_6,
    prediction_layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(img_train, labels_train, epochs=epochs, validation_split=0.2, batch_size=batch_size, callbacks=[es])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(img_test, labels_test)
print(f"\nTest accuracy: {test_acc}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model
model.save('best_model.h5')
model.save('best_model.keras')
