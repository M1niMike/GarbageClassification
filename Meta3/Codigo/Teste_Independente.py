import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Load_dataset import load_dataSet
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load dataset
img_train, img_test, labels_train, labels_test, label_encoder = load_dataSet()

# Load the saved model
best_model = load_model('best_model.keras')  # Replace with the actual name of your saved model file

print(best_model.summary())
# Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(img_test, labels_test)
print(f"\nTest accuracy: {test_acc}")

# Predictions on the test set
predictions = best_model.predict(img_test)
predicted_labels = np.argmax(predictions, axis=1)

# Convert encoded labels back to original labels
predicted_labels_original = label_encoder.inverse_transform(predicted_labels)
true_labels_original = label_encoder.inverse_transform(labels_test)

# Calculate confusion matrix
cm = confusion_matrix(true_labels_original, predicted_labels_original, labels=label_encoder.classes_)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='viridis', values_format='d')
plt.show()