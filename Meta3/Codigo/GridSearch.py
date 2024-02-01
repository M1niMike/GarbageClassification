# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 00:24:41 2023

@author: BrunoOliveira
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from LoadDataSet import load_dataSet
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load the dataset
images_train, images_test, labels_train_encoded, labels_test_encoded, label_encoder = load_dataSet()

# Define constants
num_classes = 5
input_shape = (200, 200, 3)

# Define the model
def create_model(num_layers, num_neurons):
    print(f'Creating model with {num_layers} layers and {num_neurons} neurons...')
    model = Sequential()
    model.add(layers.Flatten(input_shape=input_shape))

    for _ in range(num_layers - 1):
        model.add(layers.Dense(num_neurons, activation='relu'))

    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Create a KerasClassifier with your model-building function
keras_classifier = KerasClassifier(build_fn=create_model, verbose=0)

# Define parameters for the grid search
param_grid = {
    'num_layers': list(range(1, 6)),  # Number of layers from 1 to 5
    'num_neurons': list(range(10, 201))  # Number of neurons from 10 to 200
}

# Define callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, scoring='accuracy', cv=cv, verbose=1)
grid_search.fit(images_train, labels_train_encoded, validation_split = 0.2 ,callbacks=[early_stopping, model_checkpoint])

# Get the best parameters and score from grid search
best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

print(f'Best Hyperparameters: {best_params_grid}')
print(f'Best Cross-Validated Accuracy: {best_score_grid}')
