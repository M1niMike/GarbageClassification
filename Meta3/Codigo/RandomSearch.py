# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:03:08 2023

@author: BrunoOliveira
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from LoadDataSet import load_dataSet
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load and preprocess the dataset
images_train, images_test, labels_train_encoded, labels_test_encoded, label_encoder = load_dataSet()

num_classes = 5
input_shape = (200, 200, 3)

# Define the model
def create_model(num_layers, num_neurons):
    model = Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    
    for _ in range(num_layers - 1):
        model.add(layers.Dense(num_neurons, activation='relu'))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Create a KerasClassifier with your model-building function
keras_classifier = KerasClassifier(build_fn=create_model, verbose=0)

# Define parameters for the random search
param_dist = {
    'num_layers': [1,2,3,4,5],  # Number of layers
    'num_neurons': [10,50,100,200] # Number of neurons
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random search
random_search = RandomizedSearchCV(estimator=keras_classifier, param_distributions=param_dist, scoring='accuracy', cv=cv, n_iter=25)
random_search.fit(images_train, labels_train_encoded)

# Get the best parameters and score from random search
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print(f'Best Hyperparameters: {best_params_random}')
print(f'Best Cross-Validated Accuracy: {best_score_random}')




