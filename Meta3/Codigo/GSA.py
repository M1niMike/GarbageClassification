import numpy as np
from Load_dataset import load_dataSet
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from SwarmPackagePy import gsa

#----FUNCTIONS----

def build_model(num_layers, num_neurons):
    model = models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    for _ in range(num_layers):
        model.add(layers.Dense(num_neurons, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def fitness_function(params):
    try:
        num_layers, num_neurons = map(int, params)
    except ValueError as e:
        print(f"Error converting to int: {e}")

    print(f"Number of layers: {num_layers}")
    print(f"Number of neurons: {num_neurons}")

    model = build_model(num_layers, num_neurons)
    
    print(model.summary())

    print("Training the model...")
    model.fit(images_train, labels_train_encoded, epochs=10, validation_split=0.2, batch_size=128, callbacks=[early_stopping])

    print("Evaluating performance on the test set...")
    _, accuracy = model.evaluate(images_test, labels_test_encoded, verbose=1)
    
    print(f"Accuracy on test set: {accuracy}")
    
    return 1 - accuracy

#----MODEL----

images_train, images_test, labels_train_encoded, labels_test_encoded, label_encoder = load_dataSet()

early_stopping = EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True)

num_classes = 5

#----GSA----
dimension = 2
num_agents = 5
lb = [4, 100] 
ub = [6, 200] 
GO = 3
iteration = 2

gsa_instance = gsa(num_agents, fitness_function, lb, ub, dimension, iteration, GO)

best_num_layers, best_num_neurons = gsa_instance.get_Gbest()

print(f"BNL:{best_num_layers} BNN:{best_num_neurons}")

best_pos_list = gsa_instance.get_Gbest()  
print("\nList best_pos:", best_pos_list)

rounded_best_pos = [round(value) for value in best_pos_list]
int_best_pos = [int(value) for value in rounded_best_pos]

print("\nList best_pos (rounded and converted to integers):", int_best_pos)
