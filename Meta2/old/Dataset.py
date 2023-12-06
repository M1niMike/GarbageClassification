import os
import numpy as np
import matplotlib.pyplot as plt
import SwarmPackagePy as SPP
import pyswarms as ps


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from PIL import Image

from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from SwarmPackagePy import gsa
from SwarmPackagePy import testFunctions
#from SwarmPackagePy import *



#----FUNCTIONS----

dimension = 2

def gravitational_force(m1, m2, r, G):
    return G * (m1 * m2) / r**2

def fitness_function(position):
    """
    Evaluate the fitness of a solution for the GSA optimization based on gravitational forces.

    Parameters:
    - position (numpy array): The position of a solution in the search space.
    - dimensionality (int): The dimensionality of the search space.

    Returns:
    - fitness (float): The fitness value of the solution.
    """
    G = 6.674 * 10**-11  # gravitational constant

    # Assuming each element of the position array represents a mass
    masses = position

    # Calculate total gravitational force among masses
    total_force = 0.0
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                r = np.abs(i - j)  # Simple distance assumption for illustration
                total_force += gravitational_force(masses[i], masses[j], r, G)

    # Fitness is the negative of the total gravitational force (since we typically maximize fitness)
    fitness = -total_force

    return fitness



def is_rgb_image(img_path):
    img = Image.open(img_path)
    return img.mode == 'RGB'


def getDataset(caminho, maxImagens, altura, largura):
    #Inicializar listas vazias para imagens e rótulos
    imagens = []
    rotulos = []
    
    #Contador para cada classe
    contadores_classe = {}

    for pasta_classe in os.listdir(caminho):
        caminho_classe = os.path.join(caminho, pasta_classe)
        
        #Verificar se é um diretório
        if os.path.isdir(caminho_classe):
            contadores_classe[pasta_classe] = 0  #Inicializar contador para a classe
            
            #Iterar por cada imagem na pasta da classe
            for arquivo_img in os.listdir(caminho_classe):
                if contadores_classe[pasta_classe] >= maxImagens:
                    break  #Interromper se o número máximo de imagens for atingido para esta classe
                
                caminho_img = os.path.join(caminho_classe, arquivo_img)
              
                
                #Verificar se a imagem é RGB
                if not is_rgb_image(caminho_img):
                    print(f"Imagem removida: {caminho_img} - Não está no formato RGB")
                    continue
                
                #Abrir e pré-processar a imagem
                img = Image.open(caminho_img)
                
                img = img.resize((altura, largura))  # Redimensionar todas as imagens para as mesmas dimensões
                
                img_array = np.array(img) / 255.0  # Normalizar os valores dos pixels
                            
                #Adicionar a imagem e o rótulo às listas
                imagens.append(img_array)
                rotulos.append(pasta_classe)
                
                #Incrementar o contador para a classe
                contadores_classe[pasta_classe] += 1

    #Converter rótulos para inteiros usando o LabelEncoder
    label_encoder = LabelEncoder()
    rotulos_codificados = label_encoder.fit_transform(rotulos)

    #Converter listas para arrays NumPy após redimensionar todas as imagens para as mesmas dimensões
    imagens = np.array(imagens)
    rotulos_codificados = np.array(rotulos_codificados)
    
    return imagens, rotulos_codificados


def load_dataSet():
    #Carregar e pré-processar o conjunto de dados
    images, encoded_labels = getDataset('C:/ISEC/1Sem/IC/TP/DataSet/Garbage Classification', 1500, 100, 100)
    
    #Dividir o conjunto de dados em treinamento e teste
    images_train, images_test, labels_train, labels_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
    
    #Criar e ajustar o LabelEncoder
    label_encoder = LabelEncoder() # Criar uma instância da classe, LabelEncoder é usado para converter rotulos de texto em numeros
    labels_train_encoded = label_encoder.fit_transform(labels_train) # Essa linha está ajustando o LabelEncoder aos rótulos de treinamento (labels_train) e, ao mesmo tempo, transformando esses rótulos em representações numéricas.
    labels_test_encoded = label_encoder.transform(labels_test) # Essa linha está ajustando o LabelEncoder aos rótulos de teste (labels_test) e, ao mesmo tempo, transformando esses rótulos em representações numéricas.

    return images_train, images_test, labels_train_encoded, labels_test_encoded, label_encoder

def Original_model(img_train, img_test, labels_train, labels_test, label_encoder):
    
    #----MODEL----
    
    #Número de classes no seu conjunto de dados
    num_classes = 5  # Substitua pelo número correto de classes
    
    #Modelo sequencial
    model = models.Sequential()
    
    #Camadas do modelo
    model.add(layers.Flatten(input_shape=(100, 100, 3)))
    model.add(layers.Dense(55, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    
    #Camada de saída
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    #Compilar o modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy para rótulos inteiros
                  metrics=['accuracy'])
    
    # Display the model summary to see the total number of parameters
    model.summary()
    
    # Get the total number of weights
    total_weights = model.count_params()
    
    print("Total number of weights:", total_weights)
    
    
    #train model
    model.fit(img_train, labels_train, epochs=100, batch_size=32, validation_split=0.2)
    
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(img_test, labels_test)
    print(f"\nTest accuracy: {test_acc}")
    
    # preditions of the model
    y_pred = model.predict(img_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Apanhar as labels das classes
    class_labels = label_encoder.classes_
    
    # Calcular a matriz de confusão
    conf_matrix = confusion_matrix(labels_test, y_pred_classes, labels=np.arange(len(class_labels)))
    
    # Exibir a matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap='viridis', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()
    
    #---END_ORIGINAL_MODEL---
    
def GSA_Model(num_classes,img_train, img_test, labels_train, labels_test, label_encoder):
    #----GSA----

    # Obtém a dimensão dos parâmetros do modelo
    n_inputs = 64 * 64 * 3
    n_hidden = 55
    n_classes = 5

    #Pametros do GSA 
    total_parameters = n_inputs * n_hidden + n_hidden + n_hidden * n_classes + n_classes
    print("Total Parameters in Neural Network:", total_parameters)

    function_gsa = SPP.testFunctions.ackley_function


    # GSA parameters
    global dimension
    num_agents = 100
    lb = [1, 50] 
    ub = [5, 500] 
    GO = 3
    iteration = 200

    # GSA instance with the scaled function
    gsa_instance = SPP.gsa(num_agents, function_gsa, lb, ub, dimension, iteration, GO)


    #---NEW_MODEL_WITH_GSA---

    # get the best positions of gsa
    best_pos_list = gsa_instance.get_Gbest()  # Certifique-se de que este método retorna uma lista de valores reais

    print("\nList best_pos:", best_pos_list)

    rounded_best_pos = [round(value) for value in best_pos_list]
    int_best_pos = [int(value) for value in rounded_best_pos]

    print("\nList best_pos (rounded and converted to integers):", int_best_pos)


    #new model
    new_model = models.Sequential()

    new_model.add(layers.Flatten(input_shape=(100, 100, 3)))

    # Add the specified number of layers with the specified number of neurons
    for i in range(int_best_pos[0]):
        new_model.add(layers.Dense(int_best_pos[1], activation='relu'))

    #Camada de saída
    new_model.add(layers.Dense(num_classes, activation='softmax'))

    #Compilar o modelo
    new_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy para rótulos inteiros
                  metrics=['accuracy'])

    # Display the model summary to see the total number of parameters
    new_model.summary()

    # Get the total number of weights
    total_weights = new_model.count_params()

    print("Total number of weights:", total_weights)


    #Train the new model
    new_model.fit(img_train, labels_train, epochs=100, batch_size=32, validation_split=0.2)


    # Evaluate the model on the test set
    test_loss, test_acc = new_model.evaluate(img_test, labels_test)
    print(f"\nTest accuracy: {test_acc}")

    # Previsões no conjunto de teste
    y_pred = new_model.predict(img_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Apanhar as labels das classes
    class_labels = label_encoder.classes_

    # Calcular a matriz de confusão
    conf_matrix = confusion_matrix(labels_test, y_pred_classes, labels=np.arange(len(class_labels)))

    # Exibir a matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap='viridis', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()
    
    

def main():
    num_classes = 5
    img_train, img_test, labels_train, labels_test, label_encoder = load_dataSet()
    
    Original_model(img_train, img_test, labels_train, labels_test, label_encoder)
    
    GSA_Model(num_classes,img_train, img_test, labels_train, labels_test, label_encoder)
    

if __name__=="__main__":
    main()
    