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
from keras.layers import Conv2D, Flatten
from tensorflow.keras.utils import to_categorical

from SwarmPackagePy import gsa
from SwarmPackagePy import testFunctions
#from SwarmPackagePy import *



#----FUNCTIONS----

# Função de aptidão
def fitness_function(params):
    try:
        num_layers = int(params[0])
        num_neurons = int(params[1])
    except ValueError as e:
        print(f"Error converting to int: {e}")

    # Construir modelo de rede neural
    print(f"Number of layers: {num_layers}")
    print(f"Number of neurons: {num_neurons}")

    new_model = models.Sequential()

    # Adicione camadas convolucionais antes das camadas densas
    new_model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(100, 100, 3), activation='relu'))
    new_model.add(Flatten())

    # Adicione a primeira camada densa com achatamento
    new_model.add(layers.Dense(num_neurons, activation='relu'))

    for _ in range(num_layers - 1):
        new_model.add(layers.Dense(num_neurons, activation='relu'))

    new_model.add(layers.Dense(num_classes, activation='softmax'))

    new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Antes de treinar o modelo
    img_train_normalized = img_train / 255.0
    img_test_normalized = img_test / 255.0

    # Antes de treinar o modelo
    labels_train_encoded = to_categorical(labels_train, num_classes=num_classes)
    labels_test_encoded = to_categorical(labels_test, num_classes=num_classes)
    
    # Treinar modelo
    print("Training the model...")
    new_model.fit(img_train_normalized, labels_train_encoded, epochs=5, batch_size=64, verbose=1)

   # Avaliar desempenho no conjunto de teste
    print("Evaluating performance on the test set...")
    _, accuracy = new_model.evaluate(img_test_normalized, labels_test_encoded, verbose=1)
    
    print(f"Accuracy on test set: {accuracy}")
    
    # Inverter o valor para minimização
    return -accuracy






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
    images, encoded_labels = getDataset('C:/ISEC/1Sem/IC/TP/DataSet/Garbage Classification', 1000, 100, 100)
    
    #Dividir o conjunto de dados em treinamento e teste
    images_train, images_test, labels_train, labels_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
    
    #Criar e ajustar o LabelEncoder
    label_encoder = LabelEncoder() # Criar uma instância da classe, LabelEncoder é usado para converter rotulos de texto em numeros
    labels_train_encoded = label_encoder.fit_transform(labels_train) # Essa linha está ajustando o LabelEncoder aos rótulos de treinamento (labels_train) e, ao mesmo tempo, transformando esses rótulos em representações numéricas.
    labels_test_encoded = label_encoder.transform(labels_test) # Essa linha está ajustando o LabelEncoder aos rótulos de teste (labels_test) e, ao mesmo tempo, transformando esses rótulos em representações numéricas.

    return images_train, images_test, labels_train_encoded, labels_test_encoded, label_encoder

    

img_train, img_test, labels_train, labels_test, label_encoder = load_dataSet()

#----MODEL----

#Número de classes no seu conjunto de dados
num_classes = 5  # Substitua pelo número correto de classes

#----GSA----
function_gsa = SPP.testFunctions.ackley_function

# GSA parameters
dimension = 2
num_agents = 3
lb = [1, 10] 
ub = [5, 350] 
GO = 10
iteration = 2

# GSA instance with the scaled function
gsa_instance = SPP.gsa(num_agents, fitness_function, lb, ub, dimension, iteration, GO)


# get the best positions of gsa
best_pos_list = gsa_instance.get_Gbest()  # Certifique-se de que este método retorna uma lista de valores reais

print("\nList best_pos:", best_pos_list)

rounded_best_pos = [round(value) for value in best_pos_list]
int_best_pos = [int(value) for value in rounded_best_pos]

print("\nList best_pos (rounded and converted to integers):", int_best_pos)


# #Modelo sequencial
# model = models.Sequential()

# #Camadas do modelo
# model.add(layers.Flatten(input_shape=(100, 100, 3)))
# model.add(layers.Dense(55, activation='relu'))
# model.add(layers.Dense(25, activation='relu'))

# #Camada de saída
# model.add(layers.Dense(num_classes, activation='softmax'))

# #Compilar o modelo
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy para rótulos inteiros
#               metrics=['accuracy'])

# # Display the model summary to see the total number of parameters
# model.summary()

# # Get the total number of weights
# total_weights = model.count_params()

# print("Total number of weights:", total_weights)


# #Treinar o modelo normal
# model.fit(img_train, labels_train, epochs=10, batch_size=32, validation_split=0.2)

# # Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(img_test, labels_test)
# print(f"\nTest accuracy: {test_acc}")

# # Previsões no conjunto de teste
# y_pred = model.predict(img_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Apanhar as labels das classes
# class_labels = label_encoder.classes_

# # Calcular a matriz de confusão
# conf_matrix = confusion_matrix(labels_test, y_pred_classes, labels=np.arange(len(class_labels)))

# # Exibir a matriz de confusão
# disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
# disp.plot(cmap='viridis', values_format='d')
# plt.title('Confusion Matrix')
# plt.show()

# #---END_ORIGINAL_MODEL---