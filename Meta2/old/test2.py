# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:08:39 2023

@author: mikae
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tensorflow.keras import models, layers
from pyswarms.single.global_best import GlobalBestPSO


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







#Carregar e pré-processar o conjunto de dados
images, encoded_labels = getDataset('C:/ISEC/1Sem/IC/TP/DataSet/Garbage Classification', 1000, 50, 50)

#Dividir o conjunto de dados em treinamento e teste
img_train, img_test, labels_train, labels_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

#Criar e ajustar o LabelEncoder
label_encoder = LabelEncoder() # Criar uma instância da classe, LabelEncoder é usado para converter rotulos de texto em numeros
labels_train_encoded = label_encoder.fit_transform(labels_train) # Essa linha está ajustando o LabelEncoder aos rótulos de treinamento (labels_train) e, ao mesmo tempo, transformando esses rótulos em representações numéricas.
labels_test_encoded = label_encoder.transform(labels_test) # Essa linha está ajustando o LabelEncoder aos rótulos de teste (labels_test) e, ao mesmo tempo, transformando esses rótulos em representações numéricas.

#----MODEL----

#Número de classes no seu conjunto de dados
num_classes = 5  # Substitua pelo número correto de classes

#Modelo sequencial
model = models.Sequential()

#Camadas do modelo
model.add(layers.Flatten(input_shape=(50, 50, 3)))
model.add(layers.Dense(100, activation='relu'))
#model.add(layers.Dense(50, activation='relu'))

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


#Treinar o modelo normal
model.fit(img_train, labels_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(img_test, labels_test)
print(f"\nTest accuracy: {test_acc}")

# Previsões no conjunto de teste
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

# Função para avaliar o modelo com hiperparâmetros específicos
def evaluate_model(params):
    num_classes = 5
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(50, 50, 3)))
    model.add(layers.Dense(int(params[0]), activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(img_train, labels_train_encoded, epochs=int(params[1]), batch_size=32, validation_split=0.2, verbose=0)
    _, accuracy = model.evaluate(img_test, labels_test_encoded, verbose=0)
    return 1 - accuracy  # PSO minimiza a função, queremos maximizar a precisão

# Define limites para os hiperparâmetros
bounds = (slice(1, 100, 1),  # Número de neurônios na camada oculta
          slice(1, 10, 1))   # Número de épocas de treinamento

# Define o otimizador PSO
optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options={'c1': 0.5, 'c2': 0.3, 'w':0.9})

# Executa o PSO para otimização
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 30, 'p': 1}
best_params, _ = optimizer.optimize(evaluate_model, bounds, 100, options=options, verbose=2)

# Melhores parâmetros encontrados pelo PSO
best_hidden_layer_size = int(best_params[0])
best_epochs = int(best_params[1])

# Cria e treina o modelo final com os melhores parâmetros
best_model = models.Sequential()
best_model.add(layers.Flatten(input_shape=(50, 50, 3)))
best_model.add(layers.Dense(best_hidden_layer_size, activation='relu'))
best_model.add(layers.Dense(num_classes, activation='softmax'))
best_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
best_model.fit(img_train, labels_train_encoded, epochs=best_epochs, batch_size=32, validation_split=0.2)

# Avalia o modelo no conjunto de teste
test_loss, test_acc = best_model.evaluate(img_test, labels_test_encoded)
print(f"\nTest accuracy: {test_acc}")

# Previsões no conjunto de teste
y_pred = best_model.predict(img_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Apanhar as labels das classes
class_labels = label_encoder.classes_

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(labels_test_encoded, y_pred_classes, labels=np.arange(len(class_labels)))

# Exibir a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
disp.plot(cmap='viridis', values_format='d')
plt.title('Confusion Matrix')
plt.show()

#---END_ORIGINAL_MODEL---

    