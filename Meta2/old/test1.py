# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 16:25:56 2023

@author: mikae
"""

# -*- coding: utf-8 -*-
"""IC-DATASET TEST - 1º Tentativa"""

import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pygmo as pg
import matplotlib.pyplot as plt


#----FUNCTIONS----
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

#----MAIN----
#Carregar e pré-processar o conjunto de dados
images, encoded_labels = getDataset('C:/ISEC/1Sem/IC/TP/DataSet/Garbage Classification', 1300, 100, 100)

#Dividir o conjunto de dados em treinamento e teste
images_train, images_test, labels_train, labels_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

#Criar e ajustar o LabelEncoder
label_encoder = LabelEncoder() # Criar uma instância da classe, LabelEncoder é usado para converter rotulos de texto em numeros
labels_train_encoded = label_encoder.fit_transform(labels_train) # Essa linha está ajustando o LabelEncoder aos rótulos de treinamento (labels_train) e, ao mesmo tempo, transformando esses rótulos em representações numéricas.
labels_test_encoded = label_encoder.transform(labels_test) # Essa linha está ajustando o LabelEncoder aos rótulos de teste (labels_test) e, ao mesmo tempo, transformando esses rótulos em representações numéricas.

#Número de classes no seu conjunto de dados
num_classes = len(np.unique(labels_train_encoded))

#definir os limites dos hiperparametros para o GSA
hyperparameter_bound = [(10,100), (10,100)] #limites para o número de unidades nas camadas Dense

#criar uma instancia do problema de otimização
problem = pg.problem(pg.declare_problem(fitness_function, bounds= hyperparameter_bound))

#definir os parametros do GSA e executar a otimização
algorithm = pg.algorithm(pg.gsa(gen=50)) #50 gerações, ajustar conforme for preciso
pop = pg.population(problem, size=10) #tamanho da população

pop = algorithm.evolve(pop)

#obter melhor solução
best_solution = pop.get_f()[pop.best_idx()]

#exibir os melhores parametros
best_hp = best_solution.get_x()
print("Best hyperparameters:", best_hp)

#criar um modelo final
final_model = models.sequential()
final_model.add(layers.Flatten(input_shape=(100,100,3)))
final_model.add(layers.Dense(int(best_hp[0]), activation='relu'))
final_model.add(layers.Dense(int(best_hp[1]), activation='relu'))
final_model.add(layers.Dense(num_classes, activation='softmax'))

#compilar o modelo final
final_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

#treinar modelo final
final_model.fit(images_train, labels_train, epochs=10, batch_size=32, validation_split=0.2)

#Avaliar o modelo no conjunto de teste
test_loss, test_acc = final_model.evaluate(images_test, labels_test_encoded)
print(f"\nTest accuracy with optimized hyperparameters: {test_acc}")

#Previsões no conjunto de teste
y_pred = final_model.predict(images_test)
y_pred_classes = np.argmax(y_pred, axis=1)

#Apanhar as labels das classes
class_labels = label_encoder.classes_

#Calcular a matriz de confusão
conf_matrix = confusion_matrix(labels_test_encoded, y_pred_classes, labels=np.arange(len(class_labels)))

#Exibir a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
disp.plot(cmap='viridis', values_format='d')
plt.title('Confusion Matrix')
plt.show()