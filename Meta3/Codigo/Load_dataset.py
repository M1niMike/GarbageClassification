import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def is_rgb_image(img_path):
    img = Image.open(img_path)
    return img.mode == 'RGB'

def getDataset(caminho, maxImagens, altura, largura):
    imagens, rotulos = [], []
    contadores_classe = {}

    for pasta_classe in os.listdir(caminho):
        caminho_classe = os.path.join(caminho, pasta_classe)

        if os.path.isdir(caminho_classe):
            contadores_classe[pasta_classe] = 0

            for arquivo_img in os.listdir(caminho_classe):
                if contadores_classe[pasta_classe] >= maxImagens:
                    break

                caminho_img = os.path.join(caminho_classe, arquivo_img)

                if not is_rgb_image(caminho_img):
                    print(f"Imagem removida: {caminho_img} - Não está no formato RGB")
                    continue

                img = Image.open(caminho_img)
                img = img.resize((altura, largura))
                img_array = np.array(img) / 255.0

                imagens.append(img_array)
                rotulos.append(pasta_classe)
                contadores_classe[pasta_classe] += 1

    label_encoder = LabelEncoder()
    rotulos_codificados = label_encoder.fit_transform(rotulos)

    return np.array(imagens), np.array(rotulos_codificados)

def load_dataSet():
    images, encoded_labels = getDataset('C:/ISEC/1Sem/IC/TP/DataSet/Garbage Classification', 200, 200, 200)
    
    images_train, images_test, labels_train, labels_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
    
    label_encoder = LabelEncoder()
    labels_train_encoded = label_encoder.fit_transform(labels_train)
    labels_test_encoded = label_encoder.transform(labels_test)

    return images_train, images_test, labels_train_encoded, labels_test_encoded, label_encoder
