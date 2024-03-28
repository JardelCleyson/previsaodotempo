import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import cv2
from cv2 import IMREAD_GRAYSCALE

def carregar_dados(diretorio_raiz):
    imagens = []
    rotulos = []
    for rotulo, classe in enumerate(['semChuva', 'comChuva']):
        diretorio_classe = os.path.join(diretorio_raiz, classe)
        for nome_imagem in os.listdir(diretorio_classe):
            caminho_imagem = os.path.join(diretorio_classe, nome_imagem)
            imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)  # Carregar em escala de cinza
            if imagem is not None:
                imagem_redimensionada = cv2.resize(imagem, (100, 100))
                imagens.append(imagem_redimensionada)
                rotulos.append(rotulo)
    return np.array(imagens), np.array(rotulos)

def treinar_modelo(diretorio_treinamento='./dataset'):
    imagens_treino, rotulos_treino = carregar_dados(diretorio_treinamento)
    X_train, X_test, y_train, y_test = train_test_split(imagens_treino, rotulos_treino, test_size=0.8, random_state=42)# Dividir os dados em conjunto de treinamento e teste
    
    # Redimensionar os arrays de imagens para duas dimensões
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    modelo = SVC(kernel='linear', random_state=42)# Inicializar o classificador (SVM neste exemplo)
    modelo.fit(X_train, y_train)
    precisao = modelo.score(X_test, y_test)
    print("Precisão do modelo:", precisao)
    return modelo

def salvar_modelo(modelo, nome_arquivo='modelo_chuva.pkl'):
    joblib.dump(modelo, nome_arquivo)
    print("Modelo salvo com sucesso!")

def main():
    modelo = treinar_modelo()
    salvar_modelo(modelo)

if __name__ == "__main__":
    main()



