import numpy as np
import cv2
import joblib
from treino import treinar_modelo
from sklearn.svm import SVC


def carregar_modelo(nome_arquivo='./modelo_chuva.pkl'):
    # Carregar o modelo treinado
    modelo = joblib.load(nome_arquivo)
    return modelo

def carregar_imagem(caminho_imagem):
    # Carregar a imagem
    imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)  # Carregar em escala de cinza
    if imagem is not None:
        # Redimensionar a imagem para o tamanho esperado pelo modelo
        imagem_redimensionada = cv2.resize(imagem, (100, 100))
        return imagem_redimensionada
    else:
        print("Erro ao carregar a imagem.")
        return None

def prever_possibilidade_chuva(modelo, imagem):
    # Redimensionar a imagem para o formato esperado pelo modelo
    imagem_redimensionada = imagem.reshape(1, -1)
    # Calcular a distância assinada de cada amostra para o hiperplano de separação
    distancias = modelo.decision_function(imagem_redimensionada)
    print("Distâncias:", distancias)
    
    # Aplicar a função de ativação sigmoide para obter probabilidades
    probabilidades = 1 / (1 + np.exp(-distancias))
    print("Probabilidades:", probabilidades)
    
    # Probabilidade da classe 1 (com chuva) como percentual
    percentual_chuva = probabilidades[0] * 100
    print("Percentual de chuva:", percentual_chuva)
    
    return percentual_chuva
    

if __name__ == "__main__":
   
    # Carregar o modelo treinado
    modelo = carregar_modelo()
    precisao = treinar_modelo()
    # Carregar a imagem para fazer a previsão
    caminho_imagem = './teste4.jpg'  # Atualize com o caminho para a imagem que deseja testar
    imagem = carregar_imagem(caminho_imagem)
    if imagem is not None:
        # Fazer a previsão da possibilidade de chuva
        possibilidade_chuva = prever_possibilidade_chuva(modelo, imagem)
        if possibilidade_chuva <= 35:
            print("Não há previsão de chuva.")
        else:
            print("Existe a possibilidade de chuva.")


# import numpy as np
# import cv2
# import joblib
# from treino import treinar_modelo
# from sklearn.svm import SVC


# def carregar_modelo(nome_arquivo='./modelo_chuva.pkl'):
#     # Carregar o modelo treinado
#     modelo = joblib.load(nome_arquivo)
#     return modelo

# def carregar_imagem(caminho_imagem):
#     # Carregar a imagem
#     imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)  # Carregar em escala de cinza
#     if imagem is not None:
#         # Redimensionar a imagem para o tamanho esperado pelo modelo
#         imagem_redimensionada = cv2.resize(imagem, (100, 100))
#         return imagem_redimensionada
#     else:
#         print("Erro ao carregar a imagem.")
#         return None

# def prever_possibilidade_chuva(modelo, imagem):
#     # Redimensionar a imagem para o formato esperado pelo modelo
#     imagem_redimensionada = imagem.reshape(1, -1)
#     # Calcular a distância assinada de cada amostra para o hiperplano de separação
#     distancias = modelo.decision_function(imagem_redimensionada)
#     print("Distâncias:", distancias)
#     # Normalizar as distâncias para obter probabilidades
#     minimo_distancias = distancias.min()
#     maximo_distancias = distancias.max()
#     if minimo_distancias != maximo_distancias :
#         # Se o intervalo de valores das distâncias for zero, definimos todas as distâncias normalizadas como 0.5
#         #distancias_normalizadas = np.full_like(distancias, 0.5)
#         distancias_normalizadas = (distancias - minimo_distancias) / (maximo_distancias - minimo_distancias)
#     else:
#         # Caso contrário, realizamos a normalização normalmente
#         distancias_normalizadas = np.full_like(distancias, 0.5)
#         #distancias_normalizadas = (distancias - minimo_distancias) / (maximo_distancias - minimo_distancias)
#     print("Distâncias normalizadas:", distancias_normalizadas)
    
#     # Probabilidade da classe 1 (com chuva) como percentual
#     percentual_chuva = distancias_normalizadas[0] * 100
#     print("Percentual de chuva:", percentual_chuva)
    
#     return percentual_chuva


# if __name__ == "__main__":
   
#     # Carregar o modelo treinado
#     modelo = carregar_modelo()
#     precisao = treinar_modelo()
#     # Carregar a imagem para fazer a previsão
#     caminho_imagem = './download.jpg'  # Atualize com o caminho para a imagem que deseja testar
#     imagem = carregar_imagem(caminho_imagem)
#     if imagem is not None:
#         #print('precisao do modelo:', precisao)
#         # Fazer a previsão da possibilidade de chuva
#         possibilidade_chuva = prever_possibilidade_chuva(modelo, imagem)
#         if possibilidade_chuva <= 0.4:
#             print("Não há previsão de chuva.")
#         else:
#             print("Existe a possibilidade de chuva.")
