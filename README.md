# previsaodotempo
O projeto é um sistema de previsão de chuva com base em imagens de radar meteorológico. Ele utiliza imagens de radar para treinar um modelo de machine learning, neste caso, um classificador SVM (Support Vector Machine), para prever a possibilidade de chuva com base em imagens de nuvens.

Como Funciona:

Treinamento do Modelo:

O projeto começa carregando imagens de radar meteorológico e seus rótulos (indicando se houve ou não chuva).
Essas imagens são pré-processadas, redimensionadas e transformadas em um formato adequado para alimentar o modelo.
Um modelo de classificação SVM é treinado usando essas imagens como entrada e seus rótulos correspondentes como saída.
Carregamento do Modelo Treinado:

Após o treinamento, o modelo SVM treinado é salvo em um arquivo.
Previsão de Chuva:

O usuário pode carregar o modelo treinado e fornecer uma imagem de radar meteorológico como entrada.
A imagem é carregada, pré-processada e redimensionada para o formato esperado pelo modelo.
Em seguida, a função prever_possibilidade_chuva() é usada para calcular a possibilidade de chuva com base na imagem fornecida.
Esta função usa as distâncias assinadas das amostras ao hiperplano de separação do SVM para calcular probabilidades de chuva.
As probabilidades são mapeadas para o intervalo [0, 1] usando a função de ativação sigmoide.
Finalmente, a possibilidade de chuva é retornada como um percentual.
Este projeto fornece uma maneira de prever a possibilidade de chuva com base em imagens de radar meteorológico usando técnicas de aprendizado de máquina, permitindo que os usuários obtenham informações úteis sobre o clima com base nessas imagens.


carregue a imagen no arquivo main.py na função init main
após isso salve o arquivo 
execute o arquivo de treino.py 
e agora pode executar o main e receber as informações sobre a sua imagem
