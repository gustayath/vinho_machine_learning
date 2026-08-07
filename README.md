🍷 Classificação de Tipos de Vinho com Machine Learning:
Este repositório contém um estudo prático de Machine Learning para a classificação do tipo de vinho (Tinto vs. Branco) com base em suas propriedades físico-químicas, utilizando o algoritmo ExtraTreesClassifier.

📌 Visão Geral do Projeto:
O objetivo deste estudo é prever o estilo do vinho (style) a partir de dados analíticos das amostras.
- Tipo de Aprendizado: Supervisionado (Classificação Binária)
- Target (Variável Alvo): style (Mapeado como 0 para Tinto / Red e 1 para Branco / White)
- Algoritmo Utilizado: ExtraTreesClassifier (sklearn.ensemble)
- Acurácia Alcançada: ~99.69% no conjunto de teste

📊 Conjunto de Dados:
O dataset (wine_dataset.csv) contém características físico-químicas de diversas amostras. As variáveis preditoras utilizadas no modelo são:
![image_alt](https://github.com/gustayath/vinho_machine_learning/blob/5cf1a3858a176b8cbf4c536bd0cde91bb5501076/Captura%20de%20tela%202026-08-07%20161944.png)

🛠️ Tecnologias e Bibliotecas
- Python 3.14
- Pandas — Carregamento, tratamento e pré-processamento de dados
- Scikit-Learn — Divisão em treino/teste (train_test_split) e modelo de classificação (ExtraTreesClassifier)

⚙️ Passo a Passo do Código
1. Carregamento dos Dados
Python
import pandas as pd

arquivo = pd.read_csv(r'vinho_machine_learning/wine_dataset.csv')
2. Tratamento da Variável Categórica
Transformação da coluna style em valores numéricos (0 para vermelho/tinto e 1 para branco)[cite: 1]:

Python
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)
3. Divisão entre Variáveis Preditoras e Alvo
Python
y = arquivo['style'].astype(int)
x = arquivo.drop('style', axis=1)
4. Separação em Conjuntos de Treino e Teste
Uso de 70% dos dados para treinamento e 30% para avaliação do modelo[cite: 1]:

Python
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
5. Treinamento do Modelo e Acurácia
Python
from sklearn.ensemble import ExtraTreesClassifier

# Instanciando e treinando o modelo
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

# Avaliação do desempenho
resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)  # Resultado: ~0.9969 (99.69%)
🧪 Teste de Predição Prática
Para validar o funcionamento, realizou-se uma predição em amostras específicas do conjunto de teste (x_teste[400:403])[cite: 1]:

Python
previsoes = modelo.predict(x_teste[400:403])
print(previsoes)
# Saída esperada: array([1, 1, 0]) -> [Branco, Branco, Tinto]
Comparando com as classes reais (y_teste[400:403]):

Índice 2783: 1 (Vinho Branco)[cite: 1]

Índice 4299: 1 (Vinho Branco)[cite: 1]

Índice 1021: 0 (Vinho Tinto)[cite: 1]

🚀 Como Executar o Projeto
Clone o repositório:

Bash
git clone https://github.com/seu-usuario/seu-repositorio.git
Instale as dependências necessárias:

Bash
pip install pandas scikit-learn
Garanta que o arquivo wine_dataset.csv está no caminho correto e execute o notebook ou script
