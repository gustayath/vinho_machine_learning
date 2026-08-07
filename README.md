# 🍷 Classificação de Tipos de Vinho com Machine Learning

Este repositório contém um estudo prático de Machine Learning para a **classificação de tipos de vinho** (Tinto vs. Branco) com base em suas propriedades físico-químicas, utilizando o algoritmo **ExtraTreesClassifier**.

---

## 📌 Visão Geral do Projeto

O objetivo deste projeto é prever o estilo do vinho (`style`) a partir de dados físico-químicos das amostras. 

* **Tipo de Aprendizado:** Supervisionado (Classificação Binária)
* **Target (Variável Alvo):** `style` (Mapeado como `0` para Tinto / *Red* e `1` para Branco / *White*)
* **Algoritmo Utilizado:** `ExtraTreesClassifier` (Scikit-Learn)
* **Acurácia Alcançada:** **~99.69%** no conjunto de teste

---

## 📊 Conjunto de Dados

O dataset utilizado (`wine_dataset.csv`) contém características físico-químicas de amostras de vinho [cite: 1]. As variáveis preditoras incluem:

| Atributo | Descrição |
| :--- | :--- |
| `fixed_acidity` | Acidez fixa |
| `volatile_acidity` | Acidez volátil |
| `citric_acid` | Ácido cítrico |
| `residual_sugar` | Açúcar residual |
| `chlorides` | Cloretos |
| `free_sulfur_dioxide` | Dióxido de enxofre livre |
| `total_sulfur_dioxide` | Dióxido de enxofre total |
| `density` | Densidade |
| `pH` | Potencial hidrogeniônico (pH) |
| `sulphates` | Sulfatos |
| `alcohol` | Teor alcoólico |
| `quality` | Nota de qualidade do vinho |

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Python 3.14**
* **Pandas** — Manipulação e tratamento de dados
* **Scikit-Learn** — Divisão de dados (`train_test_split`) e modelo de Machine Learning (`ExtraTreesClassifier`)

---

## ⚙️ Passo a Passo da Implementação

### 1. Carregamento dos Dados
```python
import pandas as pd

arquivo = pd.read_csv(r'vinho_machine_learning/wine_dataset.csv')
```

### 2. Mapeamento de Variáveis Categóricas
Transformação do rótulo textual em formato numérico para o modelo de machine learning:
```python
arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)
```

### 3. Separação de Variáveis Preditoras e Alvo
```python
y = arquivo['style'].astype(int)
x = arquivo.drop('style', axis=1)
```

### 4. Divisão em Treino e Teste
Uso de 70% dos dados para treinamento e 30% para teste:
```python
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
```

### 5. Treinamento e Avaliação do Modelo
```python
from sklearn.ensemble import ExtraTreesClassifier

modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

resultado = modelo.score(x_teste, y_teste)
print("Acurácia:", resultado)  # Exemplo de saída: Acurácia 0.9969230769230769
```

---

## 🧪 Teste Prático de Predição

Exemplo de validação de predição em amostras do conjunto de teste (`x_teste[400:403]`):

```python
previsoes = modelo.predict(x_teste[400:403])
print(previsoes)
# Saída: array([1, 1, 0]) -> [Branco, Branco, Tinto]
```

Valores reais esperados (`y_teste[400:403]`):
```
2783    1 (Branco)
4299    1 (Branco)
1021    0 (Tinto)
```

---

## 🚀 Como Executar o Projeto

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   ```
2. Instale as dependências necessárias:
   ```bash
   pip install pandas scikit-learn
   ```
3. Execute o script Python ou abra o notebook no Jupyter Notebook / VS Code.

---

## 📜 Licença
GitHub: gustayath
Linkedin: Gustavo Yath
