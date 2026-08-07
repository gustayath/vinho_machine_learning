# 🍷 Classificação de Tipos de Vinho com Machine Learning

Este repositório contém um estudo prático de Machine Learning para a **classificação de tipos de vinho** (Tinto vs. Branco) com base em suas propriedades físico-químicas, utilizando o algoritmo **ExtraTreesClassifier**.

---

## 📌 Visão Geral do Projeto

O objetivo deste projeto é prever o estilo do vinho (`style`) a partir de dados físico-químicos das amostras [cite: 1]. 

* **Tipo de Aprendizado:** Supervisionado (Classificação Binária) [cite: 1]
* **Target (Variável Alvo):** `style` (Mapeado como `0` para Tinto / *Red* e `1` para Branco / *White*) [cite: 1]
* **Algoritmo Utilizado:** `ExtraTreesClassifier` (Scikit-Learn) [cite: 1]
* **Acurácia Alcançada:** **~99.69%** no conjunto de teste [cite: 1]

---

## 📊 Conjunto de Dados

O dataset utilizado (`wine_dataset.csv`) contém características físico-químicas de amostras de vinho [cite: 1]. As variáveis preditoras incluem:

| Atributo | Descrição |
| :--- | :--- |
| `fixed_acidity` | Acidez fixa [cite: 1] |
| `volatile_acidity` | Acidez volátil [cite: 1] |
| `citric_acid` | Ácido cítrico [cite: 1] |
| `residual_sugar` | Açúcar residual [cite: 1] |
| `chlorides` | Cloretos [cite: 1] |
| `free_sulfur_dioxide` | Dióxido de enxofre livre [cite: 1] |
| `total_sulfur_dioxide` | Dióxido de enxofre total [cite: 1] |
| `density` | Densidade [cite: 1] |
| `pH` | Potencial hidrogeniônico (pH) [cite: 1] |
| `sulphates` | Sulfatos [cite: 1] |
| `alcohol` | Teor alcoólico [cite: 1] |
| `quality` | Nota de qualidade do vinho [cite: 1] |

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Python 3.x**
* **Pandas** — Manipulação e tratamento de dados [cite: 1]
* **Scikit-Learn** — Divisão de dados (`train_test_split`) e modelo de Machine Learning (`ExtraTreesClassifier`) [cite: 1]

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

Este projeto está licenciado sob a Licença **MIT** - consulte o arquivo [LICENSE](LICENSE) para obter mais detalhes.
