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
