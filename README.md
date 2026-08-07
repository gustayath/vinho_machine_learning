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
AtributoDescriçãofixed_acidityAcidez fixa  volatile_acidityAcidez volátil[cite: 1]citric_acidÁcido cítrico[cite: 1]residual_sugarAçúcar residual[cite: 1]chloridesCloretos[cite: 1]free_sulfur_dioxideDióxido de enxofre livre[cite: 1]total_sulfur_dioxideDióxido de enxofre total[cite: 1]densityDensidade[cite: 1]pHPotencial hidrogeniônico (pH)[cite: 1]sulphatesSulfatos[cite: 1]alcoholTeor alcoólico[cite: 1]qualityNota de qualidade percebida[cite: 1]
