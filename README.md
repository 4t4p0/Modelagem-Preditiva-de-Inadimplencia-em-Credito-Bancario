# Modelagem Preditiva de Inadimplencia em Crédito Bancário

Modelo preditivo que analisa a chance de inadimplencia de pessoas ao solicitar um crédito bancário 

## Objetivo

Criar um modelo preditivo utilizando conceitos de ciência de dados para analisar a chance de uma pessoa pagar ou não o seu crédito bancário

## Tecnologias Utilizadas

- Linguagem: **Python**
- Biblioteca: **pandas, matplotlib, numpy, plotly, pickle, sklearn**

## Conceitos aplicados

Conceitos de Ciência de dados de um modo geral, como:
- Análise exploratória de dados (EDA)
- Avaliação de modelos
- Visualização de dados
- Estatística descritiva
- Seleção de variáveis (features)
- Pré-processamento de dados
- Modelagem preditiva

## Como foi feito

A base de dados foi retirada do kaggle (https://www.kaggle.com/datasets/laotse/credit-risk-dataset), após isso foi feita uma limpeza e reorganização da base para deixa-la pronta para a análise, deixando 75% da base para treinamento e 25% para teste, vale ressaltar que a váriavel que foi analisada foi a inadinplencia (1 - inadinplente / 0 - não inadinplente).
Foram testados vários modelos preditivos como árvores de decisão, Random forest, Naive Bayes, KNN, Regressão logística, SVM e redes neurais, aplicando a validação cruzada em todos eles para ter se ter uma menor chance de erros no treinamento dos modelos.
Assim, foi feito uma infêrencia estatística para achar o melhor modelo, ou seja, com uma alta taxa de acertos e baixo desvio padrão. Dessa forma, foi concluído que o melhor modelo foi o de Random forest com uma taxa de acertos de cerca de 93%.

## Como usar

Alguns dos modelos foram salvos em um arquivo .sav e podem ser usados. Entretando, vale ressaltar que é necessário ter conhecimentos certos conhecimetos de ciência de dados para utiliza-los e que esses modelos podem não ser representativos da realidade, já que a base de dados pode conter vários erros ou mesmo ser fictícia
