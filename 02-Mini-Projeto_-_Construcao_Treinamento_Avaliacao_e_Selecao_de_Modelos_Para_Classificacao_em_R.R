####  Big Data Real-Time Analytics com Python e Spark  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/2.Big-Data-Real-Time-Analytics-com-Python-e-Spark/8.Machine_Learning_em_Linguagem_Python")
getwd()



## Importando Pacotes
library(readxl)         # carregar arquivos
library(dplyr)          # manipula dados
library(tidyr)          # manipula dados (funcao pivot_longer)
library(ggplot2)        # gera gráficos
library(patchwork)      # unir gráficos
library(corrplot)       # mapa de Correlação
library(caret)          # pacote preProcess para normalização

library(randomForest)   # algoritmo de ML




############################  Modelagem Preditiva para Identificação de Risco de Doença Hepática  ############################  

## Etapas:
  
# - Construção, Treinamento, Avaliação e Seleção de Modelos para Classificação

## Introdução:

# - Vamos trabalhar agora em nosso primeiro Mini-Projeto de Machine Learning, cujo objetivo é fornecer um passo a passo completo do processo de construção,
#   treinamento, avaliação e seleção de modelos para classificação. Este projeto será abordado de maneira integral, desde a definição do problema de negócio
#   até as previsões com o modelo treinado.

## Contexto:

# - O número de pacientes com doença hepática tem aumentado continuamente devido a fatores como consumo excessivo de álcool, inalação de gases nocivos,
#   ingestão de alimentos contaminados, e uso de drogas e anabolizantes. Em resposta a essa crescente preocupação de saúde pública, este mini-projeto visa 
#   construir um modelo de Machine Learning capaz de prever se um paciente irá ou não desenvolver uma doença hepática com base em várias características 
#   clínicas e demográficas. Este modelo pode ser extremamente útil para médicos, hospitais ou governos no planejamento de orçamentos de saúde e na criação 
#   de políticas de prevenção eficazes.

## Objetivo:

# - O objetivo é prever uma classe (sim ou não), usaremos aprendizado supervisionado para classificação, criando diferentes versões do modelo com diferentes 
#   algoritmos e passaremos por todo o processo de Machine Learning de ponta a ponta. Usaremos como fonte de dados o dataset disponível no link a seguir.

## Dados:

# Link -> https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)

# - Os dados para este projeto são provenientes do "Indian Liver Patient Dataset", disponível no link acima. Este conjunto de dados contém registros de
#   pacientes hepáticos e não hepáticos coletados na Índia. A coluna "Dataset" atua como um rótulo de classe, dividindo os indivíduos em pacientes com doença
#   hepática (1) ou sem a doença (2).



#### Carregando os Dados


