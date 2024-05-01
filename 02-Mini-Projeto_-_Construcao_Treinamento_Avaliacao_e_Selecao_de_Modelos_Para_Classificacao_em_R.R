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
library(corrplot)       # mapa de correlação
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

df <- data.frame(read.csv2("dados/dataset.csv", sep = ","))

dim(df)
names(df)
head(df)



#### Análise Exploratória

# Tipo de dados
str(df)


## Aplicando Transformações Iniciais

# Converter todas as colunas para numeric, exceto 'Gender' (ficar igual ao código em Python)
df <- df %>%
  mutate(across(-Gender, as.numeric))


# Atualizar valores na coluna 'Dataset' (valor '2' = '0') e renomear para 'Target'
df <- df %>%
mutate(Dataset = if_else(Dataset == 2, 0, Dataset)) %>%
  rename(Target = Dataset)


# Conveter para Factor as Variáveis Gender e Target(variável alvo)
df <- df %>% 
  mutate(Gender = as.factor(Gender),
         Target = as.factor(Target))



## Realizando Análise Inicial (Sumário Estatístico, Veriricação de Valores NA, '' e especiais)

analise_inicial <- function(dataframe_recebido) {
  # Sumário
  cat("\n\n####  DIMENSÕES  ####\n\n")
  print(dim(dataframe_recebido))
  cat("\n\n\n####  INFO  ####\n\n")
  print(str(dataframe_recebido))
  cat("\n\n\n####  SUMÁRIO  ####\n\n")
  print(summary(dataframe_recebido))
  cat("\n\n\n####  VERIFICANDO QTD DE LINHAS DUPLICADAS  ####\n\n")
  print(sum(duplicated(dataframe_recebido)))
  cat("\n\n\n####  VERIFICANDO VALORES NA  ####\n\n")
  valores_na <- colSums(is.na(dataframe_recebido))
  if(any(valores_na > 0)) {
    cat("\n-> Colunas com valores NA:\n\n")
    print(valores_na[valores_na > 0])
  } else {
    cat("\n-> Não foram encontrados valores NA.\n")
  }
  cat("\n\n\n####  VERIFICANDO VALORES VAZIOS ''  ####\n\n")
  valores_vazios <- sapply(dataframe_recebido, function(x) sum(x == "", na.rm = TRUE)) # Adicionando na.rm = TRUE
  if(any(valores_vazios > 0, na.rm = TRUE)) { # Tratamento de NA na condição
    cat("\n-> Colunas com valores vazios \"\":\n\n")
    print(valores_vazios[valores_vazios > 0])
  } else {
    cat("\n-> Não foram encontrados valores vazios \"\".\n")
  }
  cat("\n\n\n####  VERIFICANDO VALORES COM CARACTERES ESPECIAIS  ####\n\n")
  caracteres_especiais <- sapply(dataframe_recebido, function(x) {
    sum(sapply(x, function(y) {
      if(is.character(y) && length(y) == 1) {
        any(charToRaw(y) > 0x7E | charToRaw(y) < 0x20)
      } else {
        FALSE
      }
    }))
  })
  if(any(caracteres_especiais > 0)) {
    cat("\n-> Colunas com caracteres especiais:\n\n")
    print(caracteres_especiais[caracteres_especiais > 0])
  } else {
    cat("\n-> Não foram encontrados caracteres especiais.\n")
  }
}

analise_inicial(df)



## Dividindo dataset em variáveis numéricas e categóricas (para criação de gráficos)
df_num <- df %>% 
  select(where(is.numeric))
df_cat <- df %>% 
  select(where(is.factor))



### Explorando Variáveis Numéricas

# Sumário Estatístico
summary(df_num)


## Visualizando Através de Gráficos

# Converter o dataframe para um formato longo para facilitar a plotagem com ggplot2
df_long <- pivot_longer(df_num, cols = everything())

# Criar histogramas usando ggplot2
ggplot(df_long, aes(x = value)) +
  geom_histogram(bins = 10, fill = "blue", color = "black") +
  facet_wrap(~name, scales = "free") +
  labs(x = "Value", y = "Frequency") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Melhorar a legibilidade dos rótulos do eixo x


# Interpretando sumário e gráficos:

# - Parece que há outlier nas variáveis "Alamine_Aminotransferase" e "Aspartate_Aminotransferase", pois o valor máximo é muito mais alto que o valor médio.




### Explorando Variável Categórica

str(df_cat)
summary(df_cat)
plot(df$Gender)
plot(df$Target)


## Aplicando Label Encoding na Variável 'Gender' e 'Target' (no dataframe original df)

str(df)

# Cria uma Nova Variável 'Gender_Num' onde Male = 0 e Female = 1
df$Gender_Num <- ifelse(df$Gender == "Male", 0, 1)

# Converte fatores para caracteres e depois para numéricos
df$Target <- as.numeric(as.character(df$Target))

str(df)



## Verificando Correlação

df_num <- df %>% 
  select(-Gender)
cor(df_num, use = "complete.obs")

# Criar um mapa de calor da matriz de correlação
corrplot(cor(df_num, use = "complete.obs"),
         method = "color",
         type = "upper",
         addCoef.col = 'springgreen2',
         tl.col = "black",
         tl.srt = 45)                                     # Esconde a diagonal principal


## Interpretando o resultado da Correlação

# - Vamos citar exemplo:
#   Podemos constatar no dados e gráfico que a variável Total_Bilirubin tem uma alta correlação positiva com a variável Direct_Bilirubin (0.87).
# - Isso é um problema pois a mesma informação está sendo replicada duas vezes e por conta disso pode deixar o modelo tendencioso.
# - O fato de duas variáveis estarem altamente relacionadas (quando tem o valor abaixo ou acima de 0.70) é chamado de Multicolinearidade. 
# - Em algum momento deveremos tomar uma decisão: deixar as duas variáveis, remover uma variável ou remover as duas.

## Atenção

# - Nosso dados ainda não foram limpos/tratados (valores ausentes, replicados ou outliers). É recomendado aplicar algum tipo de tratamento relacionado a Multicolinearidade somente quando os dados estiverem tratados.
# - Estamos na etapa de Análise Exploratória onde estamos entendendo a natureza dos nossos dados.


## Verificando Relação entre Atributs

## Verificando Através de Gráfico a Relação entre as Variáveis 'Direct_Bilirubin' e 'Total_Bilirubin' por 'Target'
ggplot(df, aes(x = Total_Bilirubin, y = Direct_Bilirubin, color = as.factor(Target))) +
  geom_point(alpha = 0.6, size = 3) +  # Pontos semitransparentes e de tamanho moderado
  scale_color_manual(values = c("blue", "red")) +  # Cores para diferentes Targets
  labs(color = "Target") +  # Legenda
  theme_minimal(base_size = 14) +  # Usando um tema minimalista para o background
  theme(
    plot.background = element_rect(fill = "grey90"),  # Cor de fundo do plot
    panel.background = element_rect(fill = "grey90", colour = "grey20", size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid', colour = "grey60"),  # Linhas principais da grade
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid', colour = "grey80")  # Linhas secundárias da grade
  ) +
  labs(title = "Relação entre Total Bilirubin e Direct Bilirubin", x = "Total Bilirubin", y = "Direct Bilirubin")


## Verificando Através de Gráfico a Relação entre as Variáveis 'Direct_Bilirubin' e 'Total_Bilirubin' por 'Gender'
ggplot(df, aes(x = Total_Bilirubin, y = Direct_Bilirubin, color = as.factor(Gender))) +
  geom_point(alpha = 0.6, size = 3) +  # Pontos semitransparentes e de tamanho moderado
  scale_color_manual(values = c("blue", "red")) +  # Cores para diferentes Targets
  labs(color = "Gender") +  # Legenda
  theme_minimal(base_size = 14) +  # Usando um tema minimalista para o background
  theme(
    plot.background = element_rect(fill = "grey90"),  # Cor de fundo do plot
    panel.background = element_rect(fill = "grey90", colour = "grey20", size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid', colour = "grey60"),  # Linhas principais da grade
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid', colour = "grey80")  # Linhas secundárias da grade
  ) +
  labs(title = "Relação entre Total Bilirubin e Direct Bilirubin", x = "Total Bilirubin", y = "Direct Bilirubin")


## Verificando Através de Gráfico a Relação entre as Variáveis 'Albumin' e 'Total_Bilirubin' por 'Target'
ggplot(df, aes(x = Total_Bilirubin, y = Albumin, color = as.factor(Target))) +
  geom_point(alpha = 0.6, size = 3) +  # Pontos semitransparentes e de tamanho moderado
  scale_color_manual(values = c("blue", "red")) +  # Cores para diferentes Targets
  labs(color = "Target") +  # Legenda
  theme_minimal(base_size = 14) +  # Usando um tema minimalista para o background
  theme(
    plot.background = element_rect(fill = "grey90"),  # Cor de fundo do plot
    panel.background = element_rect(fill = "grey90", colour = "grey20", size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid', colour = "grey60"),  # Linhas principais da grade
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid', colour = "grey80")  # Linhas secundárias da grade
  ) +
  labs(title = "Relação entre Total Bilirubin e Albumin", x = "Total Bilirubin", y = "Albumin")


## Verificando Através de Gráfico a Relação entre as Variáveis 'Albumin' e 'Total_Bilirubin' por 'Gender'
ggplot(df, aes(x = Total_Bilirubin, y = Albumin, color = as.factor(Gender))) +
  geom_point(alpha = 0.6, size = 3) +  # Pontos semitransparentes e de tamanho moderado
  scale_color_manual(values = c("blue", "red")) +  # Cores para diferentes Targets
  labs(color = "Gender") +  # Legenda
  theme_minimal(base_size = 14) +  # Usando um tema minimalista para o background
  theme(
    plot.background = element_rect(fill = "grey90"),  # Cor de fundo do plot
    panel.background = element_rect(fill = "grey90", colour = "grey20", size = 0.5, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid', colour = "grey60"),  # Linhas principais da grade
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid', colour = "grey80")  # Linhas secundárias da grade
  ) +
  labs(title = "Relação entre Total Bilirubin e Albumin", x = "Total Bilirubin", y = "Albumin")



#### Conclusões da Análise Exploratória

# - A análise exploratória ajudou a entender a natureza dos dados, preparando o caminho para limpeza de dados e análises mais profundas.
# - Identificou-se a necessidade de tratar valores ausentes e possíveis outliers.
# - A análise de correlação destacou a presença de multicolinearidade, que pode afetar a performance de modelos de aprendizado de máquina.





#### Verfificando e Tratando Valores Ausentes, Replicados e Outliers

analise_inicial(df)


## Verificando Valores Ausentes (o tratamento é aconselhado após tratamento dos valores outliers)

# Exibindo as linhas com valores ausentes
df %>% filter(is.na(Albumin_and_Globulin_Ratio))



## Tratando Valores Duplicados

# Exibindo as linhas com valores duplicados
df %>% filter(duplicated(.))

# Removendo linhas duplicadas (remove uma das duplicatas)
df <- df %>% 
  distinct()



## Tratando Valores Outliers

# - Irá ser apresentado dois cenários e tomaremos decisões diferentes para cada um deles.


## Cenário 1 (Variável 'Alamine_Aminotransferase')

# Sumário
summary(df$Alamine_Aminotransferase)

# - Através do sumário, podemos observar que a variável possui uma média de 80.14 e um valor máx de 2000. Isso é um sinal de que podemos ter um outlier.
# - Podemos checar esta informação através da criação de um Gráfico BoxPlot.


# Gráfico BoxPLot
ggplot(df, aes(y = Alamine_Aminotransferase)) +
  geom_boxplot(fill = "blue", colour = "black") +
  labs(title = "Boxplot de Alamine Aminotransferase", y = "Alamine Aminotransferase") +
  theme_minimal()  # Aplicando um tema minimalista

# Interpretando o Gráfico

# - Podemos verificar que além do valor de 2000 temos outros diversos valores acima da média (que está próxima de zero).
# - Será que os valores extremos são mesmo outliers para esta variável?
  
# Podemos responder isso verificando contagem de frequência por valor abaixo (filtrando os 5 maiores valores):


# Exibindo os cinco maiores valores únicos e suas frequências:
table(df$Alamine_Aminotransferase)[as.character(sort(unique(df$Alamine_Aminotransferase), decreasing = TRUE)[1:5])]


# Conclusão

# - Após a análise detalhada da variável 'Alamine_Aminotransferase', identificamos que o valor máximo de 2000 é consideravelmente mais alto que os outros
#   valores próximos, que também são altos mas menos frequentes.
# - Esses valores extremos podem ser considerados outliers devido ao seu afastamento significativo da média e mediana, além de serem raros no dataset,
#   como mostrado pela análise de frequência.

# - Dado esse contexto, é sugerido a avaliação de tratamento desses outliers dentro do cenário de aplicação dos dados. Se estes valores são resultantes de
#   erros de medição ou casos muito atípicos que podem distorcer análises estatísticas, a remoção ou substituição por um limite superior calculado pelo
#   método do IQR é recomendada.
# - Contudo, se esses altos valores representam casos válidos dentro da pesquisa ou aplicação prática dos dados, poderiam ser mantidos, mas com uma análise
#   adicional para confirmar sua validade.

# - Portanto neste caso específico, após verificar a validade dos dados, optou-se por não realizar o tratamento de outliers para esta variável, pois eles
#   representam casos autênticos dentro do contexto estudado.



## Cenário 2 (Variável 'Aspartate_Aminotransferase')

# Sumário
summary(df$Aspartate_Aminotransferase)

# - Através do Sumário podemos observar que a variável possui uma média de 109.89 e um valor máx de 4929. Isso é um sinal de que podemos ter um ou mais
#   valores outlier.
# - Vamos novamente verificar por um Gráfico BoxPlot.

# Gráfico BoxPLot
ggplot(df, aes(y = Aspartate_Aminotransferase)) +
  geom_boxplot(fill = "blue", colour = "black") +
  labs(title = "Boxplot de Aspartate Aminotransferase", y = "Aspartate Aminotransferase") +
  theme_minimal()  # Aplicando um tema minimalista

# Interpretando o gráfico

# - Podemos verificar que novamente temos valores outliers, mas com um comportamente diferente. Parece que temos menos dados com valores extremos.
# - E neste caso, ss valores extremos são mesmo outliers para esta variável?
  
# Podemos responder isso verificando novamente os maiores valores únicos e suas frequências:

# Exibindo os cinco maiores valores únicos e suas frequências:
table(df$Aspartate_Aminotransferase)[as.character(sort(unique(df$Aspartate_Aminotransferase), decreasing = TRUE)[1:5])]


# Conclusão

# - Vamos aplicar um tratamento para limpeza de outlier nesta variável.
# - Iremos manter no dataset todos os registros abaixo do valor 2500 para esta variável.


# Tratando Valores Outliers da Variável 'Alamine_Aminotransferase'

dim(df)
summary(df)

# Aplica tratamento mantendo somente os registros onde o valor for menor ou igual a 3000 e verifica shape
df <- df %>% filter(Aspartate_Aminotransferase <= 3000)
dim(df)

# BoxPlot
ggplot(df, aes(y = Aspartate_Aminotransferase)) +
  geom_boxplot(fill = "blue", colour = "black") +
  labs(title = "Boxplot de Aspartate Aminotransferase Após Primeiro Filtro", y = "Aspartate Aminotransferase")

# Aplica novo tratamento mantendo somente os registros onde o valor for menor ou igual a 2500 e verifica shape 
df <- df %>% filter(Aspartate_Aminotransferase <= 2500)

# BoxPlot
ggplot(df, aes(y = Aspartate_Aminotransferase)) +
  geom_boxplot(fill = "blue", colour = "black") +
  labs(title = "Boxplot de Aspartate Aminotransferase Após Segundo Filtro", y = "Aspartate Aminotransferase")


dim(df)
summary(df)


## Tratando Valores Ausentes
dim(df)

# Removendo todas linhas com valores ausentes
df <- df %>% drop_na()
dim(df)



#### Pré-Processamento de Dados Para Construção de Modelos de Machine Learning¶

# - Como vimos anteriormente ao aplicarmos o mapa de correlação as variáveis 'Direct_Bilirubin' e 'Total_Bilirubin' possuem uma alta correlação.
# - Com isso foi tomada a decisão de remover umas das variáveis.

# Removendo Variável 'Direct_Bilirubin'






