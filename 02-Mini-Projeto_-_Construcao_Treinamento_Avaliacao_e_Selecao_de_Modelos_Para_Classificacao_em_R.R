####  Big Data Real-Time Analytics com Python e Spark  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/2.Big-Data-Real-Time-Analytics-com-Python-e-Spark/8.Machine_Learning_em_Linguagem_Python")
getwd()



## Importando Pacotes
library(readxl)         # carregar arquivos
library(dplyr)          # manipula dados
library(tidyr)          # manipula dados (funcao pivot_longer)
library(ROSE)           # balanceamento de dados
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

# Exibindo a quantidade de valores acima da média:
sum(df$Alamine_Aminotransferase > mean(df$Alamine_Aminotransferase))  # Contagem de valores acima da média
length(df$Alamine_Aminotransferase)                                   # Contagem total de valores da variável


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
# - Aqui nós temos apenas dois valores outliers acima de 2000 enquanto todos os outros abaixo deste valor. 
# - E neste caso, todos esses valores extremos são mesmo outliers para esta variável?
  
# Podemos responder isso verificando novamente os maiores valores únicos e suas frequências:


# Exibindo os cinco maiores valores únicos e suas frequências:
table(df$Aspartate_Aminotransferase)[as.character(sort(unique(df$Aspartate_Aminotransferase), decreasing = TRUE)[1:5])]

# Exibindo a quantidade de valores acima da média:
sum(df$Aspartate_Aminotransferase > mean(df$Aspartate_Aminotransferase))  # Contagem de valores acima da média
length(df$Aspartate_Aminotransferase)                                     # Contagem total de valores da variável

# Exibindo a quantidade de valores acima de 2000:
sum(df$Aspartate_Aminotransferase > 2000)                                 # Contagem de valores acima de 2000
length(df$Aspartate_Aminotransferase)                                     # Contagem total de valores da variável


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



#### RESUMO

# - Antes de avançarmos para a etapa final de pré-processamento de dados, crucial para a construção de modelos de machine learning, vamos recapitular
#   os passos já concluídos no projeto.:
  
#  -> Primeiro foi definido o problema de negócio para saber o objetivo e o que temos que resolver.
#  -> Depois nós extraímos os dados e nesta etapa pode ser que tenhamos o suporte de um Engenheiro de Dados. No caso deste projeto foi feito a leitura dos
#     dados através de um arquivo csv.
#  -> Na sequência foi feita a Análise Exploratória onde nós verificamos padrões, detectamos problemas, identifica coisas que precisamos fazer.
#  -> Após isso é aplicado a Limpeza de Dados de acordo com as técnicas necessárias, estratégias e decisões.
#  -> Sempre lembrar de documentar tudo o que foi feito em cada atividade.





#### Pré-Processamento de Dados Para Construção de Modelos de Machine Learning¶

# - Como vimos anteriormente ao aplicarmos o mapa de correlação as variáveis 'Direct_Bilirubin' e 'Total_Bilirubin' possuem uma alta correlação.
# - Com isso foi tomada a decisão de remover umas das variáveis.

# Removendo Variável 'Direct_Bilirubin'
df <- df %>% 
  select(-Direct_Bilirubin)


##  Dividindo os dados em treino e teste

set.seed(100)
indices <- createDataPartition(df$Target, p = 0.75, list = FALSE)
dados_treino <- df[indices, ]
dados_teste <- df[-indices, ]
rm(indices)



## Balanceamento de Classes

# Contagem
table(dados_treino$Target)

# Por que realizar o Balanceamento de Classes ?

# - Como foi observado no table() acima podemos constatar que os dados estão desbalanceados, isso signifca que tem muito mais pacientes de uma classe do
#   que da outra.
# - E o que acontece quando não realizarmos o balanceamento? O modelo de ML aprenderá muito mais o padrão da Classe 1 do que da Classe 0.
# - Caso não aplicamos técnica de Balanceamento, o modelo tende a ficar tendencioso. Por isso precisamos fazer o Balanceamento de Classes.

# Estratégias para o Balanceamento

# Temos duas estratégias:

# - Reduzir os registros da classe majoritária e assim diminuir consideravelmente o número de registros no nosso dataset.
# - Aplicar a técnica de Oversampling onde irá ser aumentado o número de registros das classe minoritária. E como isso é feito? Sendo criado dados
#   sintéticos com base nos dados existentes (Para isso, podemos utilizar o pacote ROSE em R, que oferece funções para gerar dados sintéticos).


# Balanceamento da Variável Alvo (Aplicando a técnica Oversampling para balancear a variável alvo)
table(dados_treino$Target)
dados_balanceados <- ovun.sample(Target ~ ., data = dados_treino, method = "over", N = 2*max(table(dados_treino$Target)))$data
table(dados_balanceados$Target)


# Por que a técnica de oversamping dever se aplicada somente nos dados de treino?

# - A técnica de oversampling deve ser aplicada somente nos dados de treino para evitar o vazamento de dados (data leakage) e garantir uma avaliação justa
#   e realista do modelo durante o teste.
# - Se o balanceamento fosse aplicado ao conjunto de dados completo, incluindo os dados de teste, o modelo poderia acabar sendo avaliado com dados
#   sintéticos, não representativos da realidade, influenciando os resultados dos testes e comprometendo a capacidade de generalização do modelo para novos
#   dados não vistos.
# - Portanto, mantendo o conjunto de teste original, sem dados sintéticos, asseguramos que a performance do modelo reflete melhor sua eficácia em cenários
#   reais.


# Tamanho
dim(dados_treino)
dim(dados_balanceados)

# O dataset de treino agora passou de 423 linhas para 608 linhas.

# Ajusta o nome do dataset de treino
dados_treino <- dados_balanceados
rm(dados_balanceados)

# Contagem
table(dados_treino$Target)



## Padronização x Normalização

# - Normalmente a aplicação de técnica de padronização ou normalização serão as últimas atividades dentro do Pré-Processamento.

## Quando usar Padronização:
  
# - Padronização transforma os dados de modo que eles tenham média zero e desvio padrão igual a um.
# - Isso é útil para dados que já são centralizados e precisam de ajuste de escala sem vinculação a um intervalo específico.
# - Por exemplo, se você tem alturas de pessoas em centímetros variando de 150cm a 190cm e pesos em quilogramas de 50kg a 100kg, a padronização ajudaria a
#   comparar essas duas medidas em uma escala comum sem distorcer as diferenças de intervalo.
# - Essa técnica é particularmente útil em algoritmos como Regressão Logística e SVM, que são sensíveis à variação nas escalas das variáveis de entrada.

## Quando usar Normalização:

# - Normalização ajusta os dados para que seus valores caibam em um intervalo predefinido, geralmente de 0 a 1.
# - Isso é especialmente importante quando os dados exibem variações extremas nas escalas das características e quando um algoritmo é sensível à magnitude
#   dos dados, como K-Nearest Neighbors (KNN) e Clustering.
# - Por exemplo, imagine um cenário onde um conjunto de dados inclui os preços dos produtos variando de 1 real a 1000 reais e a quantidade vendida desses
#   produtos de 1 a 20. A normalização permitiria que ambos os atributos contribuíssem igualmente para o aprendizado do modelo, evitando que os preços
#   dominem simplesmente porque seus valores são muito maiores.

## Importante:

# - Não é necessário aplicar padronização/normalização na variável alvo.
# - Nós não aplicamos as duas técnicas, ou usamos uma ou outra.
# - A normalização pode não ser a melhor escolha se houver outliers significativos no conjunto de dados, pois isso poderia comprimir a maioria dos dados
#   em um intervalo muito estreito. Nesses casos, a padronização é recomendada.

## Por que iremos usar Padronização nos dados deste projeto?

# Uniformidade na escala: Os dados possuem variáveis com escalas muito diferentes. Por exemplo, a variável Alkaline_Phosphotase tem valores que vão até
# 2110, enquanto Gender varia apenas entre 0 e 1. A padronização ajusta todas as variáveis para terem média zero e desvio padrão um, garantindo que nenhuma
# variável domine o modelo devido à sua escala.
# - Melhor desempenho em algoritmos sensíveis à escala: Algoritmos como SVM (Máquinas de Vetores de Suporte) e Regressão Logística são sensíveis à escala
#   das variáveis. A padronização ajuda a evitar que características com maior magnitude influenciem desproporcionalmente o resultado do aprendizado.

# E por que não usamos normalização aqui?
  
# - Presença de outliers: A normalização pode não ser ideal quando há outliers significativos, pois comprime a maioria dos dados em um intervalo muito
#   estreito e deixa os outliers ainda mais destacados. Nossos dados incluem algumas variáveis com outliers extremos
#   (por exemplo, Aspartate_Aminotransferase com valor máximo de 4929), o que poderia distorcer a análise.
# - Os dados em todas as variáveis deveriam variar dentro de limites razoavelmente próximos, sem a presença de outliers extremos. Isso significa que não
#   deveria haver valores que distorcessem significativamente o intervalo geral dos dados. Por exemplo, todas as variáveis deveriam ter valores entre
#   limites como 0 a 100 ou 0 a 1.000, sem saltos drásticos como de 0 a 20.000.
# - Menos efetiva para alguns modelos: A normalização, ao contrário da padronização, pode ser menos eficaz para modelos que assumem que os dados estão
#   distribuídos normalmente, como é o caso de muitos algoritmos de machine learning.












#### Separando Dados de Treino e Teste (Python X R)

## No R:
# - É comum especificar a variável alvo diretamente nos modelos ou funções de treinamento. Por exemplo, ao usar o pacote caret ou funções nativas como lm()
#   para regressão linear, você normalmente formula o modelo dentro da função, como em lm(y ~ ., data = dados_treino), onde y é a variável alvo e
#   indica o uso de todas as outras variáveis no dataframe como preditores.
# - Isso significa que não há necessidade estrita de separar fisicamente a variável alvo das demais variáveis antes do treinamento do modelo.

## No Python:
# - Ao usar bibliotecas como scikit-learn, você geralmente precisa passar explicitamente os arrays ou matrizes de características e a variável alvo
#   separadamente para as funções de treinamento. Por exemplo, ao treinar um regressor logístico, você usaria algo como LogisticRegression().fit(X_treino,
#   y_treino). Aqui, X_treino e y_treino são passados como argumentos separados, o que requer que você prepare esses objetos com antecedência.
# - Em Python, mesmo que você esteja usando uma biblioteca que permite formulações mais semelhantes ao R (como statsmodels), a prática comum e a maioria
#   das APIs de machine learning ainda segue o padrão de passar X e y separadamente.

# Por que isso é feito dessa forma em Python?

# - A separação explícita de X e y fornece clareza e evita erros em um ecossistema que é menos integrado do que o R para análises estatísticas.
#   As bibliotecas de Python, como scikit-learn, são projetadas para serem agnósticas quanto ao tipo de dados, permitindo o trabalho com arrays numpy,
#   dataframes pandas, e outros formatos de dados, de uma maneira altamente modular e flexível. Além disso, essa separação ajuda na implementação de uma
#   variedade de pré-processamentos e transformações de maneira mais controlada e sem risco de alterar inadvertidamente a variável alvo.

# Essas diferenças refletem filosofias de design distintas e têm implicações práticas na maneira como você prepara e manipula dados para análises e
# modelagem em cada linguagem.

