Recursos Humanos
Definição do problema
A empresa MGuedes Corporation coletou dados do ano anterior com diversos atributos de funcionários de todos os setores. O objetivo é estudar a relação desses atributos e o impacto na promoção dos funcionários. Esse trabalho de análise pode ser usando mais tarde para construir modelos de Machine Learning para prever se um colaborador será ou não promovido.

Na sequência estão as perguntas que devem ser respondidas:

Pergunta 1 - Qual a Correlação Entre os Atributos Dos Funcionários?
Pergunta 2 - Qual o Tempo de Serviço da Maioria Dos Funcionários?
Pergunta 3 - Qual Avalição do Ano Anterior Foi Mais Comum?
Pergunta 4 - Qual a Distribuição Das Idades Dos Funcionários?
Pergunta 5 - Qual o Número De Treinamentos Mais Frequente?
Pergunta 6 - Qual a Proporção Dos Funcionários Por Canal De Recrutamento?
Pergunta 7 - Qual a Relação Entre a Promoção e a Avalição Do Ano Anterior?
O trabalho a ser feito é limpar e preparar os dados e então construir um Dashboard para apresentar os resultados.

Carregando e Instalando Pacotes

# Versão da linguagem Python
from platform import python_version
print('Versão da Linguagem Python usada neste notebook:', python_version())

# Importando as bibliotecas
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

Carregando os Dados

dadosRH = pd.read_csv('dadosRH.csv')

dadosRH.head()

dadosRH.shape

Análise Exploratória, Limpeza e Transformação dos Dados

Manipulando os dados e corrigindo eventuais problemas.

dadosRH.isnull().sum()

dadosRH.groupby(['educacao']).count()

sns.countplot(dadosRH['educacao'])

dadosRH.groupby(['aval_ano_anterior']).count()

sns.countplot(dadosRH['aval_ano_anterior'])

Aplicando imputação e preenchendo os valores ausentes.

dadosRH['educacao'].fillna(dadosRH['educacao'].mode()[0], inplace = True)

dadosRH['aval_ano_anterior'].fillna(dadosRH['aval_ano_anterior'].median(), inplace = True)

dadosRH.shape

dadosRH.isnull().sum()

dadosRH.groupby(['educacao']).count()

dadosRH.groupby(['aval_ano_anterior']).count()

Verificação do balanceamento de classe na variável promovido

dadosRH.groupby(['promovido']).count()

sns.countplot(dadosRH['promovido'])

df_classe_majoritaria = dadosRH[dadosRH.promovido==0]
df_classe_minonitaria = dadosRH[dadosRH.promovido==1]

df_classe_majoritaria.shape

df_classe_minonitaria.shape

# Upsample da classe minoritaria
from sklearn.utils import resample
df_classe_minonitaria_upsampled = resample(df_classe_minonitaria,
                                            replace = True,
                                            n_samples = 50140,
                                            random_state = 150)

dadosRH_balanceados = pd.concat([df_classe_majoritaria, df_classe_minonitaria_upsampled])

dadosRH_balanceados.promovido.value_counts()

dadosRH_balanceados.info()

sns.countplot(dadosRH_balanceados['promovido'])

Os dados estão balanceados. Salvando o dataset com os dados manipulados.

dadosRH_balanceados.to_csv("dadosRH_modificado.csv", encoding = 'utf-8', index = False)

Carregando os dados e seguindo com o trabalho de análise

# Carregando os dados
dataset = pd.read_csv('dadosRH_modificado.csv')

dataset.head()

dataset.shape

Pergunta 1 - Qual a Correlação Entre os Atributos dos Funcionários?

import matplotlib.pyplot as plt
import seaborn as sns
corr = dataset.corr()
sns.heatmap(corr, cmap = 'YlOrRd', linewidths = 0.1)
plt.show()

Respostas

Quanto maior é a idade maior é o tempo de serviço.

Pergunta 2 - Qual o Tempo de Serviço da Maioria dos Funcionários?

import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(dataset['tempo_servico'], color = 'teal')
plt.title('Distribuição do Tempo de Serviço dos Funcionários', fontsize = 15)
plt.xlabel('Tempo de Serviço em Anos', fontsize = 15)
plt.ylabel('Total', fontsize = 12)
plt.show()

Respostas

O tempo em média que um funcionário permanece na empresa são de cinco anos.

Pergunta 3 - Qual a Avaliação do Ano Anterior Foi Mais Comum?

import matplotlib.pyplot as plt
import seaborn as sns
dataset['aval_ano_anterior'].value_counts().sort_values().plot.bar(color = 'seagreen', figsize = (10,5))
plt.title('Distribuição da Avaliação do Ano Anterior dos Funcionários', fontsize = 15)
plt.xlabel('Avaliações', fontsize = 15)
plt.ylabel('Total', fontsize = 12)
plt.show()

Respostas

A maioria dos funcionários ficaram com a avaliação dentro da média (3,0).
Alguns dos funcionários foram avaliados acima da média (4.0 e 5,0).
Outros funcionários foram avaliados abaixo da média (1.0 e 2.0).

Pergunta 4 - Qual a Distribuição das Idades dos Funcionários?

import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(dataset['idade'], color = 'darkgoldenrod')
plt.title('Distribuição de Idade dos Funcionários', fontsize = 15)
plt.xlabel('Idade', fontsize = 15)
plt.ylabel('Total', fontsize = 12)
plt.show()

Respostas

A maioria dos funcionários encontram-se com a idade entre 30 e 40.
Um número pequeno de funcionários com a idade menor ou igual a 20.
Um número pequeno de funcionários com a idade entre 50 e 60.

Pergunta 5 - Qual o Número de Treinamentos mais Frequentes?

import matplotlib.pyplot as plt
import seaborn as sns
sns.violinplot(dataset['numero_treinamentos'], color = 'crimson')
plt.title('Número de Treinamentos Realizados Pelos Funcionários', fontsize = 15)
plt.xlabel('Número de Treinamentos', fontsize = 15)
plt.ylabel('Frequência', fontsize = 12)
plt.show()

Respostas

A maioria dos funcionários fizeram apenas 1 curso.
Um número razoável de funcionários fizeram dois cursos.
Poucos funcionários fizeram 3 cursos.
Pouquíssimos funcionários fizeram 4 cursos.
Acima de 4 cursos nenhum funcionário realizou.

Pergunta 6 - Qual a Proporção de Funcionários Por Canal de Recrutamento?

dataset['canal_recrutamento'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
dataset['canal_recrutamento'].value_counts()
fatias = [55375, 42358, 2547]
labels = 'Outro', 'Outsourcing', 'Indicação'
colors = ['orangered', 'orange', 'gold']
explode = [0, 0, 0]
plt.pie(fatias, labels = labels, colors = colors, explode = explode, shadow = True, autopct = '%.2f%%')
plt.title('Percentual de Funcionários Por Canal de Recrutamento', fontsize = 15)
plt.axis('off')
plt.show()

Respostas

Um percentual acima de cinquenta porcento (%) pelo canal de recrutamento (outro).
Um percentual de um pouco mais de quarenta porcento (%) pelo canal de recrutamento (tercerização).
Um percentual bem pequeno pelo canal de recrutamento(indicação).

Pergunta 7 - Qual a Relação Entre a Promoção e a Avaliação do Ano Anterior?

data = pd.crosstab(dataset['aval_ano_anterior'], dataset['promovido'])
data

import matplotlib.pyplot as plt
import seaborn as sns
data = pd.crosstab(dataset['aval_ano_anterior'], dataset['promovido'])
data.div(data.sum(1).astype(float), axis = 0).plot(kind = 'bar',
                                                  stacked = True,
                                                  figsize = (15,9),
                                                  color = ['darkgreen', 'crimson'])
plt.title('Relação Entre a Avaliação do Ano Anterior e a Promoção', fontsize = 15)
plt.xlabel('Avaliação do Ano Anterior', fontsize = 15)
plt.legend()
plt.show()

Respostas

Quando a avaliação foi 1.0 muitos não foram promovidos e poucos foram promovidos.
Quando a avaliação foi 5.0 muitos foram promovidos e poucos foram promovidos.
Fim