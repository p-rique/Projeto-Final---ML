# Importação das bibliotecas necessárias e o dataset para o projeto

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score)

sns.set_theme(style = 'whitegrid') # Tema para a visualização

#print("Bibliotecas importadas com sucesso!")

# Carregando o dataset

wine = load_wine()

df = pd.DataFrame(wine.data, columns = wine.feature_names)
df['target'] = wine.target

print(df.head()) # Apresentando as primeiras linhas e dimensões (número de linhas x colunas) do dataset

# Inspeção do dataset

print("Dimensões:", df.shape)
print("\nTipos:")
print(df.dtypes)

print("\nValores ausentes:")
print(df.isna().sum())

print("\nDuplicatas:", df.duplicated().sum())

print("\nEstatísticas:")
print(df.describe())

# Foi verificado que não há valores ausentes ou duplicatas, então, o dataset está limpo

# Análise e visualização dos dados
## Gráfico de barras do número de amostras por classe

print("Classes:", wine.target_names)
print(df["target"].value_counts().sort_index())

plt.figure(figsize = (8, 5))

sns.countplot(
    data=df, 
    x="target")

plt.title("Distribuição de amostras por classe") 

plt.xlabel("Classe")
plt.ylabel("Número de amostras")

plt.show()

## Gráfico de dispersão teor alcoólico x intensidade de cor

plt.figure(figsize = (8, 5))

sns.scatterplot(
    data = df,
    x = "alcohol",
    y = "color_intensity",
    hue = "target"
)

plt.title("Teor alcoólico x Intensidade de cor de cada classe")

plt.xlabel("Teor alcoólico")
plt.ylabel("Intensidade de cor")

plt.show()

# Separação entre features e target

X = df.drop(columns = "target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Treino: ", X_train.shape)
print("Teste: ", X_test.shape)

# Testando o Decision Tree Classifier

modelo_tree = DecisionTreeClassifier(
    max_depth = 3,
    random_state = 42
    )

modelo_tree.fit(X_train, y_train)

print("Modelo treinado!") # Verificando se o modelo foi treinado

# Avaliando o modelo (Decision Tree)
## Previsões do modelo (Decision Tree)

previsoes_tree = modelo_tree.predict(X_test)

print("Previsões do Decision Tree Classifier: ", previsoes_tree)

## Acurácia do modelo (Decision Tree)

acuracia_tree = accuracy_score(y_test, previsoes_tree)

print("Acurácia do Decision Tree Classifier: ", acuracia_tree)

## Acurácia do modelo é menor que 1, então, o modelo está underfitting.

# Aplicando a matriz de confusão

ConfusionMatrixDisplay.from_predictions(
    y_test,
    previsoes_tree,
    display_labels = wine.target_names
)

plt.title("Matriz de Confusão - Decision Tree Classifier")

plt.show()

# Buscando a profundidade ideal

profundidades = range(1, 11)

treino = []
teste = []

for profundidade in profundidades:
    modelo = DecisionTreeClassifier(max_depth = profundidade, random_state = 42)
    modelo.fit(X_train, y_train)
    treino.append(modelo.score(X_train, y_train))
    teste.append(modelo.score(X_test, y_test))

plt.figure(figsize = (11, 8))

# Gráfico de complexidade e desempenho do modelo

plt.figure(figsize = (8, 5))

plt.plot(
    profundidades,
    treino,
    marker = "o",
    label = "treino"
)

plt.plot(
    profundidades,
    teste,
    marker = "o",
    label = "teste"
)

plt.xlabel("Profundidade da árvore")
plt.ylabel("Acurácia")

plt.title("Complexidade e desempenho - Decision Tree Classifier")

plt.legend()

plt.show()

## Com base nos resultados do gráfico, a profundidade ideal para o modelo é 3, onde alcança o máximo desempenho sem overfitting.
## Buscando uma acurácia maior, vamos tentar usar outro modelo:

# Testando o Random Forest Classifier

modelo_forest = RandomForestClassifier(
    n_estimators = 100,
    max_depth = 5,
    random_state = 42
)

# Treinando o modelo (Random Forest)

modelo_forest = modelo_forest.fit(X_train, y_train)

# Avaliando o modelo (Random Forest)
## Previsões do modelo (Random Forest)

previsoes_forest = modelo_forest.predict(X_test)

## Acurácia do modelo (Random Forest)

acuracia_forest = accuracy_score(
    y_test,
    previsoes_forest
)

print("Acurácia do Random Forest Classifier: ", acuracia_forest)

## Alcançamos uma acurácia igual a 1.0, mas precisamos veriicar se o modelo está overfitting.

# Aplicando validação cruzada para verificar o desempenho do modelo (Random Forest)

scores = cross_val_score(
    modelo_forest,
    X,
    y,
    cv = 5 # Número de divisões para a validação cruzada
)

print("Resultados das divisões" , scores)
print("Acurácia média do modelo: ", scores.mean())
## Acurácia média igual a 0.972, sendo maior que a acurácia do modelo Decision tree.
## Além disso, a acurácia média é menor que 1.0, então, não está overitting.

# Testando a precisão do modelo (Random Forest)

precisao_forest = precision_score(
    y_test,
    previsoes_forest,
    average = "weighted"
)

print("Precisão do Random Forest Classifier: ", precisao_forest)
## A precisão do modelo é igual a 1.0, então, o modelo está performando bem.