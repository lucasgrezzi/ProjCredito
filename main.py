
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

##Carregando os dados

credits_record = pd.read_csv("C:/Users/Lucas/Downloads/credit_record.csv")
aplications_record  = pd.read_csv("C:/Users/Lucas/Downloads/application_record.csv")

#Visualizando os dados

print(credits_record.head())
print(aplications_record.head())
print("Dados Nulos:\n", 
      aplications_record.isna().sum())

aplications_record.drop("OCCUPATION_TYPE", axis=1, inplace=True)

print("Dados Nulos:\n", 
      
      aplications_record.isna().sum())

print("Dados Nulos:\n", 
      
      credits_record.isna().sum())

print("Dados Nulos:\n",
      
      aplications_record.duplicated().sum())


print("Dados Nulos:\n",
      credits_record.duplicated().sum())


# Criando figura para os graficos

fig, axxs = plt.subplots(1, 
                         1, figsize=(14, 10))

# 1 . Histograma da Renda Total

sns.histplot(aplications_record["AMT_INCOME_TOTAL"].astype(int), bins = 40, kde=True, ax=axxs)
axxs.set_title("Distribuição da Renda")
axxs.set_xlabel("Renda Total")
axxs.set_ylabel("Contagem")
plt.show()


#Verificando as colunas que são categóricas para o aplication

cat_cols = aplications_record.select_dtypes(include=["object"]).columns
print(cat_cols)

#Converter vavriáveis categóricas em valores numéricos inteiros usando LabelEncoder


le= LabelEncoder()
for col in cat_cols:
    aplications_record[col] = le.fit_transform(aplications_record[col])

print(aplications_record.head())


## Criação das classes.
#Subsituindo os valores 'C' e 'X' pela classe 0 e 1 respectivamente
#1,2,3,4,5 são classificados como 1

credits_record['STATUS'].replace({'C': 0, 'X': 0}, inplace=True)
credits_record['STATUS'] = credits_record['STATUS'].astype('int')
credits_record['STATUS'] = credits_record['STATUS'].apply(lambda x: 1 if x == 2 else 0)
credits_record['STATUS'].value_counts(normalize=True)

credits_record.head(10)

#após isso o 0 será bom pagador e 1 mau pagador



##Agrupa o credit record po ID usando a função de agregação

crecordgb = credits_record.groupby('ID').agg(max).reset_index()
crecordgb.head()



##Faz a função com o application record


df = aplications_record.join(crecordgb.set_index('ID'), on='ID', 
                    how= 'inner')
df.head()
df.info()
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



##Escalamento de todas as variáveis para um intervalo entre 0 e 1

mms = MinMaxScaler()
X_scaled = pd.DataFrame(mms.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(mms.transform(X_test), columns=X_test.columns)




## SMOTE cria novos exemplos sintéticos da classe minoritária, selecionando aleatoriamente alguns vizinhos 
# proximos e criando novos exemplos com base na media desses vizinhos 

oversample = SMOTE()
X_balanced, y_balanced = oversample.fit_resample(X_scaled, y_train)
X_test_balanced, y_test_balanced = oversample.fit_resample(X_test, y_test)
y_train.value_counts(normalize=True)
print(y_train.value_counts(normalize=True))
print(y_balanced.value_counts(normalize=True))


##Criando o modelo GradientBoosting

model = GradientBoostingClassifier(random_state=123)



##Treinando o modelo

model.fit(X_balanced, y_balanced)
pred_y = model.predict(X_test_balanced)
print("\nMatriz de confusão:")
conf_matrix = confusion_matrix(y_test_balanced, pred_y)


##Visualização da matriz de confusão

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Bom Pagador', 'Mau Pagador'], yticklabels=['Bom Pagador', 'Mau Pagador'])
plt.xlabel("Previsao")
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()


##Acuracia

from sklearn.metrics import accuracy_score
print("Acuracia Score is {:.5}".format (accuracy_score(y_test_balanced, pred_y)))


## Obter importancias
##riando DataFrame para visualização

importancias = model.feature_importances_
feat_importances = pd.Series(importancias, index=X.columns)
feat_importances.sort_values().plot(kind='barh', figsize=(8, 6))
plt.title("Importância das Características no Gradient Boosting")
plt.xlabel('Importância')
plt.show()


from sklearn.metrics import accuracy_score
print("Acuracia Score is {:.5}".format (accuracy_score(y_test_balanced, pred_y)))


##Treinando o modelo

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_balanced, y_balanced)

##Plotar Arvore

plt.figure(figsize=(18, 8))
plot_tree(clf,feature_names=X.columns, class_names=['Bom', 'Ruim'], filled=True)
plt.title("Arvore de Decisão - Previsão de Inadimplência")
plt.show()



##Efetuando previsão
pred_y_arvore = clf.predict(X_test_balanced)



##Avaliando o modelo
print("\nMatriz de confusão:")
conf_matrixarvore = confusion_matrix(y_test_balanced, pred_y_arvore)
sns.heatmap(conf_matrixarvore, annot=True, fmt="d", cmap="Blues", xticklabels=['Bom Pagador', 'Mau Pagador'], yticklabels=['Bom Pagador', 'Mau Pagador'])
plt.xlabel("Previsao")
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

from sklearn.metrics import accuracy_score

print("Acuracia Score is {:.5}".format (accuracy_score(y_test_balanced, pred_y_arvore)))



## Definindo o modelo de votação
clf_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

#Criando o modelo de votação (assemble)

voting_clf = VotingClassifier(estimators=[
    ('Árvore de Decisão',clf_tree),
    ('Gradient Boosting', clf_gb)
], voting='soft')

voting_clf.fit(X_balanced, y_balanced)

y_pred_voting = voting_clf.predict(X_test_balanced)

accuracy = accuracy_score(y_test_balanced, y_pred_voting)
print(f"\n Acurácia do Modelo de Votação(assemble): {accuracy*100:2f}%")

#Matriz de confusão para análise final

conf_matrix = confusion_matrix(y_test_balanced, y_pred_voting)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=['Bom Pagador', 'Mau Pagador'], yticklabels=['Bom Pagador', 'Mau Pagador'])
plt.xlabel("Previsao")
plt.ylabel('Real')
plt.title('Matriz de Confusão - VotingClassifier (Essemble)')
plt.tight_layout()
plt.show()