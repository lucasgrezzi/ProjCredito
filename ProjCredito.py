#predicao-de-inadinplencia

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


credits_record = pd.read_csv("C:/Users/Lucas/Downloads/credit_record.csv")
aplications_record  = pd.read_csv("C:/Users/Lucas/Downloads/application_record.csv")


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


###plotar os graficos

fig, axxs = plt.subplots(1, 
                         1, figsize=(14, 10))

sns.histplot(aplications_record["AMT_INCOME_TOTAL"].astype(int), bins = 40, kde=True, ax=axxs)
axxs.set_title("Distribuição da Renda")
axxs.set_xlabel("Renda Total")
axxs.set_ylabel("Contagem")

cat_cols = aplications_record.select_dtypes(include=["object"]).columns
print(cat_cols)

le= LabelEncoder()
for col in cat_cols:
    aplications_record[col] = le.fit_transform(aplications_record[col])

print(aplications_record.head())





credits_record['STATUS'].replace({'C': 0, 'X': 0}, inplace=True)
credits_record['STATUS'] = credits_record['STATUS'].astype('int')
credits_record['STATUS'] = credits_record['STATUS'].apply(lambda x: 1 if x == 2 else 0)
credits_record['STATUS'].value_counts(normalize=True)

credits_record.head(10)





crecordgb = credits_record.groupby('ID').agg(max).reset_index()
crecordgb.head()

df = aplications_record.join(crecordgb.set_index('ID'), on='ID', 
                    how= 'inner')

df.head()
df.info()