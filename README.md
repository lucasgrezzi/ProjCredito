# 💳 Predição de Inadimplência com Machine Learning

Este projeto tem como objetivo prever a inadimplência de clientes com base em dados de aplicação para crédito e histórico de pagamentos. Usamos algoritmos de machine learning supervisionado para construir modelos preditivos que classificam os clientes como **bons** ou **maus pagadores**.

---

## 📂 Conjuntos de Dados

Foram utilizados dois arquivos públicos:

- `application_record.csv`: Contém dados pessoais e socioeconômicos dos clientes.
- `credit_record.csv`: Contém o histórico de pagamentos dos clientes com status mensal de crédito.

---

## 🔧 Etapas do Projeto

### 1. Carregamento e limpeza de dados
- Remoção de colunas com muitos valores ausentes (`OCCUPATION_TYPE`);
- Eliminação de duplicatas;
- Conversão de variáveis categóricas para numéricas com `LabelEncoder`.

### 2. Análise exploratória (EDA)
- Visualização da distribuição da renda total;
- Verificação de desequilíbrio das classes.

### 3. Pré-processamento
- Junção dos datasets via `ID`;
- Criação da variável target (`STATUS`: 0 = bom pagador, 1 = mau pagador);
- Normalização com `MinMaxScaler`.

### 4. Balanceamento com SMOTE
- Uso do `SMOTE` para gerar dados sintéticos da classe minoritária.

### 5. Modelagem
- Modelos utilizados:
  - `GradientBoostingClassifier`
  - `DecisionTreeClassifier`
  - `VotingClassifier`
  
- Treinamento e avaliação com matriz de confusão e acurácia.

### 6. Interpretação
- Importância das variáveis no modelo de Gradient Boosting;
- Visualização da árvore de decisão.

---

## 🤖 Modelos de Machine Learning

| Modelo                   | Acurácia (exemplo) | Vantagens                            |
|--------------------------|--------------------|--------------------------------------|
| Gradient Boosting        | 93%+               | Alta performance e robustez         |
| Decision Tree (profunda) | 87%+               | Interpretação visual fácil           |

> *Os valores exatos de acurácia podem variar conforme o split do dataset.*

---

## 📈 Visualizações

- Histograma da renda dos clientes;
  
![Distribuição de renda](ProjCredito/graphics/distribuicao_de-renda.png)
  
- Matriz de confusão para avaliação de modelos;

![Matriz de confusão](ProjCredito/graphics/matriz_confusao.png)

- Gráfico de importância das variáveis;
  
![Importância das variáveis](ProjCredito/graphics/gradient_boosting.png)

- Árvore de decisão plotada com critérios de divisão.

![Árvore de decisão](ProjCredito/graphics/previsao_inadeimplencia.png)

- Validação do Modelo de confusão com ML

![Matriz de confusão (Machine Learning](ProjCredito/graphics/matriz_confusao(ml).png)

- Matriz de confusão para análise final

![Validação com o Voting Classifier](ProjCredito/graphics/voting_confusao.png)

---

## 📦 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/lucasgrezzi/ProjCredito
cd ProjCredito
