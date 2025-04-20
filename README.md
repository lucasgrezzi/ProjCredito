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
- Matriz de confusão para avaliação de modelos;
- Gráfico de importância das variáveis;
- Árvore de decisão plotada com critérios de divisão.

---

## 📦 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
