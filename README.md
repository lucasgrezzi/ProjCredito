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
  
![image](https://github.com/user-attachments/assets/4f8eb933-5c36-4c1b-a1f3-5797c30ea46e)

  
- Matriz de confusão para avaliação de modelos;

![image](https://github.com/user-attachments/assets/20d32ffd-3de9-4214-9d5a-db8a7d54be7e)


- Gráfico de importância das variáveis;
  
![image](https://github.com/user-attachments/assets/2892cca6-0a49-4b6b-9107-2c7fc57a7792)


- Árvore de decisão plotada com critérios de divisão.

![image](https://github.com/user-attachments/assets/00f8e6f1-5096-485f-ba8e-16be4332d475)


- Validação do Modelo de confusão com ML

![image](https://github.com/user-attachments/assets/5ce7eba5-d454-4304-9b19-957adccdc27a)


- Matriz de confusão para análise final

![image](https://github.com/user-attachments/assets/540c461b-928a-48b2-bc7a-46a4e13c4fde)


---

## 📦 Instalação

1. Clone o repositório:

```bash
git clone https://github.com/lucasgrezzi/ProjCredito
cd ProjCredito
