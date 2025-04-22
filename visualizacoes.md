## 📈 Visualizações e Explicações

- Histograma da renda dos clientes;
  
![image](https://github.com/user-attachments/assets/4f8eb933-5c36-4c1b-a1f3-5797c30ea46e)

A distribuição de renda visualizada neste histograma revela uma forte assimetria à direita, indicando uma significativa concentração de indivíduos nas faixas de renda mais baixas e uma longa cauda de indivíduos com rendas elevadas. Essa característica sugere uma expressiva desigualdade na distribuição da renda. A média pode ser influenciada pelos valores extremos, tornando a mediana uma medida mais robusta para representar a renda típica. 

  
- Matriz de confusão para avaliação de modelos;

![image](https://github.com/user-attachments/assets/20d32ffd-3de9-4214-9d5a-db8a7d54be7e)

A matriz de confusão do modelo de classificação de risco de crédito apresenta os seguintes resultados: [coloque os valores de VP, FN, FP, VN]. Observa-se um bom número de Verdadeiros Positivos (9885) e Verdadeiros Negativos (8833), indicando que o modelo acerta em grande parte das suas previsões. No entanto, existem erros de classificação, sendo 893 Falsos Negativos (bons pagadores classificados como maus) e 1945 Falsos Positivos (maus pagadores classificados como bons). A maior quantidade de Falsos Positivos pode indicar uma tendência do modelo em ser mais 'otimista' na classificação de bons pagadores. A análise do custo associado a cada tipo de erro é crucial para avaliar a adequação do modelo para o problema em questão.

- Gráfico de importância das variáveis;
  
![image](https://github.com/user-attachments/assets/2892cca6-0a49-4b6b-9107-2c7fc57a7792)

Este gráfico de barras horizontais ilustra a importância das diferentes características utilizadas no modelo de Gradient Boosting. As características CNT_CHILDREN, AMT_CREDIT, AMT_ANNUITY e AMT_GOODS_PRICE se destacam como as mais importantes para as previsões do modelo, indicando que o número de filhos e informações financeiras relacionadas ao crédito têm um peso significativo na decisão do modelo. Por outro lado, variáveis como FLAG_MOBIL e FLAG_WORK_PHONE apresentam uma importância muito baixa. Essa análise de importância das características é fundamental para entender o comportamento do modelo e pode guiar futuras etapas de engenharia de features e seleção de variáveis.

- Árvore de decisão plotada com critérios de divisão.

![image](https://github.com/user-attachments/assets/00f8e6f1-5096-485f-ba8e-16be4332d475)

Esta árvore de decisão para a previsão de inadimplência revela as regras utilizadas pelo modelo para classificar os clientes. A primeira divisão é baseada em MONTHS_BALANCE, sugerindo que o histórico de saldo da conta é um fator crucial na determinação do risco de inadimplência. Valores mais negativos de MONTHS_BALANCE parecem levar a um maior risco de ser classificado como 'mau pagador'. Outras variáveis importantes incluem a renda total (AMT_INCOME_TOTAL), o valor do crédito (AMT_CREDIT) e o número de filhos (CNT_CHILDREN). A estrutura da árvore permite visualizar as combinações de condições que resultam em diferentes previsões de inadimplência.

- Validação do Modelo de confusão com ML

![image](https://github.com/user-attachments/assets/5ce7eba5-d454-4304-9b19-957adccdc27a)

Esta matriz de confusão representa os resultados da validação do modelo de Machine Learning para a classificação de risco de crédito. Os resultados em dados não vistos durante o treinamento são cruciais para avaliar a capacidade de generalização do modelo. Observamos um bom desempenho na identificação de Verdadeiros Negativos (9407), indicando que o modelo consegue identificar corretamente muitos 'Mau Pagadores' em dados novos. No entanto, a alta taxa de Falsos Negativos (7009) sugere que o modelo tende a classificar muitos 'Bons Pagadores' como 'Mau Pagador' nos dados de validação, o que pode ter implicações significativas para o negócio. A comparação com a matriz de confusão obtida nos dados de treinamento (se disponível) seria importante para verificar a consistência do desempenho e identificar possíveis problemas de overfitting ou underfitting. Ajustes no modelo podem ser necessários para otimizar o equilíbrio entre a precisão e o recall, dependendo dos custos associados aos Falsos Positivos e Falsos Negativos no contexto da aplicação.

- Matriz de confusão para análise final

![image](https://github.com/user-attachments/assets/540c461b-928a-48b2-bc7a-46a4e13c4fde)

A matriz de confusão final, obtida através da validação com o VotingClassifier (Ensemble), apresenta os seguintes resultados: [insira os valores de VP, FN, FP, VN]. O VotingClassifier, que combina as previsões de múltiplos modelos base, demonstra um desempenho promissor na classificação de risco de crédito. Observamos [mencione os valores de VP e VN], indicando uma boa capacidade de identificar corretamente tanto 'Bons Pagadores' quanto 'Mau Pagadores'. Ao analisar os erros, notamos [mencione os valores de FN e FP]. É importante comparar esses resultados com as matrizes de confusão dos modelos individuais para avaliar o impacto positivo da técnica de ensemble na melhoria da performance geral e no balanceamento dos tipos de erro. A análise final deve considerar os custos associados aos Falsos Negativos e Falsos Positivos no contexto do negócio para determinar se o desempenho do VotingClassifier é satisfatório para a aplicação.

---
