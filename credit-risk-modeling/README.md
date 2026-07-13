# 💳 Credit Risk Modeling: Predição de Inadimplência em Cartões de Crédito

Projeto de Ciência de Dados desenvolvido para estimar a probabilidade de inadimplência de clientes de cartão de crédito utilizando técnicas de Machine Learning, Engenharia de Features e otimização de decisões orientadas ao negócio.

---

# 📌 Visão Geral

A concessão de crédito envolve decisões tomadas sob incerteza. Em instituições financeiras, estimar corretamente a probabilidade de inadimplência é fundamental para reduzir perdas financeiras, definir limites de crédito e otimizar o equilíbrio entre risco e retorno.

Neste projeto foi desenvolvido um modelo de classificação binária capaz de estimar a probabilidade de inadimplência de clientes de cartão de crédito no mês subsequente, utilizando informações demográficas, limite de crédito e histórico recente de comportamento financeiro.

Além da construção do modelo preditivo, também foi realizada uma etapa de otimização do threshold de decisão baseada em custos de negócio, aproximando a solução de um cenário real de concessão de crédito.

---

# 🎯 Objetivo

Desenvolver um modelo capaz de estimar:

P(Default = 1 | X)

onde:

- **Default = 1** → cliente inadimplente no próximo mês;
- **X** representa o conjunto de características disponíveis no momento da decisão.

Sob a perspectiva de negócio, o problema é análogo à estimação da **Probability of Default (PD)**, amplamente utilizada por instituições financeiras para apoiar decisões relacionadas à concessão de crédito e gestão de risco.

---

# 📐 Formulação Matemática

Para cada cliente \(i\), observa-se um vetor de características

\[
\mathbf{X}_i =
(x_{i1},x_{i2},\ldots,x_{ip}),
\]

composto por informações demográficas, limite de crédito e histórico recente de comportamento financeiro.

Define-se a variável resposta como

\[
Y_i =
\begin{cases}
1, & \text{se o cliente entrar em inadimplência no mês } t+1;\\
0, & \text{caso contrário.}
\end{cases}
\]

O objetivo do modelo consiste em estimar a probabilidade condicional

\[
P(Y_i=1 \mid \mathbf{X}_i).
\]

Na Regressão Logística, essa probabilidade é modelada por

\[
P(Y_i=1 \mid \mathbf{X}_i)
=
\frac{1}
{1+\exp\!\left[-\left(\beta_0+\boldsymbol{\beta}^{\top}\mathbf{X}_i\right)\right]}.
\]

Equivalentemente, pode-se escrever a função logit como

\[
\log
\left(
\frac{P(Y_i=1 \mid \mathbf{X}_i)}
{1-P(Y_i=1 \mid \mathbf{X}_i)}
\right)
=
\beta_0+\boldsymbol{\beta}^{\top}\mathbf{X}_i.
\]

---

# 🧠 Hipóteses de Negócio

O projeto foi conduzido a partir de três hipóteses principais:

### Hipótese 1

Clientes com atrasos mais recentes apresentam maior risco de inadimplência.

### Hipótese 2

Clientes que utilizam parcela elevada do limite de crédito apresentam maior probabilidade de default.

### Hipótese 3

Clientes com maior capacidade financeira (limite de crédito), *ceteris paribus*, apresentam menor risco de inadimplência.

Essas hipóteses orientaram toda a etapa de análise exploratória, engenharia de features e interpretação dos modelos.

---

# 📊 Dataset

**Fonte**

UCI Machine Learning Repository

**Base utilizada**

Default of Credit Card Clients

**Quantidade de observações**

30.000 clientes

**Variáveis disponíveis**

- Informações demográficas
- Limite de crédito
- Histórico de atrasos (6 meses)
- Valores faturados
- Valores pagos

**Variável alvo**

Default no mês subsequente.

---

# ⚙️ Metodologia

O projeto foi desenvolvido seguindo um fluxo completo de Machine Learning.

## 1. Análise Exploratória

- análise univariada
- análise bivariada
- análise das distribuições
- avaliação do comportamento da variável alvo
- investigação das hipóteses de negócio

## 2. Engenharia de Features

Foram construídas novas variáveis capazes de representar o comportamento financeiro dos clientes, entre elas:

- max_delay
- avg_delay
- num_delays
- severe_delay
- avg_bill
- avg_payment
- payment_ratio
- negative_bill_flag

Além disso, foram realizados tratamentos para evitar inconsistências numéricas, valores infinitos e vazamento de informação.

## 3. Modelagem

Foram avaliados quatro modelos:

- Dummy Classifier
- Logistic Regression
- Random Forest
- Gradient Boosting

## 4. Validação

A avaliação dos modelos foi realizada utilizando:

- Holdout Estratificado (70/30)
- Validação Cruzada Estratificada (5 folds)

## 5. Métricas

As principais métricas utilizadas foram:

- ROC-AUC
- Recall
- Precision
- F1-Score
- Matriz de Confusão

Como a base apresenta desbalanceamento entre as classes, a acurácia foi utilizada apenas como métrica complementar.

---

# 💼 Otimização para Negócio

Após a seleção do melhor modelo, foi realizada uma etapa adicional de otimização do threshold de decisão.

Foi construída uma função de custo que penaliza de forma mais severa os falsos negativos (clientes inadimplentes aprovados) do que os falsos positivos (bons clientes recusados).

Essa abordagem permitiu transformar probabilidades em decisões mais alinhadas ao contexto real de concessão de crédito.

---

# 📈 Resultados

## Comparação entre modelos

| Modelo | ROC-AUC |
|---------|---------:|
| Logistic Regression | **0.756** |
| Random Forest | **0.755** |
| **Gradient Boosting** | **0.780** |

Após validação cruzada:

| Modelo | ROC-AUC Médio |
|---------|--------------:|
| Logistic Regression | **0.764 ± 0.009** |
| Random Forest | **0.763 ± 0.009** |
| **Gradient Boosting** | **0.784 ± 0.008** |

O modelo **Gradient Boosting** apresentou o melhor desempenho tanto no conjunto de teste quanto na validação cruzada.

Após a otimização do threshold baseada em custos de negócio, o modelo alcançou aproximadamente:

- **ROC-AUC:** 0.780
- **Recall:** 89,7%
- **Precision:** 29%
- **Threshold otimizado:** 0.102

---

# 📊 Principais Variáveis

A análise de importância das variáveis mostrou que o comportamento recente do cliente possui impacto muito superior às características demográficas.

As principais variáveis foram:

| Variável | Importância |
|-----------|------------:|
| pay_0 | 0.528 |
| num_delays | 0.136 |
| max_delay | 0.054 |
| severe_delay | 0.035 |
| avg_delay | 0.033 |

Esses resultados reforçam a hipótese de que informações recentes sobre comportamento de pagamento são os principais determinantes do risco de inadimplência.

---

# 🚀 Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

# 📂 Estrutura do Repositório

```text
credit-risk-modeling/

│
├── data/
├── notebooks/
├── images/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

# ▶️ Como Executar

Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/credit-risk-modeling.git
```

Instale as dependências

```bash
pip install -r requirements.txt
```

Abra o notebook principal e execute as células sequencialmente.

---

# 📌 Conclusões

Este projeto demonstra um fluxo completo de desenvolvimento de modelos preditivos para risco de crédito, desde a compreensão do problema de negócio até a avaliação final dos modelos.

Ao longo do projeto foram aplicadas técnicas de análise exploratória, engenharia de features, comparação de modelos, validação cruzada, otimização de hiperparâmetros e seleção de threshold baseada em custos.

O modelo de **Gradient Boosting** apresentou o melhor desempenho entre os modelos avaliados, alcançando **ROC-AUC próximo de 0,78**, mantendo estabilidade durante a validação cruzada e apresentando elevada capacidade de identificação de clientes inadimplentes após a otimização do threshold.

Os resultados também evidenciaram que variáveis relacionadas ao comportamento recente de pagamento possuem influência significativamente maior do que características demográficas na previsão de inadimplência, reforçando uma das principais hipóteses investigadas ao longo do estudo.

Mais do que desenvolver um modelo com bom desempenho estatístico, o projeto buscou aproximar a modelagem de um contexto real de negócio, utilizando critérios de avaliação compatíveis com problemas de concessão de crédito e incorporando uma estratégia de decisão orientada ao custo financeiro dos erros de classificação.

---

## 👨‍💻 Autor

**Andrew Fraga**

Economista | Ciência de Dados | Machine Learning | Analytics

LinkedIn: *(www.linkedin.com/in/andrew-santos-fraga-43170a213)*
