# 🏠 House Prices Prediction — Advanced Regression Analysis

Este projeto apresenta uma solução completa de **Machine Learning aplicada à previsão de preços de imóveis**, com foco em **boas práticas estatísticas, validação robusta e engenharia de features**.

## 📌 Objetivo
Prever o preço de venda de imóveis residenciais utilizando técnicas modernas de regressão e ensembles, explorando diferentes vieses estatísticos e avaliando performance via **RMSE**.

---

## 🧠 Abordagem Estatística

### 🔹 Tratamento da variável alvo
- Aplicação de **transformação logarítmica** no target para:
  - Reduzir assimetria
  - Melhorar estabilidade do modelo
  - Atender pressupostos de modelos lineares

### 🔹 Análise de assimetria
- Identificação de variáveis numéricas altamente assimétricas
- Correção via transformações baseadas em skewness

---

## ⚙️ Engenharia de Features

- Separação entre variáveis:
  - Numéricas → `StandardScaler`
  - Categóricas → `OneHotEncoder` e `OrdinalEncoder`
- Uso de `ColumnTransformer` para garantir pipeline limpo e reproduzível

---

## 🤖 Modelos Avaliados

- **Modelos Lineares**
  - Ridge Regression
  - Lasso Regression

- **Modelos Baseados em Árvores**
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost

Essa diversidade permite capturar tanto relações lineares quanto não lineares complexas.

---

## 📐 Validação e Métrica

- Validação cruzada com:
  - `KFold`
  - `RepeatedKFold`
- Otimização via `GridSearchCV`
- Métrica principal:
  - **RMSE (Root Mean Squared Error)**

---

## 🚀 Como executar

```bash
pip install -r requirements.txt
