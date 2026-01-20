# House Prices Prediction - Machine Learning

Projeto de Machine Learning para previsão de preços de imóveis utilizando o dataset da competição [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Sobre o Projeto

Este projeto implementa um pipeline completo de Machine Learning para prever preços de casas residenciais em Ames, Iowa. O objetivo é praticar técnicas de regressão, feature engineering e otimização de hiperparâmetros.

## Dataset

- **Treino:** 1.460 amostras com 81 features
- **Teste:** 1.459 amostras
- **Target:** `SalePrice` (preço de venda)
- **Preço médio:** $180.921
- **Range:** $34.900 - $755.000

### Features Principais (maior correlação com preço)

| Feature | Correlação |
|---------|------------|
| OverallQual | 0.79 |
| GrLivArea | 0.71 |
| GarageCars | 0.64 |
| GarageArea | 0.62 |
| TotalBsmtSF | 0.61 |

## Pipeline de Dados

```
┌─────────────────────────────────────────────────────────────┐
│                    ColumnTransformer                        │
├─────────────────────────┬───────────────────────────────────┤
│   Numerical Pipeline    │      Categorical Pipeline         │
│  ┌───────────────────┐  │  ┌─────────────────────────────┐  │
│  │ SimpleImputer     │  │  │ SimpleImputer               │  │
│  │ (median)          │  │  │ (constant='missing')        │  │
│  └────────┬──────────┘  │  └──────────────┬──────────────┘  │
│  ┌────────▼──────────┐  │  ┌──────────────▼──────────────┐  │
│  │ StandardScaler    │  │  │ OneHotEncoder               │  │
│  └───────────────────┘  │  │ (handle_unknown='ignore')   │  │
│                         │  └─────────────────────────────┘  │
└─────────────────────────┴───────────────────────────────────┘
```

## Modelos Testados

| Modelo | CV RMSE | Desvio Padrão |
|--------|---------|---------------|
| Linear Regression | $47.245 | ± $30.813 |
| Decision Tree | $39.239 | ± $3.334 |
| **Random Forest** | **$28.040** | **± $4.053** |

## Otimização de Hiperparâmetros

Utilizei `RandomizedSearchCV` para otimizar o Random Forest:

```python
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
}
```

### Melhores Parâmetros Encontrados

- `n_estimators`: 181
- `max_depth`: None
- `min_samples_split`: 9
- `min_samples_leaf`: 2

### Resultado Final

- **RMSE (Cross-Validation):** $27.988
- **RMSE (Validação):** $40.869

## Estrutura do Projeto

```
house-prices-ml/
├── main.ipynb           # Notebook principal com análise e modelagem
├── train.csv            # Dados de treino
├── test.csv             # Dados de teste
├── data_description.txt # Descrição das features
├── sample_submission.csv
└── README.md
```

## Tecnologias Utilizadas

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
  - Pipeline
  - ColumnTransformer
  - RandomizedSearchCV
  - RandomForestRegressor

## Análises Realizadas

1. **Análise Exploratória (EDA)**
   - Distribuição do preço de venda
   - Matriz de correlação
   - Identificação de outliers

2. **Pré-processamento**
   - Tratamento de valores missing
   - Encoding de variáveis categóricas
   - Normalização de variáveis numéricas

3. **Modelagem**
   - Comparação entre modelos
   - Cross-validation (5-fold)
   - Tuning de hiperparâmetros

## Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/house-prices-ml.git
cd house-prices-ml
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

4. Execute o notebook:
```bash
jupyter notebook main.ipynb
```

## Próximos Passos

- [ ] Feature Engineering avançado
- [ ] Testar XGBoost e LightGBM
- [ ] Implementar Stacking/Ensemble
- [ ] Submeter no Kaggle

## Autor

Desenvolvido como projeto de estudo em Machine Learning.

## Licença

Este projeto está sob a licença MIT.
