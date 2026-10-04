# Ford Car Price Prediction

Predicts used Ford car prices using UK listings data (18K+ records).

## Approach
- Cleaned data: removed duplicates, fixed invalid year, imputed 
  missing engine sizes using per-model median
- Handled outliers in mileage via IQR clipping
- Grouped rare models/fuel types into "Other" to reduce sparsity
- Engineered `car_age` from `year`; one-hot encoded categoricals
- Log-transformed price to address right-skew
- Compared Linear, Ridge, Lasso regression with CV-tuned alpha

## Results
| Model     | MAE   | RMSE  | R²   |
|-----------|-------|-------|------|
| Linear    | 1158  | 1632  | 0.882|
| RidgeCV   | 1158  | 1632  | 0.882|
| LassoCV   | 1243  | 1729  | 0.867|

Best model: RidgeCV (alpha=10)

## Tech stack
Python, pandas, NumPy, scikit-learn, seaborn, matplotlib
