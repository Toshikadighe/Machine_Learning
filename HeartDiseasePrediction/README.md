# Heart Disease Prediction

Predicts presence of heart disease from 918 patient records using clinical features.

## Approach
- Identified disguised missing values and imputed using median/mean
- One-hot encoded categorical clinical features
- Standardized numerical features
- Compared 5 classifiers: Logistic Regression, KNN, Decision Tree, SVM, 
  Naive Bayes

## Results
| Model               | Accuracy | Precision (disease) | Recall (disease) |
|---------------------|----------|----------------------|-------------------|
| Logistic Regression | 86.4%    | 0.91                 | 0.85              |
| KNN                 | 84.8%    | 0.88                 | 0.85              |
| SVM                 | 84.2%    | 0.87                 | 0.86              |
| Naive Bayes         | 84.8%    | 0.91                 | 0.82              |
| Decision Tree       | 78.8%    | 0.84                 | 0.79              |
