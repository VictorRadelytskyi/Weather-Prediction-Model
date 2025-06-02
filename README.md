# Weather Prediction Model 

## Project Overview
This machine learning model classifies weather conditions into four categories:
- Sunny
- Rainy
- Snowy 
- Cloudy

The system achieves **~92% accuracy** using a Random Forest classifier with optimized hyperparameters.

## Model Performance

### Confusion matrix
![confusion_matrix](https://github.com/user-attachments/assets/1a13c295-148b-4874-a42e-65b70310114f)

### Classification Report

| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Cloudy  | 0.87      | 0.91   | 0.89     | 666     |
| Rainy   | 0.89      | 0.91   | 0.90     | 660     |
| Snowy   | 0.95      | 0.91   | 0.93     | 693     |
| Sunny   | 0.92      | 0.90   | 0.91     | 621     |


### Feature importance
![feature_importance](https://github.com/user-attachments/assets/f1ac34bf-18fb-4078-9390-d79e4e69f81b)

## Technical Implementation

### Data Processing
- Label encoding for categorical features
- Train-test split (80%-20%)
  
### Model Training
- Random Forest classifier
- Best parameters:
  ```python
  {'max_depth': 20,
   'min_samples_leaf': 1,
   'min_samples_split': 2,
   'n_estimators': 100}
