# Short_Project_2

# "Credit card Fraud Detection"

### Goal of the project :
To classify the given record of information Fraud or not. 0 representing the Not-Fraud , 1 representing fraud. 

### Number of possible ways to solve the given problem (Task : Binary Classification)
- Random Forest Classifier
- Logistic Regression
- Support Vector Machines
- Kernel SVM
- Naive Bayes
- Decision Tree Classification

The Methods used for this project are __Logistic Regression__ , __Random Forest Classifier__


### Dataset From Kaggle : 
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud


### Files Description
Total : 4 files
- 1_model_development.ipynb 
    - Starting file and the only file that contains the main code
- ruf1.ipynb
    - contains just ruf work.
- model_LogisticRegression-Credit-card-fraud-detection.pkl
    - Logistic Regression Model stored on disc in pickle format
- model_RandomForest-Credit-card-fraud-detection.pkl
    - Random Forest Classifier model stored on disc in pickle format

---
### Technologies used :
- Pandas
- Matplotlib
- sklearn
- seaborn
- missingno
- warnings

### Sklearn :
- train_test_split
```python
from sklearn.model_selection import train_test_split
```

- StandardScalar
```python
from sklearn.preprocessing import StandardScaler
```

- LogisticRegression
```python
from sklearn.linear_model import LogisticRegression
```

- RandomForestClassifier
```python
from sklearn.ensemble import RandomForestClassifier
```
    
- accuracy_score , confusion_matrix , classification_report
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Topics :
- Feature Scaline , Standardization 
- Random Forest Machine Learning Algorithm 
- Logistic Regression , Binary Classification
- Tabular data
- Evaluation Metrics , Confussion Matrix , accuracy
- Precision , Recall , F1-score , support
- Data Visualization , Count-Plot , Heat-Map
- Correlation Matrix