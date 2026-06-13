# Loan Approval Prediction – ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

A comprehensive machine learning pipeline designed to predict loan approval based on applicant and loan attributes. This project demonstrates end-to-end ML workflow including data preprocessing, exploratory data analysis, model training, evaluation, and comparison.

**Target Variable**: `loan_status` (1 = Approved, 0 = Rejected)

---

## 📊 Dataset

| Metric | Value |
|--------|-------|
| **Number of Rows** | 45,000 |
| **Number of Features** | 13 |
| **Data Type** | Mixed (Numerical & Categorical) |
| **Class Distribution** | 78% Rejected, 22% Approved |
| **Missing Values** | Handled via imputation |

### Key Features
- Income
- Credit Score
- Loan Amount
- Employment History
- Debt-to-Income Ratio
- And more...

---

## 🔧 Preprocessing Pipeline

```
Raw Data → Missing Value Imputation → Encoding → Scaling → Train-Test Split
```

### Steps:
1. **Missing Value Imputation**
   - Median imputation for numerical features
   - Mode imputation for categorical features

2. **Categorical Encoding**
   - Label encoding for categorical variables

3. **Feature Scaling**
   - Min-Max scaling (0-1 normalization) for numerical features

4. **Data Splitting**
   - Stratified train-test split (70% training, 30% testing)
   - Preserves class distribution in both sets

---

## 🤖 Models Evaluated

| Model | Accuracy | Status |
|-------|----------|--------|
| **Logistic Regression** | 0.89 | Competitive & Interpretable |
| **K-Nearest Neighbors (KNN)** | 0.89 | Sensitive to scaling |
| **Decision Tree** | 0.90 | Risk of overfitting |
| **Neural Network (MLP)** | **0.92** | ⭐ **Best Performer** |
| **KMeans** | - | Unsupervised clustering |

---

## 📈 Results & Performance

### Best Model: Multi-Layer Perceptron (MLP)
- **Accuracy**: 92%
- **Outperformed**: All other models
- **Key Advantages**: 
  - Highest accuracy
  - Best AUC score
  - Handles non-linear patterns effectively

### Model Comparison Insights
- **Logistic Regression**: Strong baseline with good interpretability
- **Decision Tree**: Competitive but shows signs of overfitting
- **KNN**: Sensitive to feature scaling and class imbalance
- **MLP**: Captures complex patterns and delivers best results

---

## 🚀 How to Use

### Prerequisites
```bash
pip install pandas numpy scikit-learn tensorflow jupyter
```

### Running the Pipeline
1. Clone this repository
2. Navigate to the project directory
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
4. Run all cells to execute the complete ML pipeline

### Making Predictions
```python
# After training, use the model to predict new loan applications
prediction = trained_model.predict(new_applicant_data)
```

---

## 📁 Project Structure

```
├── README.md                          # Project documentation
├── Loan_Approval_ML_Pipeline.ipynb   # Main notebook with full pipeline
├── data/                              # Dataset directory
│   └── loan_data.csv                 # Raw loan approval data
└── models/                            # Trained models (optional)
```

---

## 🔍 Key Findings

✅ **Successfully developed** a predictive model for loan approval with 92% accuracy

✅ **Identified best model**: MLP outperformed traditional ML algorithms

✅ **Addressed class imbalance**: Pipeline handles the 78-22 rejection-approval split

✅ **Scalable pipeline**: Can be adapted for other classification problems

---

## 🛠️ Future Improvements

- [ ] **Class Balancing**: Implement SMOTE or class weights to handle imbalance
- [ ] **Feature Engineering**: Create new features to improve model performance
- [ ] **Hyperparameter Tuning**: Grid search & cross-validation optimization
- [ ] **Ensemble Methods**: Combine multiple models for better predictions
- [ ] **Model Deployment**: Create API/Web interface for real-time predictions
- [ ] **Explainability**: Add SHAP or LIME for model interpretability

---

## 📚 Technologies Used

- **Language**: Python 3.8+
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow, Matplotlib, Seaborn
- **Environment**: Jupyter Notebook

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Ashraful Alam**  
GitHub: [@AshrafulAlam07](https://github.com/AshrafulAlam07)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/AshrafulAlam07/A-Machine-Learning-Pipeline-for-Loan-Approval-Prediction/issues).

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a star! Your support means a lot! 🌟
