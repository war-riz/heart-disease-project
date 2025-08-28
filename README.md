# Heart Disease Prediction Toolkit

## AI/ML Disease Prediction Project: Save Lives with AI

A comprehensive machine learning toolkit for predicting heart disease using patient medical data. This project demonstrates the power of AI in healthcare by achieving **85.3% accuracy** in heart disease prediction using Random Forest algorithms.

![Heart Disease Banner](https://img.shields.io/badge/Healthcare-AI-red) ![Machine Learning](https://img.shields.io/badge/ML-Classification-blue) ![Accuracy](https://img.shields.io/badge/Accuracy-85.3%25-green)

---

## Project Overview

Heart disease is one of the leading causes of death worldwide. This project builds machine learning models to predict the likelihood of heart disease in patients based on medical attributes, helping healthcare professionals make informed decisions and take preventive measures.

### Problem Statement
- **Challenge**: Early detection of heart disease to reduce mortality rates
- **Solution**: ML-powered prediction system using patient medical data
- **Impact**: Enable doctors to identify high-risk patients proactively

### Key Achievements
- **85.3% Accuracy** with Random Forest model
- **91.6% ROC-AUC Score** for excellent discrimination
- **Interactive prediction system** for real-world deployment
- **Feature importance analysis** for medical insights

---

## Dataset Information

**Source**: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Total Samples**: 920 patients
- **Features**: 16 medical attributes
- **Target**: Binary classification (Disease/No Disease)

### Key Features Analyzed:
- **Clinical Measurements**: Age, Blood Pressure, Cholesterol
- **Cardiac Indicators**: Maximum Heart Rate, ST Depression
- **Symptoms**: Chest Pain Type, Exercise-Induced Angina
- **Demographics**: Sex, Medical History

---

## Quick Start

### Prerequisites
```bash
# Required Python packages
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

### Installation
```bash
# Clone the repository
git clone https://github.com/war-riz/heart-disease-project.git
cd heart-disease-project

# Install dependencies
pip install -r requirements.txt

# Open the main notebook
jupyter notebook Heart_Disease_Prediction.ipynb
```

### Quick Prediction
```python
# Load the trained model
import joblib
import pandas as pd

# Load saved components
model = joblib.load('heart_rf_model.pkl')
scaler = joblib.load('heart_scaler.pkl')

# Make predictions on new data
predictions = model.predict(scaled_data)
probabilities = model.predict_proba(scaled_data)[:, 1]
```

---

## Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: Median imputation for numerical, mode for categorical
- **Feature Engineering**: One-hot encoding for categorical variables
- **Data Scaling**: StandardScaler for numerical features
- **Train-Test Split**: 80/20 with stratification

### 2. Model Training
Three algorithms were implemented and compared:
- **Logistic Regression**: Linear baseline model
- **Decision Tree**: Non-linear pattern recognition
- **Random Forest**: Ensemble method (best performer)

### 3. Model Evaluation
Comprehensive evaluation using:
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC Analysis**
- **Confusion Matrix Visualization**
- **Feature Importance Analysis**

---

## Results & Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **85.3%** | **84.4%** | **90.2%** | **87.2%** | **91.6%** |
| Logistic Regression | 84.2% | 84.1% | 88.2% | 86.1% | 90.3% |
| Decision Tree | 79.9% | 81.0% | 83.3% | 82.1% | 79.5% |

### Key Insights

#### Top 5 Most Important Features:
1. **Cholesterol Level** (14.5%) - Primary risk factor
2. **Age** (13.0%) - Strong age correlation
3. **Maximum Heart Rate** (12.7%) - Cardiac fitness indicator
4. **ST Depression** (11.0%) - ECG abnormality marker
5. **Exercise-Induced Angina** (9.1%) - Symptom indicator

#### Medical Interpretations:
- **High cholesterol** is the strongest predictor of heart disease
- **Older patients** show significantly higher risk
- **Reduced exercise capacity** strongly indicates heart problems
- **ECG abnormalities** provide crucial diagnostic information

---

## Interactive Prediction System

The toolkit includes an interactive system for real-world predictions:

### Features:
- **CSV Upload**: Easy patient data input
- **Automatic Preprocessing**: Handles missing values and encoding
- **Risk Assessment**: Low/Medium/High risk categorization
- **Probability Scores**: Detailed prediction confidence
- **Batch Processing**: Multiple patients simultaneously

### Usage Example:
```python
# Upload patient data using the provided template
user_data = pd.read_csv('patient_data.csv')

# Get predictions
predictions, probabilities, risk_levels = predict_heart_disease(user_data)

# Results include:
# - Binary prediction (0/1)
# - Probability score (0.0-1.0)
# - Risk level (Low/Medium/High)
```

---

## Repository Structure

```
heart-disease-project/
│
├── Heart_Disease_Prediction.ipynb    # Main analysis notebook
└── README.md                        # Project documentation
```

---

## Business Impact

### For Healthcare Providers:
- **Early Detection**: Identify high-risk patients before symptoms appear
- **Resource Optimization**: Focus attention on patients who need it most
- **Decision Support**: Data-driven insights for treatment planning
- **Cost Reduction**: Prevent expensive emergency interventions

### For Patients:
- **Preventive Care**: Take action before disease progression
- **Peace of Mind**: Clear risk assessment and monitoring
- **Personalized Treatment**: Tailored care based on risk factors

---

## Future Enhancements

### Technical Improvements:
- [ ] **Deep Learning Models**: Neural networks for complex pattern recognition
- [ ] **Real-time Monitoring**: Integration with wearable devices
- [ ] **Explainable AI**: SHAP values for better interpretability
- [ ] **Model Deployment**: Web application with REST API

### Medical Extensions:
- [ ] **Multi-class Prediction**: Severity levels and disease types
- [ ] **Temporal Analysis**: Disease progression over time
- [ ] **Drug Interaction**: Medication impact on predictions
- [ ] **Genetic Factors**: Incorporation of genetic risk markers

---

## Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Google Colab / Jupyter Notebook
- **Version Control**: Git & GitHub

---

## Model Validation

### Cross-Validation Results:
- **5-Fold CV Accuracy**: 83.7% (±2.1%)
- **Consistent Performance**: Low variance across folds
- **Robust Predictions**: Stable across different data splits

### Clinical Validation:
- **Sensitivity**: 90.2% (correctly identifies disease cases)
- **Specificity**: 78.9% (correctly identifies healthy cases)
- **PPV**: 84.4% (positive prediction accuracy)
- **NPV**: 87.1% (negative prediction accuracy)

---

## Contributing

We welcome contributions to improve the Heart Disease Prediction Toolkit!

### How to Contribute:
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution:
- Model improvements and new algorithms
- Additional visualization features
- Medical domain expertise and validation
- Documentation and tutorial enhancements

---

## Contact & Support

**Developer**: [Kehinde Waris]
- **Email**: kehindewaris54@gmail.com
- **LinkedIn**: [war-riz](https://www.linkedin.com/in/kehinde-waris-220426283?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
- **GitHub**: [@war-riz](https://github.com/war-riz)

**Project Links**:
- **Repository**: [GitHub Repo](https://github.com/war-riz/heart-disease-project)
- **Dataset**: [Kaggle Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

---

## License

This project is licensed under the MIT License

---

## Acknowledgments

- **Kaggle Community** for providing the heart disease dataset
- **Scikit-learn Team** for the excellent machine learning library
- **Healthcare Professionals** who inspire AI applications in medicine
- **DevTown Bootcamp** for the learning opportunity and project guidance

---

## References

1. World Health Organization. (2021). *Cardiovascular diseases (CVDs)*
2. American Heart Association. (2022). *Heart Disease and Stroke Statistics*
3. DevTown [LinkedIn](https://www.linkedin.com/company/66902474/admin/dashboard/)
4. Ishan Mishra 💼 [LinkedIn](https://www.linkedin.com/in/ihrm-ishan/) 📧 [Email](ihrm.aiml@gmail.com)
5. Janosi, A., Steinbrunn, W., et al. (1988). *Heart Disease Dataset*, UCI ML Repository
6. Breiman, L. (2001). *Random Forests*, Machine Learning, 45(1), 5-32

---

<div align="center">

** Star this repository if you found it helpful! **

*Making healthcare smarter with AI, one prediction at a time.*

</div>
