# Diabetes-Prediction-using-Machine-Learning-


## 📋 Overview

This project implements multiple machine learning algorithms to predict diabetes risk based on medical and demographic data. The system analyzes various health indicators to provide accurate predictions that could assist in early diabetes detection and prevention.

## ✨ Key Features

- **Multiple ML Models**: Comparative analysis of 7 different algorithms
- **Advanced Preprocessing**: Outlier detection, feature scaling, and class balancing using SMOTE
- **Feature Engineering**: Age grouping and Fisher Score-based feature selection
- **Deep Learning**: Neural Network implementation with MLP architecture
- **Comprehensive Evaluation**: Detailed performance metrics and visualization

## 🎯 Project Highlights

- **Best Model**: Random Forest achieved **95.93% accuracy**
- **Dataset Size**: 100,000 records with 9 features
- **Balanced Classes**: SMOTE technique applied for class imbalance
- **ROC-AUC Scores**: All models achieved >0.90 AUC scores

## 📊 Dataset

### Features
- **Gender**: Male, Female
- **Age**: Converted to age groups (0-20, 21-40, 41-60, 61+)
- **Hypertension**: Binary indicator
- **Heart Disease**: Binary indicator
- **Smoking History**: Never, Former, Current, No Info
- **BMI**: Body Mass Index
- **HbA1c Level**: Hemoglobin A1c percentage
- **Blood Glucose Level**: mg/dL

### Target Variable
- **Diabetes**: 0 (Non-diabetic) or 1 (Diabetic)

## 🔧 Technologies Used

### Core Libraries
```python
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Imbalanced-learn
```

### Visualization
```python
- Matplotlib
- Seaborn
```

### ML Algorithms
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Multi-Layer Perceptron (Neural Network)


## 📈 Model Performance

| Model | Training Accuracy | Testing Accuracy | Precision | Recall | F1-Score |
|-------|------------------|------------------|-----------|--------|----------|
| **Random Forest** | **98.62%** | **95.93%** | **95.94%** | **95.93%** | **95.93%** |
| KNN | 98.36% | 93.39% | 93.40% | 93.39% | 93.38% |
| Decision Tree | 92.07% | 91.19% | 91.21% | 91.19% | 91.19% |
| MLP (Neural Network) | 88.51% | 88.49% | 85.66% | 92.45% | 89.94% |
| SVM | 87.52% | 87.54% | 88.06% | 87.54% | 87.50% |
| Logistic Regression | 85.42% | 85.18% | 85.28% | 85.18% | 85.17% |
| Naive Bayes | 81.64% | 81.65% | 81.76% | 81.65% | 81.64% |

## 🔍 Data Preprocessing Pipeline

1. **Data Cleaning**
   - Removed 'Other' gender category (18 records)
   - Eliminated 3,844 duplicate entries
   - Final dataset: 86,526 samples

2. **Outlier Removal**
   - Applied IQR method for BMI, Blood Glucose Level, and HbA1c Level
   - Reduced dataset to 90,370 samples after outlier removal

3. **Feature Engineering**
   - Created age groups for better pattern recognition
   - Encoded categorical variables using Label Encoding
   - Applied Fisher Score for feature selection

4. **Class Balancing**
   - Original distribution: 82,039 (Non-diabetic) vs 4,487 (Diabetic)
   - After SMOTE: 82,039 vs 82,039 (Balanced)

5. **Scaling**
   - Applied MinMaxScaler for feature normalization

## 📊 Key Findings

### Most Important Features
1. **HbA1c Level** (Fisher Score: 0.0745)
2. **Blood Glucose Level** (Fisher Score: 0.0492)
3. **Age Group** (Fisher Score: 0.0448)

### Correlation with Diabetes
- HbA1c Level: 0.263
- Blood Glucose Level: 0.217
- Age Group: 0.207

## 🧠 Neural Network Architecture

```
Input Layer (6 features)
    ↓
Dense Layer (512 neurons, ReLU) + Dropout (0.2)
    ↓
Dense Layer (256 neurons, ReLU) + Dropout (0.2)
    ↓
Dense Layer (256 neurons, ReLU) + Dropout (0.2)
    ↓
Dense Layer (256 neurons, ReLU) + Dropout (0.2)
    ↓
Dense Layer (128 neurons, ReLU) + Dropout (0.2)
    ↓
Dense Layer (64 neurons, ReLU) + Dropout (0.2)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Total Parameters**: 307,713


## 🎨 Visualizations

The project includes comprehensive visualizations:
- Distribution plots for numerical features
- Count plots for categorical features
- Correlation heatmaps
- ROC-AUC curves for all models
- Confusion matrices
- Classification reports
- Training history plots for Neural Network

## 🔮 Future Improvements

- [ ] Implement advanced deep learning architectures (CNN, LSTM)
- [ ] Deploy model as a web application using Flask/FastAPI
- [ ] Add real-time prediction capabilities
- [ ] Integrate with medical database systems
- [ ] Perform hyperparameter tuning with Optuna
- [ ] Add explainability with SHAP values
- [ ] Validate with real-world clinical data


## 📧 Contact

**Abu Taher**
. LinkedIn: https://www.linkedin.com/in/abu-taher-adi-/ 
. Email: abutaher643309@gmail.com 
. Kaggle: https://www.kaggle.com/adicse09



## 🙏 Acknowledgments

- Dataset source: Kaggle
- Inspired by medical research in diabetes prediction
- Thanks to the open-source community for the amazing tools

