# Waste Management System Using AI - Infosys Springboard AI Internship

## Project Overview
This project focuses on developing a classification model for waste management using sensor data. By leveraging machine learning and AI techniques, we analyze sensor readings to classify different types of waste effectively. The primary goal is to facilitate better waste segregation, leading to improved recycling and disposal processes.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Evaluation and Results](#evaluation-and-results)
7. [Conclusion](#conclusion)
8. [Dependencies](#dependencies)
9. [How to Run the Project](#how-to-run-the-project)
10. [Acknowledgments](#acknowledgments)

---

## Introduction
Waste management is a critical challenge in modern society. This project utilizes machine learning to analyze waste sensor data and classify waste types into categories such as organic, recyclable, and non-recyclable. This automated system aims to streamline waste segregation at the source, benefiting both environmental and operational efficiency.

---

## Dataset
The dataset used in this project includes sensor data collected from waste management systems. Key attributes:
- **Sensor ID:** Unique identifier for sensors.
- **Inductive Property:** Measures metallic properties of the waste.
- **Capacitive Property:** Indicates dielectric characteristics.
- **Moisture Property:** Determines the moisture content.
- **Infrared Property:** Identifies waste surface composition.
- **Waste Type:** Target variable categorizing waste into types.

### Data Preprocessing
1. **Handling Missing Values:** Replaced missing values with mean/mode for numerical and categorical columns.
2. **Outlier Removal:** Used the Interquartile Range (IQR) method to clip extreme values.
3. **Encoding:** Applied Label Encoding for categorical data.
4. **Scaling:** StandardScaler and MinMaxScaler were used to normalize numerical features.

---

## Exploratory Data Analysis (EDA)
### Key Insights
1. **Distribution of Waste Types:** Visualized with bar plots.
2. **Correlation Analysis:** Heatmap of numerical features to understand relationships.
3. **Temporal Trends:** Time-series analysis of waste types over days.
4. **Feature Relationships:** Pairwise plots and scatterplots reveal patterns.

#### Example Visualizations
- **Bar Plot:** Distribution of waste types.
- **Heatmap:** Correlation between features.
- **Box Plot:** Variation of moisture properties across waste types.

---

## Feature Engineering
1. **Interaction Features:** Created new features such as `inductive_capacitive` (inductive x capacitive) and `moisture_infrared` (moisture x infrared).
2. **Dimensionality Reduction:** Principal Component Analysis (PCA) reduced feature space while retaining 95% variance.
3. **Oversampling:** Applied SMOTE to balance waste type distribution.

---

## Model Development
### Models Used
1. Logistic Regression
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. Naive Bayes
6. Random Forest
7. XGBoost

### Training and Testing
- Split dataset into 80% training and 20% testing.
- Used stratified sampling to ensure balanced class distribution.

### Model Optimization
- **Random Forest:** Tuned hyperparameters such as `n_estimators` and `max_depth`.
- **XGBoost:** Optimized learning rate, depth, and regularization.

---

## Evaluation and Results
### Metrics
- **Accuracy:** Percentage of correct predictions.
- **Classification Report:** Precision, recall, and F1-score.
- **Confusion Matrix:** Detailed error analysis.

### Key Results
- **Random Forest:** Accuracy: 93.5%.
- **XGBoost:** Accuracy: 94.2%.
- **Top Features:** Moisture property, capacitive property, and inductive property significantly contributed to predictions.

---

## Conclusion
The AI-powered waste classification system demonstrated high accuracy in segregating waste types. This project highlights the potential of machine learning in addressing real-world environmental challenges. Future enhancements could include:
- Integrating real-time data streams.
- Expanding the dataset with additional waste categories.

---

## Dependencies
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn`

---

## How to Run the Project
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/AtharvaDomale/Waste_Management_Project.git
   cd Waste_Management_Project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3.Load the dataset: Ensure waste_sensor_data.csv is in the working directory.

4.Run the Jupyter notebook or Python script:
   ```bash
   jupyter notebook waste_management.ipynb