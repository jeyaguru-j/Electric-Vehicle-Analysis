# EV Range Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green)](https://scikit-learn.org/)  
[![Pandas](https://img.shields.io/badge/Pandas-data-blueviolet)](https://pandas.pydata.org/)  
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)](https://matplotlib.org/)  
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-lightblue)](https://seaborn.pydata.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 

---

## Table of Contents
- [About the Project](#about-the-project)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [Workflow](#workflow)
- [Results](#results)
- [Future Scope](#future-scope)
- [Author](#author)
- [License](#license)

---

## About the Project

This project predicts the driving range of electric vehicles (EVs) using machine learning models trained on battery, vehicle performance, charging, and environmental data. It includes data preprocessing, model training (Random Forest, Gradient Boosting, etc.), evaluation, and prediction modules in a modular pipeline. Key factors influencing EV range, such as battery capacity and health, are analyzed to optimize predictive accuracy. Designed for production-readiness with clear documentation and future extensions for AI chatbot integration.

---

## Tech Stack

- **Language:** Python 3.8+
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** scikit-learn (Random Forest, Gradient Boosting)
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Version Control:** Git & GitHub

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/ev-range-prediction.git
   cd ev-range-prediction


2. **Create Virtual Environment & Install Dependencies**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

   pip install -r requirements.txt


3. **Run the Project**
   ```bash
   python main.py   

---

## Workflow

1. **Data Loading and Preprocessing**
   - Load EV dataset containing 3000 records with vehicle, battery, and environmental data.
   - Check data quality and conduct exploratory data analysis.
   - Prepare features and target variable for modeling.
2. **Model Training and Evaluation**
   - Train multiple regression models including Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
   - Evaluate models using R², RMSE, and MAE metrics.
   - Select the best model (Random Forest) and save it.
3. **Prediction**
   - Load the best model for inference.
   - Predict EV driving range based on custom input features.

---

## Results

- Achieved R² ~ 0.87 using Random Forest for range prediction.
- Root Mean Squared Error (RMSE) of approximately 48 km.
- Feature importance highlights battery capacity and health as most influential.

---

## Future Scope

- Implement hyperparameter tuning and cross-validation.
- Integrate a generative AI chatbot for EV customer queries.
- Deploy the prediction model as a web API or mobile app.
- Extend dataset with real-time sensor inputs for dynamic predictions.

---

## Author

Jeyaguru J

B.Tech Artificial Intelligence and Data Science

Email: jeyaguru1507@gmail.com

Internship Project – Edunet Foundation, in collaboration with AICTE & Shell

---

## License

This project is licensed under the MIT License
