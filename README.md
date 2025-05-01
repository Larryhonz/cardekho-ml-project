# cardekho-ml-project
Predicting car prices using ML models

# 🚗 Car Price Prediction — Machine Learning Project

This project aims to predict the selling price of used cars using machine learning techniques on the CarDekho dataset. It demonstrates a full ML workflow — from data preprocessing and exploratory analysis to model training, optimization, and evaluation.

---

## 📊 Dataset

- **Source:** CarDekho (Kaggle)
- **Records:** ~8,000 used car listings
- **Features include:**
  - Year of manufacture
  - Kilometers driven
  - Fuel type
  - Transmission type
  - Seller type
  - Ownership history
  - Selling price (target variable)

---

## 🧹 Data Preprocessing

- Dropped non-informative columns (e.g., car name)
- Created new feature: `car_age` from year of manufacture
- Log-transformed `km_driven` to reduce skew
- One-hot encoded categorical variables (`fuel`, `seller_type`, `transmission`, `owner`)
- Handled feature scaling where necessary

---

## 📈 Modeling Approach

- **Baseline Model:** Linear Regression — R² ≈ 0.40
- **Improved Model:** Random Forest Regressor
  - Used `GridSearchCV` for hyperparameter tuning
  - Applied `train_test_split` and 5-fold cross-validation
- **Evaluation Metrics:**
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
- Final model performance showed a significant improvement over the baseline.

---

## 🔧 Tools & Libraries

- Python 3.10+
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- Jupyter Notebook (via VS Code)
- Git & GitHub for version control

---

## 📁 Folder Structure

