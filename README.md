# Customer Churn Prediction using Neural Networks

## Overview
This project focuses on predicting customer churn using a Neural Network model. The goal was to identify patterns that indicate whether a customer is likely to stop using a service. The dataset contains customer demographic, service usage, and payment information.

---

## Key Features

### 1. **Data Preprocessing**
- **Handled Missing Values**:
  - Removed rows with empty `TotalCharges` values after identifying these entries.
  - Converted the `TotalCharges` column to a numeric type for analysis.
- **Encoding Categorical Variables**:
  - Converted binary columns (e.g., `Churn`, `Partner`, `Dependents`) into numeric values (`Yes` = 1, `No` = 0).
  - Applied one-hot encoding for multi-class categorical columns like `InternetService`, `Contract`, and `PaymentMethod`.
- **Normalized Numerical Data**:
  - Standardized numerical features to bring them to the same scale, improving model convergence.

### 2. **Exploratory Data Analysis (EDA)**
- Visualized the distribution of key features such as `tenure` and `MonthlyCharges`.
- Created histograms to compare `Churn` vs. non-`Churn` for insights into customer behavior.

### 3. **Neural Network Model**
- **Model Architecture**:
  - Input Layer: Matches the number of features in the dataset.
  - Two Hidden Layers: ReLU activation for non-linear feature interactions.
  - Output Layer: Single neuron with a sigmoid activation for binary classification.
- **Model Compilation**:
  - Loss Function: Binary Cross-Entropy.
  - Optimizer: Adam for adaptive learning rates.
  - Metric: Accuracy to evaluate model performance.
- **Training**:
  - Trained the model on 80% of the dataset over 100 epochs.

### 4. **Evaluation**
- Evaluated the model on the test dataset (20% of the total data).
- Achieved an accuracy of **80%**.

---

## Dataset
- **Source**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download
- **Size**: 27 columns and over 3000 customer records.
- **Features**:
  - `gender`, `Partner`, `Dependents`, `PhoneService` (Demographics).
  - `InternetService`, `OnlineSecurity`, `StreamingTV` (Service usage).
  - `MonthlyCharges`, `TotalCharges` (Payment details).
  - `Churn` (Target variable).

---

## Key Results
| Metric             | Value        |
|--------------------|--------------|
| Neural Network Accuracy | **80%** |

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Neural Networks: TensorFlow/Keras

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <https://github.com/SanthoshBotcha/Customer-Churn-Prediction-using-ANN.git>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook churn_prediction_nn.ipynb
   ```

---

## Next Steps
- Explore hyperparameter tuning for the Neural Network.
- Compare with other models like XGBoost to identify the most effective approach.
- Experiment with additional feature engineering techniques to improve performance.

---

## Author
Santhosh Botcha - https://www.linkedin.com/in/santhosh-botcha/
