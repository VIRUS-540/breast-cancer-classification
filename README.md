# 🩺 Breast Cancer Classification Using Machine Learning

## 📌 Project Overview

Breast cancer is one of the most common forms of cancer, and accurate diagnosis is important for ensuring that patients receive the appropriate treatment. However, diagnostic errors can occur, including cases where a malignant tumor is incorrectly classified as benign or a benign tumor is incorrectly classified as malignant.

This project develops a **machine learning classification system** that predicts whether a breast tumor is **benign or malignant** based on measurements obtained from breast tissue samples.

The primary goal of the project is to build a model capable of accurately distinguishing between the two classes while paying particular attention to **false negative predictions**, where a malignant tumor is incorrectly classified as benign.

The final model was deployed as an interactive **Streamlit web application**, allowing users to enter tumor measurements and receive a predicted classification and probability.

---

# 🎯 Problem Statement

The problem is a **binary classification problem**.

Given a set of numerical measurements describing a breast tumor, the model must determine whether the tumor is:

* **0 → Benign**
* **1 → Malignant**

The project places particular importance on minimizing **false negatives**, because incorrectly classifying a malignant tumor as benign could potentially be more serious than incorrectly classifying a benign tumor as malignant.

The machine learning workflow therefore focuses not only on accuracy but also on metrics such as:

* Precision
* Recall
* F1-score
* Confusion matrix
* ROC curve
* AUC

---

# 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset**, containing measurements computed from digitized images of breast mass cell nuclei.

The dataset contains:

* **569 observations**
* **30 numerical features**
* **1 target variable**

The target variable is `diagnosis`, containing two classes:

* `B` — Benign
* `M` — Malignant

### Class Distribution

| Diagnosis | Count | Percentage |
| --------- | ----: | ---------: |
| Benign    |   357 |     62.74% |
| Malignant |   212 |     37.26% |

The dataset therefore contains more benign observations than malignant observations, although both classes have sufficient representation for binary classification.

---

# 🔎 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed before model training to understand the structure and characteristics of the dataset.

The analysis focused on:

* Dataset dimensions
* Data types
* Summary statistics
* Missing values
* Feature distributions
* Class distribution
* Feature relationships
* Correlations between variables
* Potential outliers

## Dataset Structure

The dataset initially contained **569 rows and 33 columns**.

Of these columns:

* 30 were numerical predictive features
* 1 was the diagnosis target
* Additional non-predictive columns were removed during preprocessing

## Missing Values

The dataset was inspected for null and missing values to determine whether imputation or other missing-value treatment was required.

## Class Distribution

A frequency plot was used to visualize the distribution of benign and malignant tumors.

The analysis showed that benign tumors represented the majority class.

## Correlation Analysis

A correlation heatmap was generated to examine relationships between numerical features.

The analysis showed strong relationships among several measurements, particularly between features describing related physical characteristics such as:

* Radius
* Perimeter
* Area

This is expected because these measurements describe related properties of the tumor.

## Outliers

Several variables contained observations with relatively large values, particularly features such as:

* `area_mean`
* `area_se`
* `perimeter_worst`
* `area_worst`

These observations were examined during EDA rather than automatically removed, since extreme measurements may represent legitimate characteristics of malignant tumors.

---

# 🧹 Data Preprocessing

After completing the initial exploratory analysis, the dataset was prepared for machine learning.

## Feature and Target Separation

The target variable, `diagnosis`, was separated from the predictor variables.

The feature matrix was represented as:

```python
X = df.drop(columns='diagnosis')
```

The target variable was represented as:

```python
y = df['diagnosis']
```

## Encoding the Target

The categorical target values were converted into numerical values using `LabelEncoder`.

This transformed the classes into:

```text
Benign → 0
Malignant → 1
```

## Train-Test Split

The dataset was divided into training and testing sets.

* Training set: **455 observations**
* Testing set: **114 observations**

The split allowed the models to be trained using one portion of the dataset and evaluated on previously unseen data.

## Feature Scaling

`StandardScaler` was used to standardize the numerical features for models that benefit from feature scaling.

The scaler was fitted on the training data and then used to transform both the training and testing data.

This prevents information from the test set from influencing the scaling process.

---

# 🤖 Machine Learning Models

Several classification algorithms were explored and evaluated.

The models included:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest

The purpose of testing multiple algorithms was to determine which model performed best on the classification task.

---

# 📈 Logistic Regression

Logistic Regression was used as one of the baseline classification models.

The model achieved approximately:

**Accuracy: 96.49%**

The confusion matrix was:

```text
[[71, 1],
 [ 3, 39]]
```

This means:

* 71 benign tumors were correctly classified
* 1 benign tumor was classified as malignant
* 3 malignant tumors were classified as benign
* 39 malignant tumors were correctly classified

The model achieved a malignant-class recall of approximately **92.86%**.

Cross-validation was also performed, with a mean score of approximately **97.14%**.

---

# 📍 K-Nearest Neighbors

A K-Nearest Neighbors classifier was also evaluated.

The initial model achieved:

**Accuracy: 95.61%**

The confusion matrix was:

```text
[[71, 1],
 [ 4, 38]]
```

GridSearchCV was later used to identify the optimal number of neighbors.

The best value obtained was:

```text
n_neighbors = 3
```

Although KNN performed well, it did not outperform the final Random Forest model.

---

# 🌳 Decision Tree

A Decision Tree classifier was also trained and evaluated.

The model achieved approximately:

**Accuracy: 93.86%**

The confusion matrix was:

```text
[[69, 3],
 [ 4, 38]]
```

The Decision Tree performed reasonably well but produced more classification errors than Logistic Regression and Random Forest.

---

# 🌲 Random Forest

Random Forest was selected as the primary model because it performed strongly on the classification task.

The initial Random Forest achieved approximately:

**Accuracy: 96.49%**

with the confusion matrix:

```text
[[72, 0],
 [ 4, 38]]
```

The model produced **zero false positives** on the test set.

---

# 🔧 Hyperparameter Tuning

GridSearchCV was used to find better Random Forest hyperparameters.

The parameter grid included:

```python
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

Five-fold cross-validation was used.

Recall was selected as the scoring metric because correctly identifying malignant tumors was an important objective of the project.

The best parameters obtained were:

```text
n_estimators = 50
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
```

The final tuned model was:

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
```

---

# 🏆 Final Model Performance

The tuned Random Forest achieved:

**Accuracy: 97.37%**

The confusion matrix was:

```text
[[72, 0],
 [ 3, 39]]
```

This produced:

| Metric    | Benign | Malignant |
| --------- | -----: | --------: |
| Precision |   0.96 |      1.00 |
| Recall    |   1.00 |      0.93 |
| F1-score  |   0.98 |      0.96 |

Overall:

* **Accuracy:** 97%
* **Macro F1-score:** 0.97
* **Malignant recall:** 93%
* **Malignant precision:** 100%

The tuned Random Forest achieved a higher accuracy than the Logistic Regression model while achieving the same malignant recall.

---

# 🧮 Confusion Matrix Interpretation

The final confusion matrix was:

```text
[[72, 0],
 [ 3, 39]]
```

It can be interpreted as:

### True Negatives — 72

72 benign tumors were correctly classified as benign.

### False Positives — 0

No benign tumors were incorrectly classified as malignant.

### False Negatives — 3

3 malignant tumors were incorrectly classified as benign.

### True Positives — 39

39 malignant tumors were correctly classified as malignant.

The model therefore successfully identified **39 out of 42 malignant tumors** in the test set.

However, the presence of 3 false negatives highlights an important limitation of the model and reinforces why recall is an important metric for this problem.

---

# ⭐ Feature Importance

Feature importance was extracted from the trained Random Forest model to determine which features contributed most to the model's decisions.

The ten most important features were:

| Rank | Feature                | Importance |
| ---: | ---------------------- | ---------: |
|    1 | `concave_points_worst` |   0.109992 |
|    2 | `concave_points_mean`  |   0.104194 |
|    3 | `area_worst`           |   0.102906 |
|    4 | `radius_worst`         |   0.099437 |
|    5 | `perimeter_mean`       |   0.083051 |
|    6 | `perimeter_worst`      |   0.077482 |
|    7 | `radius_mean`          |   0.062782 |
|    8 | `area_mean`            |   0.048824 |
|    9 | `concavity_mean`       |   0.048072 |
|   10 | `concavity_worst`      |   0.046118 |

The results show that features associated with **concavity, radius, perimeter, and area**, particularly the "worst" measurements, contributed substantially to the model's predictions.

The most important feature was:

```text
concave_points_worst
```

with an importance of approximately **0.110**.

A horizontal bar chart was generated to visualize the top feature importances.

---

# 📉 ROC Curve and AUC

A Receiver Operating Characteristic (ROC) curve was generated to examine how the model's **True Positive Rate (TPR)** and **False Positive Rate (FPR)** changed as the classification threshold changed.

The ROC curve demonstrates the trade-off between:

* **TPR / Recall:** proportion of malignant tumors correctly identified
* **FPR:** proportion of benign tumors incorrectly classified as malignant

As the classification threshold decreases, the model becomes more willing to classify observations as malignant. Consequently, both TPR and FPR generally increase.

The ROC curve showed that the model maintained a high true positive rate while keeping the false positive rate relatively low across much of the threshold range.

The calculated **ROC-AUC was approximately 0.994**.

An AUC close to 1 indicates that the model has a very strong ability to distinguish between benign and malignant tumors.

An important interpretation of the AUC is that the model generally assigns a higher predicted probability to malignant tumors than to benign tumors.

---

# 💾 Model Persistence

After training and tuning, the final Random Forest model was saved using `joblib`.

```python
joblib.dump(final_model, 'breast_cancer_model.pkl')
```

The saved model was later loaded by the Streamlit application:

```python
model = joblib.load('breast_cancer_model.pkl')
```

This allowed the deployed application to use the trained model without retraining it every time the application starts.

---

# 🌐 Web Application

The final model was integrated into an interactive Streamlit application.

Users can enter the 30 tumor measurements through the web interface.

The application then:

1. Collects the measurements.
2. Creates a pandas DataFrame.
3. Passes the input to the trained Random Forest model.
4. Generates a prediction.
5. Generates class probabilities.
6. Displays either **Benign** or **Malignant**.
7. Displays the corresponding prediction probability.
8. Stores the prediction in a SQLite database.

🚀Live Demo: https://breast-cancer-classification-atp4bkgbbtcswbovuauxtv.streamlit.app/

<img width="1580" height="1530" alt="image" src="https://github.com/user-attachments/assets/2eaec5e5-b812-4557-9118-78ac98697068" />
Fig 1: Web App Overview

<img width="1468" height="1312" alt="image" src="https://github.com/user-attachments/assets/84a2100a-41af-4da5-bc9b-38f2b07c148d" />
Fig 2: Malignant Prediction

<img width="1546" height="1354" alt="Screenshot 2026-09-01 135627" src="https://github.com/user-attachments/assets/5f8b28d0-c17f-4b83-b842-0eae507bafd2" />
Fig 3: Benign Prediction


---

# 🗄️ SQLite Database

SQLite was integrated into the application to store prediction records.

Each prediction can contain:

* Timestamp
* 30 tumor measurements
* Prediction
* Prediction probability

The database table is named:

```text
predictions
```

The database allows the application to maintain a record of predictions rather than simply displaying the result and discarding it.

The stored records can also be queried using pandas and later exported as CSV data.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* K-Nearest Neighbors
* Decision Tree
* Random Forest
* GridSearchCV
* StandardScaler
* LabelEncoder

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Database

* SQLite

---

# 📁 Project Structure

```text
breast_cancer_app/
│
├── app.py
├── breast_cancer_model.pkl
├── requirements.txt
├── prediction.db
└── README.md
```

---

# 🚀 Running the Application Locally

## 1. Clone the repository

```bash
git clone <repository-url>
cd breast_cancer_app
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit server.

---

# ⚠️ Limitations

Although the model achieved strong performance, this project should **not be used as a medical diagnostic tool**.

The model was developed as a machine learning project for educational and demonstration purposes.

Some important limitations include:

* The dataset is relatively small.
* The model was evaluated using a single train-test split.
* Three malignant tumors were classified as benign in the test set.
* The model's performance on this dataset does not guarantee equivalent performance on new populations or clinical environments.
* Feature importance from Random Forest does not necessarily imply causal relationships.
* A high ROC-AUC does not eliminate the possibility of clinically important classification errors.
* Real-world medical diagnosis requires qualified healthcare professionals and appropriate clinical testing.

---

# 📌 Conclusion

This project demonstrates an end-to-end machine learning workflow for binary classification of breast tumors.

The workflow included:

```text
Data Collection
      ↓
Exploratory Data Analysis
      ↓
Data Preprocessing
      ↓
Feature/Target Separation
      ↓
Target Encoding
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Feature Importance Analysis
      ↓
ROC/AUC Analysis
      ↓
Model Persistence
      ↓
Streamlit Deployment
      ↓
SQLite Prediction Storage
```

Among the evaluated models, the tuned **Random Forest classifier** produced the strongest overall performance, achieving approximately **97.37% accuracy** and a **0.994 ROC-AUC** on the evaluated test set.

The project demonstrates how a machine learning model can be taken beyond experimentation in a notebook and transformed into an interactive application capable of accepting real-time inputs, producing predictions, and storing prediction records.

---

## 👨‍💻 Author

**Mide**

Computer Science Student | Aspiring Data Scientist

---

## ⚠️ Disclaimer

This project is intended strictly for **educational and demonstration purposes**. It is not a substitute for professional medical advice, diagnosis, or treatment.
