# Diabetes Prediction Using Machine Learning

## Project Overview

This project applies supervised machine learning classification techniques to predict whether a patient has diabetes based on medical attributes. It uses the Pima Indians Diabetes Dataset and compares multiple classification algorithms to identify a suitable predictive model.

The project covers data exploration, preprocessing, model training, performance evaluation, ROC curve visualization, and model serialization using Joblib.

**Note:** This is an educational machine learning project, not a clinical diagnostic tool.

## Dataset

**Dataset:** Pima Indians Diabetes Dataset

**Source:** [Kaggle: Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

The dataset contains medical and demographic information used to predict diabetes status.

### Features

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of pregnancies        |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure     |
| SkinThickness            | Triceps skinfold thickness   |
| Insulin                  | Two-hour serum insulin       |
| BMI                      | Body mass index              |
| DiabetesPedigreeFunction | Diabetes pedigree function   |
| Age                      | Age in years                 |
| Outcome                  | Target variable: 0 or 1      |

The target variable `Outcome` indicates the recorded diabetes status in the dataset.

## Project Objectives

* Explore and understand the dataset using Pandas.
* Identify missing values and potential data-quality issues.
* Prepare features and target variables for machine learning.
* Split the dataset into training and testing sets.
* Train and compare at least two classification models.
* Evaluate model performance using multiple metrics.
* Plot ROC curves and compare ROC-AUC scores.
* Select a suitable model and save it using Joblib.

## Methodology

### 1. Data Loading and Exploration

Load the dataset using Pandas and perform exploratory analysis.

Tasks include:

* Display the first few rows using `head()`.
* Check the dataset dimensions using `shape`.
* Inspect column names and data types.
* Check missing values.
* Generate descriptive statistics using `describe()`.
* Examine the distribution of the target variable.
* Identify potentially invalid zero values in medical measurements.

### 2. Data Preprocessing

Prepare the data for model training.

Steps include:

* Separate the independent variables (`X`) from the target (`y`).
* Investigate missing and potentially invalid values.
* Apply appropriate preprocessing where necessary.
* Split the dataset into training and testing sets, using an 80/20 split.
* Apply feature scaling for models that benefit from it, such as Logistic Regression, SVM, and KNN.

Preprocessing steps should be fitted on the training data only to avoid data leakage. A Scikit-learn pipeline can help ensure consistent preprocessing during training and prediction.

### 3. Model Training

Train and compare at least two classification algorithms.

Possible models include:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* Decision Tree
* K-Nearest Neighbors (KNN)

Each model is trained on the training set and evaluated on the same held-out test set for a fair comparison.

### 4. Model Evaluation

Evaluate each model using the following metrics:

| Metric    | Purpose                                                                             |
| --------- | ----------------------------------------------------------------------------------- |
| Accuracy  | Proportion of correct predictions                                                   |
| Precision | Proportion of predicted positive cases that are actually positive                   |
| Recall    | Proportion of actual positive cases correctly identified                            |
| F1 Score  | Harmonic mean of precision and recall                                               |
| ROC-AUC   | Measures how well the model distinguishes between the two classes across thresholds |

Additional outputs include:

* Classification reports
* Confusion matrices
* ROC curves
* A comparison chart of model evaluation scores

ROC curves should use predicted probabilities or decision scores rather than binary predictions.

### 5. ROC Curve and Model Comparison

Plot ROC curves for the trained models on the test set and compare their ROC-AUC scores.

The comparison helps assess the trade-off between the true positive rate and false positive rate at different classification thresholds.

### 6. Best Model Selection and Saving

Compare the evaluation results and select a suitable model based on the project's objectives.

Save the selected model using Joblib:

```python
import joblib

joblib.dump(best_model, "diabetes_model.pkl")
```

If preprocessing is required, save the complete fitted pipeline rather than the classifier alone. This ensures that new input data receives the same preprocessing as the training data.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Google Colab / Jupyter Notebook

## Project Structure

```text
diabetes-prediction-ml/
│
├── Diabetes_Prediction.ipynb
├── diabetes_model.pkl
└── README.md
```

The saved model file should be included only if appropriate for the repository. If the file is large or generated during notebook execution, it may be better to provide instructions for regenerating it.

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Download the dataset from Kaggle and upload it to the notebook environment, or configure access to the dataset.
3. Install any required libraries that are not already available.
4. Run the notebook cells in order.
5. Review the evaluation metrics and visualizations.
6. Save the selected model using Joblib.

## Results and Findings

Complete this section after running the notebook.

Include:

* The classification models evaluated.
* Accuracy, precision, recall, F1 score, and ROC-AUC for each model.
* ROC curve comparison.
* The selected model and the reason for choosing it.
* Any observations about class imbalance, preprocessing, or model performance.

Do not report performance values until they have been obtained from the actual experiment.

## Conclusion

This project demonstrates a complete machine learning classification workflow, from data exploration and preprocessing to model evaluation and saving a trained model. Comparing multiple algorithms provides insight into their relative strengths and helps identify a model suited to the project's evaluation criteria.

The results should be interpreted in an educational context. This dataset and model are not sufficient for real-world medical diagnosis.

## Author

Nadiya Nowshin
