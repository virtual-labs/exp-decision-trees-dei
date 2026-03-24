## Part 1: Binary Classification

The objective of this part is to classify loan applications into two categories, approved and rejected, using a Decision Tree classifier. The model learns feature-based decision rules from applicant information such as income, loan amount, employment status, credit score, and asset values, and its predictive performance is evaluated using standard classification measures.

**Step 1:** Import the required libraries such as NumPy, Pandas, Matplotlib, Seaborn, and the relevant modules from `sklearn`.

**Step 2:** Load the `Loan_dataset.csv` dataset. The dataset contains 4,269 records and 13 columns, including applicant details and the binary target variable `loan_status`.

**Step 3:** Perform exploratory data analysis using `head()`, `info()`, `describe()`, and `isnull().sum()` to understand the structure, data types, and quality of the dataset.

**Step 4:** Check the class distribution of the target variable using `value_counts()` to understand the balance between approved and rejected loan applications.

**Step 5:** Define the feature matrix `X` and target variable `y`.

- **Features (X):** `loan_id`, `no_of_dependents`, `education`, `self_employed`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`
- **Target (y):** `loan_status`

**Step 6:** Encode categorical variables using an appropriate encoding method such as label encoding or one-hot encoding so that the data can be used by the Decision Tree model.

**Step 7:** Split the dataset into 80% training data and 20% testing data using `train_test_split()` with a fixed `random_state` for reproducibility.

**Step 8:** Initialize a `DecisionTreeClassifier` by setting suitable hyperparameters such as the splitting criterion (`gini` or `entropy`) and `max_depth`.

**Step 9:** Train the Decision Tree classifier using the training dataset.

**Step 10:** Generate predictions on the test dataset using the trained model.

**Step 11:** Evaluate the model using Accuracy, Precision, Recall, and F1-Score.

**Step 12:** Display the classification report to summarize per-class precision, recall, F1-score, and support.

**Step 13:** Compute the confusion matrix to compare predicted and actual outcomes.

**Step 14:** Plot a confusion matrix heatmap to visualize correct and incorrect classifications clearly.

**Step 15:** Interpret the learned decision rules and discuss how the selected features influence the binary classification outcome.

---

## Part 2: Multi-Class Classification

The objective of this part is to classify individuals into different obesity categories using a Decision Tree classifier. The input features include demographic, dietary, and lifestyle attributes, while the target variable represents the obesity level of each individual. This part demonstrates how Decision Trees handle multi-class classification problems and how their predictive performance can be assessed.

**Step 1:** Import the required libraries such as NumPy, Pandas, Matplotlib, Seaborn, and the necessary `sklearn` modules.

**Step 2:** Load the `Obesity_Dataset.csv` dataset. The dataset contains 2,111 records and 17 columns, including multiple personal, behavioral, and physical attributes along with the obesity category label.

**Step 3:** Perform basic exploratory data analysis using `head()`, `info()`, `describe()`, and `isnull().sum()`.

**Step 4:** Check the class distribution of the target variable using `value_counts()` to understand how the obesity categories are represented in the dataset.

**Step 5:** Define the feature matrix `X` and target variable `y`.

- **Features (X):** `Gender`, `Age`, `Height`, `Weight`, `family_history_with_overweight`, `FAVC`, `FCVC`, `NCP`, `CAEC`, `SMOKE`, `CH2O`, `SCC`, `FAF`, `TUE`, `CALC`, `MTRANS`
- **Target (y):** `NObeyesdad`

**Step 6:** Encode categorical variables appropriately so that the dataset can be processed by the Decision Tree algorithm.

**Step 7:** Split the dataset into 80% training data and 20% testing data using `train_test_split()` with stratification and a fixed `random_state` for reproducibility.

**Step 8:** Initialize the `DecisionTreeClassifier` by selecting the splitting criterion and other relevant parameters such as `max_depth`, `min_samples_split`, or `min_samples_leaf` if required.

**Step 9:** Train the Decision Tree model using the training dataset.

**Step 10:** Predict the obesity classes for the testing dataset.

**Step 11:** Evaluate the model using Accuracy, Precision (macro), Recall (macro), and F1-Score (macro).

**Step 12:** Generate the classification report to examine the model performance for each obesity category.

**Step 13:** Compute the confusion matrix to analyze correct and incorrect classifications across all classes.

**Step 14:** Plot a confusion matrix heatmap for a visual interpretation of the multi-class results.

**Step 15:** Discuss how Decision Tree depth, splitting criteria, and data characteristics influence multi-class classification performance.
