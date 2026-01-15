#### Part 1: Binary Classification

The objective of this experiment is to classify applicants into two categories—loan approved and loan not approved—based on a set of applicant attributes. The input to the model consists of multiple independent variables representing financial and personal features of the applicants, while the output is a binary dependent variable indicating whether the loan is approved or not.

**Step 1:** Import required libraries: pandas, matplotlib, Seaborn.

**Step 2:** Load 'Loan_dataset.csv'. The Loan Approval Prediction dataset contains 4,269 rows and 13 columns. It includes applicant details such as income, loan amount, credit history, and employment status, with the target column indicating whether the loan is approved or not.

**Step 3:** Perform data analysis using head(), tail(), info(), describe(), and isnull().sum().

**Step 4:** Define Features of X and Y:
- **Features X:** loan_id, no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value
- **Y:** loan_status

**Step 5:** Split the Dataset into 80% Training Set and 20% Testing Sets.

**Step 6:** Train the Decision Tree Classifier:
In this step, the Decision Tree Classifier is initialized and configured by setting important parameters such as the splitting criteria and the maximum depth (max_depth) of the tree. After defining these parameters, the classifier is trained using the training dataset, allowing it to learn decision rules from the data and build a tree structure for accurate prediction.

**Step 7:** Make Predictions on Test Data:
Using the trained Decision Tree model, predictions are made on the test dataset by providing its feature values as input. These predicted outcomes are then compared with the actual values to evaluate the performance of the model.

**Step 8:** Evaluate Model Performance:
The Decision Tree model is evaluated using the accuracy score to measure overall correctness, the confusion matrix to display true and false predictions, and the classification report to summarize precision, recall, and F1-score for each class. These metrics together help assess the model's performance.

**Step 9:** Plot:
To visualize the model's performance, a confusion matrix heatmap is plotted.

---

#### Part 2: Multi-class Classification

The objective of this experiment is to classify individuals into different obesity categories, such as normal weight, overweight, or various obesity levels, based on a set of personal and lifestyle attributes. The input to the model consists of multiple independent variables, including demographic information (e.g., age, gender, height, weight), dietary habits, physical activity, and other behavioral factors, while the output is a categorical dependent variable indicating the obesity level of each individual.

**Step 1:** Import required libraries: pandas, matplotlib, seaborn.

**Step 2:** Load 'Obesity_Dataset.csv'. The obesity dataset contains 2,111 records and 17 columns, representing individuals with various lifestyle, dietary, and physical attributes. It is used to classify obesity levels, where multiple personal and behavioral features act as inputs and the obesity category is the target variable.

**Step 3:** Define Feature X and Y:
- **Features X (Independent Variables):** Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
- **Target Variable Y:** NObeyesdad (represents the obesity category/class of an individual)

**Step 4:** Split the Dataset into 80% Training Set and 20% Testing Sets.

**Step 5:** Train the Decision Tree Classifier:
In this step, the Decision Tree Classifier is initialized and configured by setting important parameters such as the splitting criteria and the maximum depth (max_depth) of the tree. After defining these parameters, the classifier is trained using the training dataset, allowing it to learn decision rules from the data and build a tree structure for accurate prediction.

**Step 6:** Make Predictions on Test Data:
Using the trained Decision Tree model, predictions are made on the test dataset by providing its feature values as input. These predicted outcomes are then compared with the actual values to evaluate the performance of the model.

**Step 7:** Evaluate Model Performance:
The Decision Tree model is evaluated using the accuracy score to measure overall correctness, the confusion matrix to display true and false predictions, and the classification report to summarize precision, recall, and F1-score for each class. These metrics together help assess the model's performance.

**Step 8:** Plot:
To visualize the model's performance, a confusion matrix heatmap is plotted.