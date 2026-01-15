// ==========================================
// TRAINING STEPS DATA WITH ALL VISUALIZATIONS
// ==========================================

const STEPS_MULTICLASS = [
    {
        title: "Importing Libraries",
        blocks: [{
            code: `# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
print("Libraries imported successfully!")`,
            output: `<div class="output-success">Libraries imported successfully!</div>`
        }]
    },
    {
        title: "Loading Data",
        blocks: [
            {
                code: `# Load the dataset
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
print("Dataset loaded successfully")`,
                output: `<div class="output-text">Dataset loaded successfully</div>`
            },
            {
                code: `# Check the shape of the dataset
print("Data Shape:")
print(data.shape)`,
                output: `<div class="output-text">Data Shape:<br>(2111, 17)</div>`
            },
            {
                code: `# Display first 5 rows of the dataset
print("First 5 rows of the dataset:")
data.head()`,
                output: `<div class="output-text">First 5 rows of the dataset:</div>
<div class="table-wrapper">
<table class="data-table">
  <thead>
    <tr>
      <th></th><th>Gender</th><th>Age</th><th>Height</th><th>Weight</th><th>family_history_with_overweight</th><th>FAVC</th><th>FCVC</th><th>NCP</th><th>CAEC</th><th>SMOKE</th><th>CH2O</th><th>SCC</th><th>FAF</th><th>TUE</th><th>CALC</th><th>MTRANS</th><th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr><th>0</th><td>Female</td><td>21.0</td><td>1.62</td><td>64.0</td><td>yes</td><td>no</td><td>2.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.0</td><td>no</td><td>0.0</td><td>1.0</td><td>no</td><td>Public_Transportation</td><td>Normal_Weight</td></tr>
    <tr><th>1</th><td>Female</td><td>21.0</td><td>1.52</td><td>56.0</td><td>yes</td><td>no</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>yes</td><td>3.0</td><td>yes</td><td>3.0</td><td>0.0</td><td>Sometimes</td><td>Public_Transportation</td><td>Normal_Weight</td></tr>
    <tr><th>2</th><td>Male</td><td>23.0</td><td>1.80</td><td>77.0</td><td>yes</td><td>no</td><td>2.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.0</td><td>no</td><td>2.0</td><td>1.0</td><td>Frequently</td><td>Public_Transportation</td><td>Normal_Weight</td></tr>
    <tr><th>3</th><td>Male</td><td>27.0</td><td>1.80</td><td>87.0</td><td>no</td><td>no</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.0</td><td>no</td><td>2.0</td><td>0.0</td><td>Frequently</td><td>Walking</td><td>Overweight_Level_I</td></tr>
    <tr><th>4</th><td>Male</td><td>22.0</td><td>1.78</td><td>89.8</td><td>no</td><td>no</td><td>2.0</td><td>1.0</td><td>Sometimes</td><td>no</td><td>2.0</td><td>no</td><td>0.0</td><td>0.0</td><td>Sometimes</td><td>Public_Transportation</td><td>Overweight_Level_II</td></tr>
  </tbody>
</table>
</div>`
            },
            {
                code: `# Display last 5 rows of the dataset
print("Last 5 rows of the dataset:")
data.tail()`,
                output: `<div class="output-text">Last 5 rows of the dataset:</div>
<div class="table-wrapper">
<table class="data-table">
  <thead>
    <tr>
      <th></th><th>Gender</th><th>Age</th><th>Height</th><th>Weight</th><th>family_history_with_overweight</th><th>FAVC</th><th>FCVC</th><th>NCP</th><th>CAEC</th><th>SMOKE</th><th>CH2O</th><th>SCC</th><th>FAF</th><th>TUE</th><th>CALC</th><th>MTRANS</th><th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr><th>2106</th><td>Female</td><td>20.976842</td><td>1.710730</td><td>131.408528</td><td>yes</td><td>yes</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>1.728139</td><td>no</td><td>1.676269</td><td>0.906247</td><td>Sometimes</td><td>Public_Transportation</td><td>Obesity_Type_III</td></tr>
    <tr><th>2107</th><td>Female</td><td>21.982942</td><td>1.748584</td><td>133.742943</td><td>yes</td><td>yes</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.005130</td><td>no</td><td>1.341390</td><td>0.599270</td><td>Sometimes</td><td>Public_Transportation</td><td>Obesity_Type_III</td></tr>
    <tr><th>2108</th><td>Female</td><td>22.524036</td><td>1.752206</td><td>133.689352</td><td>yes</td><td>yes</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.054193</td><td>no</td><td>1.414209</td><td>0.646288</td><td>Sometimes</td><td>Public_Transportation</td><td>Obesity_Type_III</td></tr>
    <tr><th>2109</th><td>Female</td><td>24.361936</td><td>1.739450</td><td>133.346641</td><td>yes</td><td>yes</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.852339</td><td>no</td><td>1.139107</td><td>0.586035</td><td>Sometimes</td><td>Public_Transportation</td><td>Obesity_Type_III</td></tr>
    <tr><th>2110</th><td>Female</td><td>23.664709</td><td>1.738836</td><td>133.472641</td><td>yes</td><td>yes</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.863513</td><td>no</td><td>1.026452</td><td>0.714137</td><td>Sometimes</td><td>Public_Transportation</td><td>Obesity_Type_III</td></tr>
  </tbody>
</table>
</div>`
            }
        ]
    },
    {
        title: "Preprocessing",
        blocks: [
            {
                code: `print("Statistical summary of the dataset:")
data.describe()`,
                output: `<div class="output-header">Statistical summary of the dataset:</div>
<div class="table-wrapper">
<table border="1" class="dataframe data-table">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
      <td>2111.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>24.312600</td>
      <td>1.701677</td>
      <td>86.586058</td>
      <td>2.419043</td>
      <td>2.685628</td>
      <td>2.008011</td>
      <td>1.010298</td>
      <td>0.657866</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.345968</td>
      <td>0.093305</td>
      <td>26.191172</td>
      <td>0.533927</td>
      <td>0.778039</td>
      <td>0.612953</td>
      <td>0.850592</td>
      <td>0.608927</td>
    </tr>
    <tr>
      <th>min</th>
      <td>14.000000</td>
      <td>1.450000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19.947192</td>
      <td>1.630000</td>
      <td>65.473343</td>
      <td>2.000000</td>
      <td>2.658738</td>
      <td>1.584812</td>
      <td>0.124505</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.777890</td>
      <td>1.700499</td>
      <td>83.000000</td>
      <td>2.385502</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.625350</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.000000</td>
      <td>1.768464</td>
      <td>107.430682</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.477420</td>
      <td>1.666678</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>61.000000</td>
      <td>1.980000</td>
      <td>173.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>`
            },
            {
                code: `# Initialize LabelEncoder
le = LabelEncoder()
target_names = []
print("LabelEncoder initialized")`,
                output: `<div class="output-text">LabelEncoder initialized</div>`
            },
            {
                code: `# Apply encoding to categorical columns
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
        if col == 'NObeyesdad':
            target_names = list(le.classes_)
print("Categorical variables encoded")`,
                output: `<div class="output-success">Categorical variables encoded</div>`
            },
            {
                code: `# Split data into training and testing sets
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete")`,
                output: `<div class="output-success">Data split complete</div>`
            }
        ]
    },
    {
        title: "Training (Gini)",
        blocks: [
            {
                code: `# Initialize and train Decision Tree with Gini impurity
model = DecisionTreeClassifier(criterion='gini')
model.fit(X_train, y_train)
print("Model trained using Gini impurity")`,
                output: `<div class="output-success">Model trained using Gini impurity</div>`
            },
            {
                code: `# Evaluate Gini model
y_pred = model.predict(X_test)
print("Accuracy (Gini, Full Tree):", accuracy_score(y_test, y_pred))
print("\\nConfusion Matrix (Gini, Full Tree):\\n", confusion_matrix(y_test, y_pred))`,
                output: `<div class="output-text">Accuracy (Gini, Full Tree): 0.9433</div><br><div class="output-text">Confusion Matrix (Gini, Full Tree):<br>[[52  3  0  1  0  0  0]<br> [ 4 56  0  0  0  0  0]<br> ...<br> [ 0  0  0  2 60  1  0]]</div>`
            },
            {
                code: `# Compute confusion matrix for Gini
print("Calculating confusion matrix (Gini)...")
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix calculated successfully")`,
                output: `<div class="output-text">Calculating confusion matrix (Gini)...<br>Confusion matrix calculated successfully</div>`
            },
            {
                 code: `# Plot confusion matrix (Gini)
print("Displaying plot...")
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (Gini)')
plt.show()`,
                 output: `<img src="./images/confusion_matrix_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            },
            {
                 code: `# Visualize the Gini Decision Tree
print("Plotting the Decision Tree (Gini)...")
plt.figure(figsize=(20,10))
plot_tree(model, feature_names=X.columns, class_names=target_names, filled=True, rounded=True)
plt.show()`,
                 output: `<img src="./images/full_tree_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            }
        ]
    },
    {
        title: "Training (Entropy)",
        blocks: [
            {
                code: `# Train and evaluate Decision Tree with Entropy
model_entropy = DecisionTreeClassifier(criterion='entropy')
model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)
print("Accuracy (Entropy, Full Tree):", accuracy_score(y_test, y_pred_entropy))`,
                output: `<div class="output-text">Accuracy (Entropy, Full Tree): 0.9598</div>`
            },
            {
                code: `# Compute confusion matrix for Entropy
print("Calculating confusion matrix (Entropy)...")
cm_entropy = confusion_matrix(y_test, y_pred_entropy)
print("Confusion matrix (Entropy) calculated successfully")`,
                output: `<div class="output-text">Calculating confusion matrix (Entropy)...<br>Confusion matrix (Entropy) calculated successfully</div>`
            },
            {
                 code: `# Plot confusion matrix (Entropy)
print("Displaying plot...")
plt.figure(figsize=(8,6))
sns.heatmap(cm_entropy, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.show()`,
                 output: `<img src="./images/confusion_matrix_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            },
            {
                 code: `# Visualize the Entropy Decision Tree
print("Plotting the Decision Tree (Entropy)...")
plot_tree(model_entropy, feature_names=X.columns, class_names=target_names, filled=True, rounded=True)
plt.show()`,
                 output: `<img src="./images/full_tree_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            }
        ]

    },
    {
        title: "Model Evaluation (Depth)",
        blocks: [
            {
                code: `# Analyze Decision Tree with Max Depth = 2
d = 2
clf = DecisionTreeClassifier(max_depth=d, criterion='gini', random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Max Depth: {d}, Accuracy: {acc:.4f}")
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=target_names, filled=True, rounded=True)
plt.title(f"Decision Tree (Max Depth = {d})")
plt.show()`,
                output: `Max Depth: 2, Accuracy: 0.5432<br><br><img src="./images/decision_tree_depth_2_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
             {
                code: `# Analyze Decision Tree with Max Depth = 4
d = 4
clf = DecisionTreeClassifier(max_depth=d, criterion='gini', random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Max Depth: {d}, Accuracy: {acc:.4f}")
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=target_names, filled=True, rounded=True)
plt.title(f"Decision Tree (Max Depth = {d})")
plt.show()`,
                output: `Max Depth: 4, Accuracy: 0.9433<br><br><img src="./images/decision_tree_depth_4_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                 code: `# Analyze Decision Tree with Max Depth = 2 (Entropy)
d = 2
clf = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Max Depth (Entropy): {d}, Accuracy: {acc:.4f}")
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=target_names, filled=True, rounded=True)
plt.title(f"Decision Tree (Max Depth = {d}, Entropy)")
plt.show()`,
                 output: `Max Depth (Entropy): 2, Accuracy: 0.9456<br><br><img src="./images/decision_tree_depth_2_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                 code: `# Analyze Decision Tree with Max Depth = 4 (Entropy)
d = 4
clf = DecisionTreeClassifier(max_depth=d, criterion='entropy', random_state=42)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Max Depth (Entropy): {d}, Accuracy: {acc:.4f}")
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=target_names, filled=True, rounded=True)
plt.title(f"Decision Tree (Max Depth = {d}, Entropy)")
plt.show()`,
                 output: `Max Depth (Entropy): 4, Accuracy: 0.9598<br><br><img src="./images/decision_tree_depth_4_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            }
        ]
    }
];

const STEPS_BINARY = [
    {
        title: "Importing Libraries",
        blocks: [{
            code: `# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
print("Libraries imported successfully!")`,
            output: `<div class="output-success">Libraries imported successfully!</div>`
        }]
    },
    {
        title: "Initial Data Loading & Exploration",
        blocks: [
            {
                code: `# Load the dataset
data = pd.read_csv("loan_approval_dataset.csv")
print("Dataset loaded successfully")`,
                output: `<div class="output-text">Dataset loaded successfully</div>`
            },
            {
                code: `# Check the shape of the dataset
print("Dataset Shape:", data.shape)`,
                output: `<div class="output-text">Dataset Shape: (4269, 13)</div>`
            },
            {
                code: `# Fix column spacing
data.columns = data.columns.str.strip()

# Print columns in vertical form
print("\\nColumns (Vertical List):")
for col in data.columns:
    print(col)`,
                output: `<div class="output-text"><br>Columns (Vertical List):<br>loan_id<br>no_of_dependents<br>education<br>self_employed<br>income_annum<br>loan_amount<br>loan_term<br>cibil_score<br>residential_assets_value<br>commercial_assets_value<br>luxury_assets_value<br>bank_asset_value<br>loan_status</div>`
            },
            {
                code: `# Display first 5 rows
print("First 5 rows of the dataset:")
data.head()`,
                output: `<div class="output-text">First 5 rows of the dataset:</div><div class="table-wrapper"><table class="data-table"><thead><tr style="text-align: right;"><th></th><th>loan_id</th><th>no_of_dependents</th><th>education</th><th>self_employed</th><th>income_annum</th><th>loan_amount</th><th>loan_term</th><th>cibil_score</th><th>residential_assets_value</th><th>commercial_assets_value</th><th>luxury_assets_value</th><th>bank_asset_value</th><th>loan_status</th></tr></thead><tbody><tr><th>0</th><td>1</td><td>2</td><td>Graduate</td><td>No</td><td>9600000</td><td>29900000</td><td>12</td><td>778</td><td>2400000</td><td>17600000</td><td>22700000</td><td>8000000</td><td>Approved</td></tr><tr><th>1</th><td>2</td><td>0</td><td>Not Graduate</td><td>Yes</td><td>4100000</td><td>12200000</td><td>8</td><td>417</td><td>2700000</td><td>2200000</td><td>8800000</td><td>3300000</td><td>Rejected</td></tr><tr><th>2</th><td>3</td><td>3</td><td>Graduate</td><td>No</td><td>9100000</td><td>29700000</td><td>20</td><td>506</td><td>7100000</td><td>4500000</td><td>33300000</td><td>12800000</td><td>Rejected</td></tr><tr><th>3</th><td>4</td><td>3</td><td>Graduate</td><td>No</td><td>8200000</td><td>30700000</td><td>8</td><td>467</td><td>18200000</td><td>3300000</td><td>23300000</td><td>7900000</td><td>Rejected</td></tr><tr><th>4</th><td>5</td><td>5</td><td>Not Graduate</td><td>Yes</td><td>9800000</td><td>24200000</td><td>20</td><td>382</td><td>12400000</td><td>8200000</td><td>29400000</td><td>5000000</td><td>Rejected</td></tr></tbody></table></div>`
            },
            {
                code: `# Display last 5 rows
print("Last 5 rows of the dataset:")
data.tail()`,
                output: `<div class="output-text">Last 5 rows of the dataset:</div><div class="table-wrapper"><table class="data-table"><thead><tr style="text-align: right;"><th></th><th>loan_id</th><th>no_of_dependents</th><th>education</th><th>self_employed</th><th>income_annum</th><th>loan_amount</th><th>loan_term</th><th>cibil_score</th><th>residential_assets_value</th><th>commercial_assets_value</th><th>luxury_assets_value</th><th>bank_asset_value</th><th>loan_status</th></tr></thead><tbody><tr><th>4264</th><td>4265</td><td>5</td><td>Graduate</td><td>Yes</td><td>1000000</td><td>2300000</td><td>12</td><td>317</td><td>2800000</td><td>500000</td><td>3300000</td><td>800000</td><td>Rejected</td></tr><tr><th>4265</th><td>4266</td><td>0</td><td>Not Graduate</td><td>Yes</td><td>3300000</td><td>11300000</td><td>20</td><td>559</td><td>4200000</td><td>2900000</td><td>11000000</td><td>1900000</td><td>Approved</td></tr><tr><th>4266</th><td>4267</td><td>2</td><td>Not Graduate</td><td>No</td><td>6500000</td><td>23900000</td><td>18</td><td>457</td><td>1200000</td><td>12400000</td><td>18100000</td><td>7300000</td><td>Rejected</td></tr><tr><th>4267</th><td>4268</td><td>1</td><td>Not Graduate</td><td>No</td><td>4100000</td><td>12800000</td><td>8</td><td>780</td><td>8200000</td><td>700000</td><td>14100000</td><td>5800000</td><td>Approved</td></tr><tr><th>4268</th><td>4269</td><td>1</td><td>Graduate</td><td>No</td><td>9200000</td><td>29700000</td><td>10</td><td>607</td><td>17800000</td><td>11800000</td><td>35700000</td><td>12000000</td><td>Approved</td></tr></tbody></table></div>`
            }
        ]
    },
    {
        title: "Data Preprocessing & Information",
        blocks: [
            {
                code: `# Target Column Encoding
data.columns = data.columns.str.strip()
data['loan_status'] = data['loan_status'].str.strip().map({
    'Approved': 1,
    'Rejected': 0
})

print("Target column loan_status encoded successfully")
data.head()`,
                output: `<div class="output-text">Target column loan_status encoded successfully</div><div class="table-wrapper"><table class="data-table"><thead><tr style="text-align: right;"><th></th><th>loan_id</th><th>no_of_dependents</th><th>education</th><th>self_employed</th><th>income_annum</th><th>loan_amount</th><th>loan_term</th><th>cibil_score</th><th>residential_assets_value</th><th>commercial_assets_value</th><th>luxury_assets_value</th><th>bank_asset_value</th><th>loan_status</th></tr></thead><tbody><tr><th>0</th><td>1</td><td>2</td><td>Graduate</td><td>No</td><td>9600000</td><td>29900000</td><td>12</td><td>778</td><td>2400000</td><td>17600000</td><td>22700000</td><td>8000000</td><td>1</td></tr><tr><th>1</th><td>2</td><td>0</td><td>Not Graduate</td><td>Yes</td><td>4100000</td><td>12200000</td><td>8</td><td>417</td><td>2700000</td><td>2200000</td><td>8800000</td><td>3300000</td><td>0</td></tr><tr><th>2</th><td>3</td><td>3</td><td>Graduate</td><td>No</td><td>9100000</td><td>29700000</td><td>20</td><td>506</td><td>7100000</td><td>4500000</td><td>33300000</td><td>12800000</td><td>0</td></tr><tr><th>3</th><td>4</td><td>3</td><td>Graduate</td><td>No</td><td>8200000</td><td>30700000</td><td>8</td><td>467</td><td>18200000</td><td>3300000</td><td>23300000</td><td>7900000</td><td>0</td></tr><tr><th>4</th><td>5</td><td>5</td><td>Not Graduate</td><td>Yes</td><td>9800000</td><td>24200000</td><td>20</td><td>382</td><td>12400000</td><td>8200000</td><td>29400000</td><td>5000000</td><td>0</td></tr></tbody></table></div>`
            },
            {
                code: `#Data Information
print("All Required Information Related To Data: ")
data.info()`,
                output: `<div class="output-text">All Required Information Related To Data:<br>&lt;class 'pandas.core.frame.DataFrame'&gt;<br>RangeIndex: 4269 entries, 0 to 4268<br>Data columns (total 13 columns):<br> #   Column                    Non-Null Count  Dtype <br>---  ------                    --------------  ----- <br> 0   loan_id                   4269 non-null   int64 <br> 1   no_of_dependents          4269 non-null   int64 <br> 2   education                 4269 non-null   object<br> 3   self_employed             4269 non-null   object<br> 4   income_annum              4269 non-null   int64 <br> 5   loan_amount               4269 non-null   int64 <br> 6   loan_term                 4269 non-null   int64 <br> 7   cibil_score               4269 non-null   int64 <br> 8   residential_assets_value  4269 non-null   int64 <br> 9   commercial_assets_value   4269 non-null   int64 <br> 10  luxury_assets_value       4269 non-null   int64 <br> 11  bank_asset_value          4269 non-null   int64 <br> 12  loan_status               4269 non-null   int64 <br>dtypes: int64(11), object(2)<br>memory usage: 433.7+ KB</div>`
            },
            {
                code: `# Check for missing (null) values in each column
print("Missing values in each column:")
print(data.isnull().sum())`,
                output: `<div class="output-text">Missing values in each column:<br>loan_id                     0<br>no_of_dependents            0<br>education                   0<br>self_employed               0<br>income_annum                0<br>loan_amount                 0<br>loan_term                   0<br>cibil_score                 0<br>residential_assets_value    0<br>commercial_assets_value     0<br>luxury_assets_value         0<br>bank_asset_value            0<br>loan_status                 0<br>dtype: int64</div>`
            }
        ]
    },
    {
        title: "Feature Engineering & Data Splitting",
        blocks: [
            {
                code: `# Define target column
target_col = "loan_status"
print("Target column selected:", target_col)

# Separate features and target
X = data.drop(columns=[target_col])
y = data[target_col]
print("Features (X) and target (y) separated successfully")

# Identify categorical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns identified:", cat_cols)

# Apply One-Hot Encoding to categorical columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder='passthrough'
)

# Transform feature data
X_transformed = preprocessor.fit_transform(X)
print("Categorical features one-hot encoded and dataset transformed successfully")`,
                output: `<div class="output-success">Target column selected: loan_status<br>Features (X) and target (y) separated successfully<br>Categorical columns identified: ['education', 'self_employed']<br>Categorical features one-hot encoded and dataset transformed successfully</div>`
            },
            {
                code: `# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset successfully split into training and testing sets")
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)`,
                output: `<div class="output-text">Dataset successfully split into training and testing sets<br>Training set size: (3415, 14)<br>Testing set size: (854, 14)</div>`
            }
        ]
    },
    {
        title: "Model Training & Evaluation (Gini Criterion)",
        blocks: [
            {
                code: `# Initialize and train Decision Tree model using Gini impurity
dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree model trained successfully using Gini impurity")`,
                output: `<div class="output-success">Decision Tree model trained successfully using Gini impurity</div>`
            },
            {
                code: `# Evaluate Decision Tree model (Gini impurity)
y_pred = dt_model.predict(X_test)
print("Accuracy (Gini, Full Tree):", accuracy_score(y_test, y_pred))
print("\\nConfusion Matrix (Gini, Full Tree):")
print(confusion_matrix(y_test, y_pred))`,
                output: `<div class="output-text">Accuracy (Gini, Full Tree): 0.9789227166276346<br><br>Confusion Matrix (Gini, Full Tree):<br>[[313  10]<br> [  8 523]]</div>`
            },
            {
                code: `# Compute confusion matrix for Decision Tree model (Gini impurity)
print("Calculating confusion matrix (Gini)...")
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix calculated successfully")`,
                output: `<div class="output-text">Calculating confusion matrix (Gini)...<br>Confusion matrix calculated successfully</div>`
            },
            {
                code: `# Plot confusion matrix for Decision Tree model (Gini impurity)
print("Displaying confusion matrix plot (Gini)...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix (Gini)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
print("Confusion matrix plot displayed successfully")`,
                output: `<div class="output-text">Displaying confusion matrix plot (Gini)...</div><img src="./images/binary_confusion_matrix_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd;"><div class="output-text">Confusion matrix plot displayed successfully</div>`
            },
            {
                code: `# Visualize the Gini Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, rounded=True, label='root', impurity=False, proportion=False)
plt.title("Decision Tree (Gini)")
plt.show()`,
                output: `<img src="./images/binary_full_tree_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            }
        ]
    },
    {
        title: "Model Training & Evaluation (Entropy Criterion)",
        blocks: [
            {
                code: `# Train and evaluate Decision Tree model using Entropy
dt_model_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model_entropy.fit(X_train, y_train)
print("Decision Tree model trained successfully using Entropy")

# Predict on test data
y_pred_entropy = dt_model_entropy.predict(X_test)
print("Accuracy (Entropy, Full Tree):", accuracy_score(y_test, y_pred_entropy))
print("\\nConfusion Matrix (Entropy, Full Tree):")
print(confusion_matrix(y_test, y_pred_entropy))`,
                output: `<div class="output-success">Decision Tree model trained successfully using Entropy</div><div class="output-text">Accuracy (Entropy, Full Tree): 0.9789227166276346<br><br>Confusion Matrix (Entropy, Full Tree):<br>[[312  11]<br> [  7 524]]</div>`
            },
            {
                code: `# Compute confusion matrix for Decision Tree model (Entropy)
print("Calculating confusion matrix (Entropy)...")
cm_entropy = confusion_matrix(y_test, y_pred_entropy)
print("Confusion matrix (Entropy) calculated successfully")`,
                output: `<div class="output-text">Calculating confusion matrix (Entropy)...<br>Confusion matrix (Entropy) calculated successfully</div>`
            },
            {
                code: `# Plot confusion matrix for Decision Tree model (Entropy)
target_names = ['Rejected', 'Approved']
plt.figure(figsize=(8, 6))
sns.heatmap(cm_entropy, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (Entropy)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
print("Confusion matrix (Entropy) plot displayed successfully")`,
                output: `<div class="output-text">Displaying confusion matrix plot (Entropy)...</div><img src="./images/binary_confusion_matrix_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd;"><div class="output-text">Confusion matrix (Entropy) plot displayed successfully</div>`
            },
            {
                code: `# Visualize the Entropy Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model_entropy, filled=True, rounded=True, label='root', impurity=False, proportion=False)
plt.title("Decision Tree (Entropy)")
plt.show()`,
                output: `<img src="./images/binary_full_tree_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            }
        ]
    },
    {
        title: "Decision Tree Depth Analysis (Gini)",
        blocks: [
            {
                code: `# Train Decision Tree with max_depth=3 (Gini)
dt_gini_3 = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_gini_3.fit(X_train, y_train)
y_pred_3 = dt_gini_3.predict(X_test)
accuracy_3 = accuracy_score(y_test, y_pred_3)
print(f"Accuracy (Max Depth 3, Gini): {accuracy_3:.4f}")
print("Confusion Matrix (Max Depth 3):")
print(confusion_matrix(y_test, y_pred_3))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_3), annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix Heatmap (Max Depth 3, Gini)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()`,
                output: `<div class="output-text">Accuracy (Max Depth 3, Gini): 0.9403<br>Confusion Matrix (Max Depth 3):<br>[[280  43]<br> [  8 523]]</div><img src="./images/binary_confusion_matrix_gini_dep3.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Plot the tree (Gini - Max Depth 3)
plt.figure(figsize=(20, 10))
plot_tree(dt_gini_3, filled=True, rounded=True, label='root', impurity=False, proportion=False)
plt.title("Decision Tree (Gini) - Max Depth 3")
plt.show()`,
                output: `<img src="./images/binary_tree_depth3_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Train Decision Tree with max_depth=5 (Gini)
dt_gini_5 = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
dt_gini_5.fit(X_train, y_train)
y_pred_5 = dt_gini_5.predict(X_test)
accuracy_5 = accuracy_score(y_test, y_pred_5)
print(f"Accuracy (Max Depth 5, Gini): {accuracy_5:.4f}")
print("Confusion Matrix (Max Depth 5):")
print(confusion_matrix(y_test, y_pred_5))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_5), annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix Heatmap (Max Depth 5, Gini)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()`,
                output: `<div class="output-text">Accuracy (Max Depth 5, Gini): 0.9707<br>Confusion Matrix (Max Depth 5):<br>[[308  15]<br> [ 10 521]]</div><img src="./images/binary_confusion_matrix_gini_dep5.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Plot the tree (Gini - Max Depth 5)
plt.figure(figsize=(20, 10))
plot_tree(dt_gini_5, filled=True, rounded=True, label='root', impurity=False, proportion=False)
plt.title("Decision Tree (Gini) - Max Depth 5")
plt.show()`,
                output: `<img src="./images/binary_tree_depth5_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Train Decision Tree with max_depth=7 (Gini)
dt_gini_7 = DecisionTreeClassifier(criterion='gini', max_depth=7, random_state=42)
dt_gini_7.fit(X_train, y_train)
y_pred_7 = dt_gini_7.predict(X_test)
accuracy_7 = accuracy_score(y_test, y_pred_7)
print(f"Accuracy (Max Depth 7, Gini): {accuracy_7:.4f}")
print("Confusion Matrix (Max Depth 7):")
print(confusion_matrix(y_test, y_pred_7))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_7), annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix Heatmap (Max Depth 7, Gini)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()`,
                output: `<div class="output-text">Accuracy (Max Depth 7, Gini): 0.9766<br>Confusion Matrix (Max Depth 7):<br>[[312  11]<br> [  9 522]]</div><img src="./images/binary_confusion_matrix_gini_dep7.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Plot the tree (Gini - Max Depth 7)
plt.figure(figsize=(20, 10))
plot_tree(dt_gini_7, filled=True, rounded=True, label='root', impurity=False, proportion=False)
plt.title("Decision Tree (Gini) - Max Depth 7")
plt.show()`,
                output: `<img src="./images/binary_tree_depth7_gini.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            }
        ]
    },
    {
        title: "Decision Tree Depth Analysis (Entropy)",
        blocks: [
            {
                code: `# Train Decision Tree with max_depth=3 (Entropy)
dt_entropy_3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
dt_entropy_3.fit(X_train, y_train)
y_pred_3 = dt_entropy_3.predict(X_test)
accuracy_3 = accuracy_score(y_test, y_pred_3)
print(f"Accuracy (Max Depth 3, Entropy): {accuracy_3:.4f}")
print("Confusion Matrix (Max Depth 3):")
print(confusion_matrix(y_test, y_pred_3))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_3), annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix Heatmap (Max Depth 3, Entropy)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()`,
                output: `<div class="output-text">Accuracy (Max Depth 3, Entropy): 0.9333<br>Confusion Matrix (Max Depth 3):<br>[[275  48]<br> [  9 522]]</div><img src="./images/binary_confusion_matrix_entropy_dep3.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Plot the tree (Entropy - Max Depth 3)
plt.figure(figsize=(20, 10))
plot_tree(dt_entropy_3, filled=True, rounded=True, label='root', impurity=False, proportion=False)
plt.title("Decision Tree (Entropy) - Max Depth 3")
plt.show()`,
                output: `<img src="./images/binary_tree_depth3_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Train Decision Tree with max_depth=5 (Entropy)
dt_entropy_5 = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
dt_entropy_5.fit(X_train, y_train)
y_pred_5 = dt_entropy_5.predict(X_test)
accuracy_5 = accuracy_score(y_test, y_pred_5)
print(f"Accuracy (Max Depth 5, Entropy): {accuracy_5:.4f}")
print("Confusion Matrix (Max Depth 5):")
print(confusion_matrix(y_test, y_pred_5))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_5), annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix Heatmap (Max Depth 5, Entropy)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()`,
                output: `<div class="output-text">Accuracy (Max Depth 5, Entropy): 0.9637<br>Confusion Matrix (Max Depth 5):<br>[[304  19]<br> [ 12 519]]</div><img src="./images/binary_confusion_matrix_entropy_dep5.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Plot the tree (Entropy - Max Depth 5)
plt.figure(figsize=(20, 10))
plot_tree(dt_entropy_5, filled=True, rounded=True, label='root', impurity=False, proportion=False)
plt.title("Decision Tree (Entropy) - Max Depth 5")
plt.show()`,
                output: `<img src="./images/binary_tree_depth5_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Train Decision Tree with max_depth=7 (Entropy)
dt_entropy_7 = DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=42)
dt_entropy_7.fit(X_train, y_train)
y_pred_7 = dt_entropy_7.predict(X_test)
accuracy_7 = accuracy_score(y_test, y_pred_7)
print(f"Accuracy (Max Depth 7, Entropy): {accuracy_7:.4f}")
print("Confusion Matrix (Max Depth 7):")
print(confusion_matrix(y_test, y_pred_7))

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_7), annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
plt.title("Confusion Matrix Heatmap (Max Depth 7, Entropy)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()`,
                output: `<div class="output-text">Accuracy (Max Depth 7, Entropy): 0.9766<br>Confusion Matrix (Max Depth 7):<br>[[312  11]<br> [  9 522]]</div><img src="./images/binary_confusion_matrix_entropy_dep7.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            },
            {
                code: `# Plot the tree (Entropy - Max Depth 7)
plt.figure(figsize=(20, 10))
plot_tree(dt_entropy_7, filled=True, rounded=True, label='root', impurity=False, proportion=False)
plt.title("Decision Tree (Entropy) - Max Depth 7")
plt.show()`,
                output: `<img src="./images/binary_tree_depth7_entropy.png" style="max-width:100%; height:auto; border: 1px solid #ddd;">`
            }
        ]
    }
];

// ==========================================
// FEATURE DESCRIPTIONS
// ==========================================

const FEATURE_DESCRIPTIONS = {
    "FAVC": "High Caloric Food",
    "FCVC": "Vegetable Consumption",
    "NCP": "Number of Main Meals",
    "CAEC": "Food Between Meals",
    "SMOKE": "Smokes",
    "CH2O": "Water Consumption",
    "SCC": "Calorie Monitoring",
    "FAF": "Physical Activity",
    "TUE": "Technology Use Time",
    "CALC": "Alcohol Consumption",
    "MTRANS": "Transportation",
    "Age": "Age",
    "Height": "Height",
    "Weight": "Weight",
    "Gender": "Gender",
    "family_history_with_overweight": "Family History",
    // Binary Classification Features
    "no_of_dependents": "Dependents",
    "education": "Education",
    "self_employed": "Self Employed",
    "income_annum": "Annual Income",
    "loan_amount": "Loan Amount",
    "loan_term": "Loan Term",
    "cibil_score": "CIBIL Score",
    "residential_assets_value": "Residential Assets",
    "commercial_assets_value": "Commercial Assets",
    "luxury_assets_value": "Luxury Assets",
    "bank_asset_value": "Bank Assets",
    "loan_status": "Loan Status"
};

// ==========================================
// GLOBAL STATE
// ==========================================

let TREE_DATA = null;
let STEPS = [];
let EXPERIMENT_STATE = {
    stepIndex: 0,
    subStepIndex: 0,
    stepsStatus: []
};
let CLASSIFICATION_TYPE = null; // 'binary' or 'multiclass'

// Animation State
let animState = {
    currentStep: -1,
    isAnimating: false,
    isComplete: false,
    camera: { x: 0, y: 0, scale: 1 },
    isDragging: false,
    lastMouse: { x: 0, y: 0 },
    speed: 1.0,
    animationTimeout: null,
    fitTimeout: null
};

let currentConfig = {
    depth: '4',
    criterion: 'gini',
    sampleIndex: 0
};

// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    // Classification selector is shown by default in HTML
    // DO NOT load tree data here - wait for user to select classification type
    // loadTreeData() will be called from enterAnimation() after selection
});

function selectClassificationType(type) {
    CLASSIFICATION_TYPE = type;
    const baseSteps = type === 'multiclass' ? STEPS_MULTICLASS : STEPS_BINARY;
    
    // Load only first 4 common steps + 3 placeholder steps (total 7)
    // Using EXACT standardized step names
    STEPS = [
        { ...baseSteps[0], title: "Importing Libraries" },        // 1. Importing Libraries
        { ...baseSteps[1], title: "Loading Dataset" },            // 2. Loading Dataset
        { ...baseSteps[2], title: "Data Analysis" },              // 3. Data Analysis
        { ...baseSteps[3], title: "Data Preprocessing" },         // 4. Data Preprocessing
        { title: "Model Training", blocks: [] },                  // 5. Model Training (placeholder)
        { title: "Model Evaluation", blocks: [] },                // 6. Model Evaluation (placeholder)
        { title: "Model Simulation", blocks: [] }                    // 7. Model Testing (placeholder)
    ];
    
    // Initialize State - only first 4 steps are real, rest are placeholders
    EXPERIMENT_STATE.stepsStatus = STEPS.map((_, i) => ({ 
        unlocked: i === 0, 
        completed: false, 
        partial: false 
    }));
    
    // Hide selector
    document.getElementById('classificationSelector').style.display = 'none';
    
    // Show training section with steps
    document.getElementById('trainingSection').style.display = 'flex';
    
    // Initialize DOM elements if not already done
    if (!stepsContainer) {
        stepsContainer = document.getElementById('stepsContainer');
        codeDisplay = document.getElementById('codeDisplay');
        outputDisplay = document.getElementById('outputDisplay');
        bottomPane = document.querySelector('.bottom-pane');
        runBtn = document.getElementById('runBtn');
    }
    
    // Initialize the sidebar and first step
    currentStepIndex = 0;
    EXPERIMENT_STATE.stepIndex = 0;
    EXPERIMENT_STATE.subStepIndex = 0;
    renderSidebar();
    loadStep(0);
}

// Global functions for HTML access
window.selectClassificationType = selectClassificationType;
window.selectCriterion = selectCriterion;
window.selectMaxDepth = selectMaxDepth;
window.runStep = runStep;
window.nextSubStep = nextSubStep;
window.restartExperiment = restartExperiment;
window.showCriterionSelector = showCriterionSelector;

// Show Criterion Selector (Called after Step 4 - Data Preprocessing completes)
function showCriterionSelector() {
    // Hide code/output panes and show criterion selector
    document.querySelector('.top-pane').style.display = 'none';
    document.querySelector('.bottom-pane').style.display = 'none';
    document.getElementById('criterionSelectorPane').style.display = 'flex';
}

// Criterion Selection - Replaces placeholder steps with actual training/depth/conclusion
function selectCriterion(criterion) {
    currentConfig.criterion = criterion;
    
    // Hide criterion selector
    document.getElementById('criterionSelectorPane').style.display = 'none';
    
    // Show code/output panes
    document.querySelector('.top-pane').style.display = '';
    document.querySelector('.bottom-pane').style.display = '';
    
    // Replace placeholder steps (indices 4,5,6) with actual steps based on criterion
    const baseSteps = CLASSIFICATION_TYPE === 'multiclass' ? STEPS_MULTICLASS : STEPS_BINARY;
    
    if (CLASSIFICATION_TYPE === 'binary') {
        // Binary: steps 0-3 common, 4=Gini training, 5=Entropy training
        if (criterion === 'gini') {
            STEPS[4] = { ...baseSteps[4], title: "Model Training" }; // Gini training
        } else {
            STEPS[4] = { ...baseSteps[5], title: "Model Training" }; // Entropy training
        }
        // Step 6 (Model Evaluation) and Step 7 (Model Testing) remain placeholders or get updated
        STEPS[5] = { title: "Model Evaluation", blocks: [], placeholder: true }; 
        
        const modelVar = criterion === 'gini' ? 'dt_model' : 'dt_model_entropy';
        STEPS[6] = { 
            title: "Model Simulation", 
            comment: "Test prediction on sample data",
            blocks: [{ 
                code: `# Predict on new sample data
sample_data = [[1, 2, 0, 1, 9600000, 29900000, 12, 778, 2400000, 17600000, 22700000, 8000000, 1, 0]]
prediction = ${modelVar}.predict(sample_data)

if prediction[0] == 1:
    print("Prediction: Loan Approved")
else:
    print("Prediction: Loan Rejected")`, 
                output: '<div class="output-text">Prediction: Loan Approved</div>' 
            }] 
        };
    } else {
        // Multiclass: similar logic
        STEPS[4] = { ...(criterion === 'gini' ? baseSteps[3] : baseSteps[4]), title: "Model Training" };
        STEPS[5] = { title: "Model Evaluation", blocks: [], placeholder: true };
        
        const modelVar = criterion === 'gini' ? 'model' : 'model_entropy';
        STEPS[6] = { 
            title: "Model Simulation", 
            comment: "Test prediction on sample data",
            blocks: [{ 
                code: `# Predict on new sample data (using sample from dataset)
# Gender, Age, Height, Weight, family_history, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
sample_data = [[1, 21.0, 1.62, 64.0, 1, 0, 2.0, 3.0, 2, 0, 2.0, 0, 0.0, 1.0, 0, 3]]
prediction = ${modelVar}.predict(sample_data)
print(f"Predicted Class: {prediction[0]}")`, 
                output: '<div class="output-text">Predicted Class: 1</div>' 
            }] 
        };
    }
    
    // Update state - steps 0-3 already completed, unlock step 4
    EXPERIMENT_STATE.stepsStatus[4].unlocked = true;
    EXPERIMENT_STATE.stepIndex = 4;
    EXPERIMENT_STATE.subStepIndex = 0;
    
    renderSidebar();
    loadStep(4); // Load the Model Training step
}

// Show Depth Selector (Called during Model Evaluation step)
function showDepthSelector() {
    // Hide code/output panes and show depth selector
    document.querySelector('.top-pane').style.display = 'none';
    document.querySelector('.bottom-pane').style.display = 'none';

    // Update Depth Options dynamically based on type
    const depthOptionsContainer = document.querySelector('.depth-options');
    if (depthOptionsContainer) {
        depthOptionsContainer.innerHTML = '';
        const options = CLASSIFICATION_TYPE === 'multiclass' 
            ? [{d: 2, t: 'Simple'}, {d: 4, t: 'Balanced'}]
            : [{d: 3, t: 'Simple'}, {d: 5, t: 'Balanced'}, {d: 7, t: 'Deeper'}];
        
        options.forEach(opt => {
            const div = document.createElement('div');
            div.className = 'depth-option';
            div.onclick = () => selectMaxDepth(opt.d);
            div.innerHTML = `<h3>Depth ${opt.d}</h3><p>${opt.t}</p>`;
            depthOptionsContainer.appendChild(div);
        });

        // Add Full Tree option only for Binary (for now, based on data)
        if (CLASSIFICATION_TYPE === 'binary') {
            const fullDiv = document.createElement('div');
            fullDiv.className = 'depth-option';
            fullDiv.onclick = () => selectMaxDepth(null);
            fullDiv.innerHTML = `<h3>Full Tree</h3><p>No limit</p>`;
            depthOptionsContainer.appendChild(fullDiv);
        }
    }

    document.getElementById('depthSelectorPane').style.display = 'flex';
}

// Max Depth Selection
function selectMaxDepth(depth) {
    // Hide depth selector immediately
    const selector = document.getElementById('depthSelectorPane');
    if (selector) selector.style.display = 'none';

    currentConfig.depth = depth ? depth.toString() : 'Full';
    
    // Hide depth selector
    document.getElementById('depthSelectorPane').style.display = 'none';
    
    // Show code/output panes
    document.querySelector('.top-pane').style.display = '';
    document.querySelector('.bottom-pane').style.display = '';
    
    // Populating Step 6 (Evaluation) with specific depth code
    const baseSteps = CLASSIFICATION_TYPE === 'multiclass' ? STEPS_MULTICLASS : STEPS_BINARY;
    const criterion = currentConfig.criterion;
    
    if (CLASSIFICATION_TYPE === 'binary') {
        const sourceStep = criterion === 'gini' ? baseSteps[6] : baseSteps[7];
        let selectedBlocks = [];
        
        if (depth === 3) selectedBlocks = [sourceStep.blocks[0], sourceStep.blocks[1]];
        else if (depth === 5) selectedBlocks = [sourceStep.blocks[2], sourceStep.blocks[3]];
        else if (depth === 7) selectedBlocks = [sourceStep.blocks[4], sourceStep.blocks[5]];
        else {
            // Full Tree fallback
            selectedBlocks = [sourceStep.blocks[4], sourceStep.blocks[5]];
        }
        
        STEPS[5] = { 
            title: `Model Evaluation (Depth ${depth || 'Full'})`, 
            blocks: selectedBlocks 
        };
    } else {
        // Multiclass
        const sourceStep = baseSteps[5];
        let blockIndex = 0;
        if (criterion === 'gini') {
            blockIndex = (depth === 2) ? 0 : 1;
        } else {
            blockIndex = (depth === 2) ? 2 : 3;
        }
        
        STEPS[5] = { 
            title: `Model Evaluation (Depth ${depth || 'Full'})`, 
            blocks: [sourceStep.blocks[blockIndex]] 
        };
    }

    // Unlock and Load Step 6
    EXPERIMENT_STATE.stepsStatus[5].unlocked = true;
    EXPERIMENT_STATE.stepsStatus[5].completed = false; // Reset if switching
    EXPERIMENT_STATE.stepIndex = 5;
    EXPERIMENT_STATE.subStepIndex = 0;
    
    renderSidebar();
    loadStep(5);
}

// Ensure showDepthSelector is also global
window.showDepthSelector = showDepthSelector;
window.showCriterionSelector = showCriterionSelector;

// DOM Elements
let stepsContainer, codeDisplay, outputDisplay, runBtn, bottomPane;

function init() {
    stepsContainer = document.getElementById('stepsContainer');
    codeDisplay = document.getElementById('codeDisplay');
    outputDisplay = document.getElementById('outputDisplay');
    bottomPane = document.querySelector('.bottom-pane');
    runBtn = document.getElementById('runBtn');

    renderSidebar();
    loadStep(0);
}

// Render Sidebar with Color Logic
function renderSidebar() {
    stepsContainer.innerHTML = '';

    STEPS.forEach((step, index) => {
        const status = EXPERIMENT_STATE.stepsStatus[index];
        const btn = document.createElement('button');
        btn.classList.add('step-btn');
        
        // Label Logic
        let label = `${index + 1}. ${step.title}`;
        if (status.completed) label = `✓ ${step.title}`;
        btn.innerText = label;

        // Styling & Interaction Logic
        if (status.unlocked) {
            if (status.completed) {
                // Completed -> Green (from CSS)
                btn.classList.add('completed');
            } else if (status.partial) {
                // In Progress -> Yellow/Orange (from CSS)
                btn.classList.add('in-progress');
            }
            btn.disabled = false;
            btn.style.cursor = 'pointer';

             // Active Styling
            if (index === EXPERIMENT_STATE.stepIndex) {
                btn.classList.add('active');
            }

            btn.onclick = () => loadStep(index);
        } else {
             // Locked -> Grey (from CSS)
            btn.classList.add('disabled');
            btn.disabled = true;
        }

        stepsContainer.appendChild(btn);
    });

    // Add Restart Button at the end
    const restartBtn = document.createElement('button');
    restartBtn.classList.add('step-btn');
    restartBtn.classList.add('restart-btn');
    restartBtn.innerText = "Restart Experiment";
    restartBtn.style.textAlign = 'center';
    restartBtn.style.marginTop = "auto";
    restartBtn.onclick = restartExperiment;
    stepsContainer.appendChild(restartBtn);
    
    // Add Download Button below Restart
    const downloadBtn = document.createElement('button');
    downloadBtn.classList.add('step-btn');
    downloadBtn.classList.add('download-btn');
    downloadBtn.style.textAlign = 'center';
    downloadBtn.style.marginTop = "10px";
    downloadBtn.style.marginBottom = "20px";
    downloadBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
            stroke-linejoin="round" style="margin-right:8px; vertical-align: middle;">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Download Experiment
    `;
    downloadBtn.onclick = downloadTrainingAsPDF;
    stepsContainer.appendChild(downloadBtn);
}

function loadStep(index) {
    EXPERIMENT_STATE.stepIndex = index;
    // IMPORTANT: Do NOT reset subStepIndex if the step is already completed.
    // If the user goes back to a completed step, show the FINAL state of that step?
    // OR, mimics LR: LR resets subStepIndex to 0. 
    // "progress will not be restored" -> This implies if I go back, I have to run it again? 
    // BUT the user said "when the step is completed it is completed".
    // Let's stick to LR behavior: STATE.subStepIndex = 0;
    EXPERIMENT_STATE.subStepIndex = 0;
    
    renderSidebar();
    updateUI();
}

function restartExperiment() {
    // Reset State
    EXPERIMENT_STATE.stepIndex = 0;
    EXPERIMENT_STATE.subStepIndex = 0;
    EXPERIMENT_STATE.stepsStatus = STEPS.map(() => ({ unlocked: false, completed: false, partial: false }));
    EXPERIMENT_STATE.stepsStatus[0].unlocked = true;

    // Clear Outputs & Restore DOM Structure
    bottomPane.classList.remove('active-output');
    bottomPane.style.display = '';
    // Restore structure in case it was overwritten by completion message
    bottomPane.innerHTML = '<div class="output-content" id="outputDisplay"><div class="placeholder-text">Click the Run button to execute...</div></div>';
    
    // Re-bind outputDisplay since we just recreated it
    outputDisplay = document.getElementById('outputDisplay');

    init();
}

function updateUI() {
    const step = STEPS[EXPERIMENT_STATE.stepIndex];
    if (!step || !step.blocks || step.blocks.length === 0) return;
    if (EXPERIMENT_STATE.subStepIndex >= step.blocks.length) EXPERIMENT_STATE.subStepIndex = 0;

    const block = step.blocks[EXPERIMENT_STATE.subStepIndex];
    if (!block) return;

    const codeHeaderBar = document.getElementById('codeHeaderBar');

    // Use comment field like RF, fallback to extracting from code
    let headerComment = "";
    let displayCode = block.code || "";
    
    if (block && block.comment) {
        // Preferred: Use comment field
        headerComment = block.comment;
    } else if (step.comment) {
        // Fallback: Use step-level comment
        headerComment = step.comment;
    } else if (block && block.code) {
        // Last resort: Extract from code (legacy support)
        const commentMatch = block.code.match(/#\s*([^<\n\r]*)/);
        if (commentMatch) {
            headerComment = commentMatch[1].trim();
            // Strip ALL comment lines from code display
            displayCode = block.code.split('\n').filter(line => !line.trim().startsWith('#')).join('\n').trim();
        }
    }

    if (codeHeaderBar) {
        if (headerComment) {
            codeHeaderBar.innerText = "# " + headerComment;
            codeHeaderBar.style.display = 'block';
        } else {
            codeHeaderBar.style.display = 'none';
        }
    }

    // Update Code
    if (displayCode) {
        codeDisplay.innerHTML = highlightCode(displayCode);
    }

    // Reset Output Pane
    bottomPane.classList.remove('active-output', 'completed-output');
    bottomPane.style.display = '';
    bottomPane.style.flexDirection = '';
    bottomPane.style.justifyContent = '';
    bottomPane.style.alignItems = '';
    
    // Ensure outputDisplay exists and is properly inside bottomPane
    let outputDisplay = document.getElementById('outputDisplay');
    if (!outputDisplay) {
         bottomPane.innerHTML = '<div class="output-content" id="outputDisplay"></div>';
         outputDisplay = document.getElementById('outputDisplay');
    }
    outputDisplay.innerHTML = '<div class="placeholder-text">Click the Run button to execute...</div>';

    // Reset Run Button
    runBtn.style.display = 'flex';
    runBtn.classList.remove('completed', 'arrow-mode'); // Remove any prior status classes
    runBtn.style.backgroundColor = '#F57C2A'; // Orange default
    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    runBtn.disabled = false;
    runBtn.onclick = runStep;
}

function runStep() {
    const step = STEPS[EXPERIMENT_STATE.stepIndex];
    const block = step.blocks[EXPERIMENT_STATE.subStepIndex];

    // 1. Loading State
    outputDisplay.innerHTML = '<div class="loading-spinner">Running code...</div>';
    runBtn.disabled = true;

    // 2. Simulated Delay
    setTimeout(() => {
        // 3. Show Output - Dynamically wrap tables for horizontal scroll
        let outputContent = block.output;
        if (outputContent.includes('<table')) {
            // Check if it's already wrapped to avoid double wrapping
            if (!outputContent.includes('table-wrapper')) {
                outputContent = `<div class="table-wrapper">${outputContent}</div>`;
            }
        }
        outputDisplay.innerHTML = outputContent;
        bottomPane.classList.add('active-output');

        // 4. Update Button State (Success)
        runBtn.classList.add('completed');
        runBtn.style.backgroundColor = '#A6CE63'; // Green
        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        bottomPane.classList.add('completed-output'); // Green border

        // Mark partial progress
        EXPERIMENT_STATE.stepsStatus[EXPERIMENT_STATE.stepIndex].partial = true;
        renderSidebar();

        // 5. Handle Next Logic
        const hasNextBlock = EXPERIMENT_STATE.subStepIndex < step.blocks.length - 1;

        if (hasNextBlock) {
             setTimeout(() => {
                runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                runBtn.classList.remove('completed');
                runBtn.classList.add('arrow-mode'); // Green arrow class can stay, but override color
                runBtn.style.backgroundColor = '#5FA8E4'; // Blue for Next
                runBtn.disabled = false;
                runBtn.onclick = nextSubStep; // Go to next block
             }, 500);
        } else {
            // Step Fully Completed
            EXPERIMENT_STATE.stepsStatus[EXPERIMENT_STATE.stepIndex].completed = true;
            renderSidebar();

            // Unlock next step logic
            if (EXPERIMENT_STATE.stepIndex < STEPS.length - 1) {
                EXPERIMENT_STATE.stepsStatus[EXPERIMENT_STATE.stepIndex + 1].unlocked = true;
                renderSidebar();
                
             setTimeout(() => {
                    // Step 4 (index 3) Special Logic for Branching - Show Arrow, wait for click
                    if (EXPERIMENT_STATE.stepIndex === 3) {
                        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                        runBtn.classList.remove('completed');
                        runBtn.classList.add('arrow-mode');
                        runBtn.style.backgroundColor = '#5FA8E4'; // Blue
                        runBtn.disabled = false;
                        runBtn.onclick = function() {
                            EXPERIMENT_STATE.stepsStatus[3].completed = true;
                            showCriterionSelector(); // Only on button click
                        };
                        return;
                    }

                    // Step 5 (index 4) Special Logic for Depth Selection
                    if (EXPERIMENT_STATE.stepIndex === 4) {
                        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                        runBtn.classList.remove('completed');
                        runBtn.classList.add('arrow-mode');
                        runBtn.style.backgroundColor = '#5FA8E4'; // Blue
                        runBtn.disabled = false;
                        runBtn.onclick = function() {
                            EXPERIMENT_STATE.stepsStatus[4].completed = true;
                            showDepthSelector(); // Only on button click
                        };
                        return;
                    }
                    
                    // Arrow for Next Step
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.classList.remove('completed');
                    runBtn.classList.add('arrow-mode'); 
                    runBtn.style.backgroundColor = '#5FA8E4'; // Blue for Next Step
                    runBtn.disabled = false;
                    runBtn.onclick = function() {
                        loadStep(EXPERIMENT_STATE.stepIndex + 1);
                    };
                 }, 500);
            } else {
                 // End of Experiment - Show Blue Arrow to trigger completion
                 setTimeout(() => {
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.classList.remove('completed');
                    runBtn.classList.add('arrow-mode'); 
                    runBtn.style.backgroundColor = '#5FA8E4'; // Blue for Finish
                    runBtn.disabled = false;
                    runBtn.onclick = showCompletionMessage;
                 }, 500);
            }
        }

    }, 800);
}

function nextSubStep() {
    EXPERIMENT_STATE.subStepIndex++;
    updateUI();
}

function showCompletionMessage() {
    const outputDisplay = document.getElementById('outputDisplay');
    
    // Set container styles for centering
    bottomPane.style.display = 'flex';
    bottomPane.style.flexDirection = 'column';
    bottomPane.style.justifyContent = 'center';
    bottomPane.style.alignItems = 'center';

    outputDisplay.innerHTML = `
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; gap: 20px; width: 100%; padding: 40px 0;">
            <div>
                <h2 style="color: #3d8b8b; font-family: 'Courier New', monospace; font-size: 2rem; font-weight: bold; margin-bottom: 10px;">
                    Experiment Completed! 
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6b5b95" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-left: 5px;">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                </h2>
                <p style="color: #555; font-size: 1.1rem; font-family: 'Courier New', monospace;">You have completed decision tree successfully!</p>
            </div>

            <button onclick="enterAnimation()" style="
                background: #5FA8E4;
                color: white;
                border: none;
                padding: 16px 32px;
                border-radius: 12px;
                font-weight: 700;
                font-size: 1rem;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 12px;
                transition: all 0.2s;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            " onmouseover="this.style.background='#4A90C7'; this.style.transform='translateY(-2px)'" 
               onmouseout="this.style.background='#5FA8E4'; this.style.transform='translateY(0)'">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Enter Interactive Animation
            </button>
        </div>
    `;

    runBtn.style.display = 'none';
}

// Helper
// Helper
function highlightCode(code) {
  if (!code) return '';
  
  // First, handle comments (lines starting with #)
  let lines = code.split('\n');
  let highlightedLines = lines.map(line => {
    let trimmed = line.trim();
    if (trimmed.startsWith('#')) {
      return `<span class="output-success">${line}</span>`; // Green comments
    }
    
    // Then handle keywords in non-comment lines
    return line
      .replace(/\bimport\b/g, '<span class="kw">import</span>')
      .replace(/\bfrom\b/g, '<span class="kw">from</span>')
      .replace(/\bas\b/g, '<span class="kw">as</span>')
      .replace(/\bprint\b/g, '<span class="func">print</span>')
      .replace(/\bdef\b/g, '<span class="kw">def</span>')
      .replace(/\breturn\b/g, '<span class="kw">return</span>');
  });

  return highlightedLines.join('\n');
}

function escapeHtml(text) {
    if (!text) return text;
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function downloadTrainingAsPDF() {
    // Redirect to the PDF file for download
    window.open('assets/EXP-6.pdf', '_blank');
}

// ==========================================
// ANIMATION SECTION
// ==========================================

async function loadTreeData() {
    try {
        // Force clear to prevent caching
        TREE_DATA = null;
        
        console.log("loadTreeData called with CLASSIFICATION_TYPE:", CLASSIFICATION_TYPE);
        
        if (CLASSIFICATION_TYPE === 'binary') {
            console.log("Loading TREE_DATA_BINARY...");
            console.log("TREE_DATA_BINARY exists?", typeof TREE_DATA_BINARY !== 'undefined');
            if (typeof TREE_DATA_BINARY !== 'undefined') {
                TREE_DATA = TREE_DATA_BINARY;
                console.log("✅ Binary data loaded. Inputs:", TREE_DATA.inputs.length);
                console.log("Sample features:", Object.keys(TREE_DATA.inputs[0].features));
            } else {
                console.error("❌ TREE_DATA_BINARY is undefined!");
            }
        } else {
            console.log("Loading TREE_DATA_MULTICLASS...");
            TREE_DATA = TREE_DATA_MULTICLASS;
            console.log("✅ Multiclass data loaded. Inputs:", TREE_DATA.inputs.length);
        }
        
        if (!TREE_DATA) {
            console.error("TREE_DATA is undefined!");
            return;
        }
        
        // Setup depth buttons dynamically
        setupDepthButtons();
        
        // Generate sample buttons
        generateSampleButtons();
        
        // Setup event listeners
        setupAnimationListeners();
        
        // IMPORTANT: Ensure currentConfig.depth is valid before rendering
        const availableDepths = Object.keys(TREE_DATA.models);
        if (!availableDepths.includes(currentConfig.depth)) {
            currentConfig.depth = availableDepths[0];
            console.log("Reset depth to:", currentConfig.depth);
        }
        
        // Initial render
        updateVisualization();
        
    } catch (error) {
        console.error("Failed to load tree data:", error);
    }
}

function generateSampleButtons() {
    const sampleButtons = document.getElementById('sampleButtons');
    if (!sampleButtons) return;
    
    sampleButtons.innerHTML = '';
    TREE_DATA.inputs.forEach((_, index) => {
        const btn = document.createElement('button');
        btn.className = `sample-btn ${index === 0 ? 'active' : ''}`;
        btn.textContent = index + 1;
        btn.onclick = () => selectSample(index);
        sampleButtons.appendChild(btn);
    });
}

function setupDepthButtons() {
    console.log("setupDepthButtons called");
    
    // Find depth container
    // Robustly find the Max Depth control section
    const sections = document.querySelectorAll('.control-section');
    console.log("Found control sections:", sections.length);
    
    let container = null;
    sections.forEach((sec, idx) => {
        const label = sec.querySelector('label');
        console.log(`Section ${idx} label:`, label ? label.innerText : 'no label');
        if (label && label.innerText.toUpperCase().includes('MAX DEPTH')) {
            console.log("Found Max Depth section:", sec);
            console.log("Section HTML:", sec.innerHTML);
            container = sec.querySelector('.button-group');
            console.log("Found Max Depth container:", container);
        }
    });
    
    if (!container) {
        console.error("Could not find Max Depth button container!");
        return;
    }
    
    container.innerHTML = '';
    if (!TREE_DATA || !TREE_DATA.models) {
        console.error("TREE_DATA or TREE_DATA.models is undefined");
        return;
    }

    const depths = Object.keys(TREE_DATA.models).sort((a,b) => parseInt(a) - parseInt(b));
    console.log("Available depths:", depths);
    
    // Set default depth if current is invalid
    if (!depths.includes(currentConfig.depth)) {
        currentConfig.depth = depths[0];
        console.log("Reset currentConfig.depth to:", currentConfig.depth);
    }

    depths.forEach((depth, idx) => {
        const btn = document.createElement('button');
        btn.className = `depth-btn ${depth === currentConfig.depth ? 'active' : ''}`;
        btn.dataset.depth = depth;
        btn.textContent = depth;
        btn.onclick = () => {
            document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentConfig.depth = depth;
            updateVisualization();
        };
        container.appendChild(btn);
        console.log(`Added depth button: ${depth}`);
    });
    
    console.log("setupDepthButtons complete. Buttons in container:", container.children.length);
}

function setupAnimationListeners() {
    // Depth buttons are now dynamic, moved to setupDepthButtons
    
    // Criterion buttons
    
    // Criterion buttons
    document.querySelectorAll('.criterion-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.criterion-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentConfig.criterion = btn.dataset.criterion;
            updateVisualization();
        });
    });
    
    // Speed slider
    const speedSlider = document.getElementById('speedSlider');
    if (speedSlider) {
        speedSlider.addEventListener('input', (e) => {
            animState.speed = parseFloat(e.target.value);
        });
    }
    
    // Replay button
    const replayBtn = document.getElementById('replayBtn');
    if (replayBtn) {
        replayBtn.addEventListener('click', updateVisualization);
    }
    
    // Back button
    const backBtn = document.getElementById('backBtn');
    if (backBtn) {
        backBtn.addEventListener('click', exitAnimation);
    }
    
    // Pan/Zoom controls
    const animationArea = document.getElementById('animationArea');
    if (animationArea) {
        animationArea.addEventListener('mousedown', handleMouseDown);
        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        animationArea.addEventListener('wheel', handleWheel, { passive: false });
    }
}

function selectSample(index) {
    currentConfig.sampleIndex = index;
    document.querySelectorAll('.sample-btn').forEach((btn, i) => {
        btn.classList.toggle('active', i === index);
    });
    updateVisualization();
}

function updateVisualization() {
    if (!TREE_DATA) {
        console.error("updateVisualization: TREE_DATA is null");
        return;
    }
    
    console.log("updateVisualization called");
    console.log("  currentConfig.depth:", currentConfig.depth);
    console.log("  Available depths:", Object.keys(TREE_DATA.models));
    console.log("  currentConfig.criterion:", currentConfig.criterion);
    console.log("  currentConfig.sampleIndex:", currentConfig.sampleIndex);
    
    // Stop any ongoing animation
    if (animState.animationTimeout) {
        clearTimeout(animState.animationTimeout);
    }
    if (animState.fitTimeout) {
        clearTimeout(animState.fitTimeout);
    }
    
    animState.isAnimating = false;
    animState.isComplete = false;
    animState.currentStep = -1;
    
    // Validate depth exists
    if (!TREE_DATA.models[currentConfig.depth]) {
        console.error(`Depth ${currentConfig.depth} not found in models. Available:`, Object.keys(TREE_DATA.models));
        return;
    }
    
    // Validate criterion exists
    if (!TREE_DATA.models[currentConfig.depth][currentConfig.criterion]) {
        console.error(`Criterion ${currentConfig.criterion} not found for depth ${currentConfig.depth}`);
        return;
    }
    
    const sample = TREE_DATA.inputs[currentConfig.sampleIndex];
    const model = TREE_DATA.models[currentConfig.depth][currentConfig.criterion];
    const path = sample.paths[currentConfig.depth][currentConfig.criterion];
    
    console.log("  Model nodes:", model.length);
    console.log("  Path length:", path.length);
    
    // Update sidebar
    renderFeatureSidebar(sample);
    
    // Update true label
    const trueLabelValue = document.getElementById('trueLabelValue');
    if (trueLabelValue) {
        trueLabelValue.textContent = TREE_DATA.class_names[sample.true_label].replace(/_/g, ' ');
    }
    
    // Render tree
    renderTree(model, path);
    
    // Start animation
    setTimeout(() => startAnimation(sample, model, path), 500);
}

function renderFeatureSidebar(sample) {
    const featuresGrid = document.getElementById('featuresGrid');
    if (!featuresGrid) return;
    
    featuresGrid.innerHTML = '';
    
    Object.entries(sample.features).forEach(([key, value]) => {
        const mapping = TREE_DATA.feature_mappings[key];
        const displayValue = mapping && typeof value === 'number'
            ? mapping[value]
            : (typeof value === 'number' ? value.toFixed(2) : value);
        
        const item = document.createElement('div');
        item.className = 'feature-item';
        item.innerHTML = `
            <div class="feature-label" title="${FEATURE_DESCRIPTIONS[key] || key}">${FEATURE_DESCRIPTIONS[key] || key}</div>
            <div class="feature-value" title="${displayValue}">${displayValue}</div>
        `;
        featuresGrid.appendChild(item);
    });
}

function renderTree(nodes, pathIds) {
    const treeContainer = document.getElementById('treeContainer');
    if (!treeContainer) return;
    
    treeContainer.innerHTML = '';
    
    // Create SVG for edges
    // Calculate dynamic bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodes.forEach(n => {
        minX = Math.min(minX, n.x);
        maxX = Math.max(maxX, n.x);
        minY = Math.min(minY, n.y);
        maxY = Math.max(maxY, n.y);
    });

    const padding = 500; // Extra padding
    const width = Math.max(5000, maxX + padding); // Minimum 5000 or wider
    const height = Math.max(5000, maxY + padding);

    // Create SVG for edges
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "connections-layer");
    svg.setAttribute("width", width.toString());
    svg.setAttribute("height", height.toString());
    treeContainer.appendChild(svg);
    
    const pathSet = new Set(pathIds);
    
    // Render edges
    nodes.forEach(node => {
        if (!node.is_leaf) {
            const leftChild = nodes.find(n => n.id === node.left_child);
            const rightChild = nodes.find(n => n.id === node.right_child);
            
            if (leftChild) {
                drawEdge(svg, node, leftChild, pathSet.has(node.id) && pathSet.has(leftChild.id));
            }
            if (rightChild) {
                drawEdge(svg, node, rightChild, pathSet.has(node.id) && pathSet.has(rightChild.id));
            }
        }
    });
    
    // Render nodes
    nodes.forEach(node => {
        const nodeEl = createNodeElement(node, pathSet.has(node.id));
        treeContainer.appendChild(nodeEl);
    });
    
    // Initial camera position
    const rootNode = nodes.find(n => n.id === 0);
    if (rootNode) {
        focusOnNode(rootNode, 1.5, 0);
    }
}

function drawEdge(svg, startNode, endNode, isPath) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", startNode.x);
    line.setAttribute("y1", startNode.y + 40);
    line.setAttribute("x2", endNode.x);
    line.setAttribute("y2", endNode.y - 40);
    line.setAttribute("data-start", startNode.id);
    line.setAttribute("data-end", endNode.id);
    line.setAttribute("class", `connection-line ${isPath ? 'path' : ''}`);
    svg.appendChild(line);
}

function createNodeElement(node, isInPath) {
    const div = document.createElement('div');
    div.className = `tree-node ${isInPath ? 'in-path' : ''}`;
    div.id = `node-${node.id}`;
    div.style.left = `${node.x}px`;
    div.style.top = `${node.y}px`;
    
    if (node.is_leaf) {
        div.innerHTML = `
            <div class="node-prediction">PREDICTION</div>
            <div class="node-class">${node.class.replace(/_/g, ' ')}</div>
            <div class="node-samples" style="display:none;">${node.samples} samples</div>
        `;
    } else {
        const featureDesc = FEATURE_DESCRIPTIONS[node.feature] || node.feature;
        div.innerHTML = `
            <div class="node-label">${featureDesc}</div>
            <div class="node-value">&le; ${node.threshold.toFixed(2)}</div>
            <div class="node-feature-value" style="display:none;"></div>
        `;
    }
    
    return div;
}

function startAnimation(sample, model, pathIds) {
    animState.isAnimating = true;
    animState.currentStep = 0;
    
    // Create sequence from pathIds
    let sequence = pathIds.map(id => model.find(n => n.id === id));
    
    // Sort sequence by Y coordinate (depth) to ensure Parent->Child order
    // This fixes issues where JSON path order might be slightly shuffled
    sequence.sort((a, b) => a.y - b.y);
    
    // Force layout flush to ensure elements are positioned correctly
    if (document.getElementById('treeContainer')) {
        void document.getElementById('treeContainer').offsetWidth;
    }
    
    function animateNextStep() {
        if (animState.currentStep >= sequence.length) {
            animState.isAnimating = false;
            animState.isComplete = true;
            animState.fitTimeout = setTimeout(() => fitTreeToScreen(), 1500);
            return;
        }
        
        const node = sequence[animState.currentStep];
        const isLeaf = node.is_leaf;
        
        highlightNode(node.id);

        // Highlight edge to previous node if not start
        if (animState.currentStep > 0) {
            const prevNode = sequence[animState.currentStep - 1];
            // Find connection line using reliable data attributes
            const svg = document.querySelector('.connections-layer');
            if (svg) {
                const line = svg.querySelector(`line[data-start="${prevNode.id}"][data-end="${node.id}"]`);
                if (line) {
                    line.classList.add('path');
                    // Ensure visibility
                    line.style.opacity = '1';
                }
            }
        }
        
        if (!isLeaf) {
            showFeatureValue(node, sample.features[node.feature]);
        } else {
            const nodeEl = document.getElementById(`node-${node.id}`);
            if (nodeEl) {
                const samplesEl = nodeEl.querySelector('.node-samples');
                if (samplesEl) samplesEl.style.display = 'block';
                
                const prediction = TREE_DATA.class_names.indexOf(node.class);
                if (prediction === sample.true_label) {
                    nodeEl.querySelector('.node-class').classList.add('correct');
                }
            }
        }
        
        focusOnNode(node, 1.5, 1000 / animState.speed);
        
        const duration = isLeaf ? 2000 : 1500;
        animState.currentStep++;
        animState.animationTimeout = setTimeout(animateNextStep, duration / animState.speed);
    }
    
    animateNextStep();
}

function highlightNode(nodeId) {
    const nodeEl = document.getElementById(`node-${nodeId}`);
    if (nodeEl) {
        document.querySelectorAll('.tree-node.active').forEach(n => n.classList.remove('active'));
        nodeEl.classList.add('visited', 'active');
    }
}

function showFeatureValue(node, value) {
    const nodeEl = document.getElementById(`node-${node.id}`);
    if (!nodeEl) return;
    
    const valueEl = nodeEl.querySelector('.node-feature-value');
    if (valueEl) {
        const mapping = TREE_DATA.feature_mappings[node.feature];
        const displayValue = mapping && typeof value === 'number'
            ? mapping[value]
            : (typeof value === 'number' ? value.toFixed(2) : value);
        
        valueEl.textContent = `Val: ${displayValue}`;
        valueEl.style.display = 'inline-block';
    }
}

function setCameraTransform(x, y, scale, duration = 0) {
    animState.camera = { x, y, scale };
    const treeContainer = document.getElementById('treeContainer');
    if (!treeContainer) return;
    
    if (duration > 0) {
        treeContainer.style.transition = `transform ${duration}ms cubic-bezier(0.455, 0.030, 0.515, 0.955)`;
    } else {
        treeContainer.style.transition = 'none';
    }
    
    treeContainer.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
}

function focusOnNode(node, targetScale, duration) {
    const animationArea = document.getElementById('animationArea');
    if (!animationArea) return;
    
    const screenW = animationArea.clientWidth;
    const screenH = animationArea.clientHeight;
    
    const targetX = (screenW / 2) - (node.x * targetScale);
    const targetY = (screenH / 2) - (node.y * targetScale);
    
    setCameraTransform(targetX, targetY, targetScale, duration);
}

function fitTreeToScreen() {
    if (!TREE_DATA) return;
    
    const nodes = TREE_DATA.models[currentConfig.depth][currentConfig.criterion];
    
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodes.forEach(n => {
        minX = Math.min(minX, n.x);
        maxX = Math.max(maxX, n.x);
        minY = Math.min(minY, n.y);
        maxY = Math.max(maxY, n.y);
    });
    
    const padding = 100;
    const width = maxX - minX + 400;
    const height = maxY - minY + 300;
    
    const animationArea = document.getElementById('animationArea');
    if (!animationArea) return;
    
    const screenW = animationArea.clientWidth;
    const screenH = animationArea.clientHeight;
    
    const scaleX = (screenW - padding) / width;
    const scaleY = (screenH - padding) / height;
    const fitScale = Math.min(scaleX, scaleY, 0.8);
    
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    
    const targetX = (screenW / 2) - (centerX * fitScale);
    const targetY = (screenH / 2) - (centerY * fitScale);
    
    setCameraTransform(targetX, targetY, fitScale, 1500);
}

function handleMouseDown(e) {
    if (!animState.isComplete) return;
    animState.isDragging = true;
    animState.lastMouse = { x: e.clientX, y: e.clientY };
    const treeContainer = document.getElementById('treeContainer');
    if (treeContainer) treeContainer.classList.add('dragging');
}

function handleMouseMove(e) {
    if (!animState.isDragging || !animState.isComplete) return;
    
    const dx = e.clientX - animState.lastMouse.x;
    const dy = e.clientY - animState.lastMouse.y;
    
    setCameraTransform(
        animState.camera.x + dx,
        animState.camera.y + dy,
        animState.camera.scale,
        0
    );
    
    animState.lastMouse = { x: e.clientX, y: e.clientY };
}

function handleMouseUp() {
    animState.isDragging = false;
    const treeContainer = document.getElementById('treeContainer');
    if (treeContainer) treeContainer.classList.remove('dragging');
}

function handleWheel(e) {
    if (!animState.isComplete) return;
    e.preventDefault();
    
    const scaleFactor = 1.1;
    const direction = e.deltaY > 0 ? -1 : 1;
    const newScale = direction > 0
        ? Math.min(animState.camera.scale * scaleFactor, 4)
        : Math.max(animState.camera.scale / scaleFactor, 0.05);
    
    const animationArea = document.getElementById('animationArea');
    if (!animationArea) return;
    
    const rect = animationArea.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const newX = mouseX - (mouseX - animState.camera.x) * (newScale / animState.camera.scale);
    const newY = mouseY - (mouseY - animState.camera.y) * (newScale / animState.camera.scale);
    
    setCameraTransform(newX, newY, newScale, 0);
}

function enterAnimation() {
    document.getElementById('trainingSection').style.display = 'none';
    document.getElementById('animationSection').style.display = 'flex';
    
    // CRITICAL: Ensure CLASSIFICATION_TYPE is set before loading data
    // If not set (e.g., direct animation entry), default to binary for testing
    if (!CLASSIFICATION_TYPE) {
        console.warn("CLASSIFICATION_TYPE not set, defaulting to multiclass");
        CLASSIFICATION_TYPE = 'multiclass';
    }
    
    console.log("Entering animation with type:", CLASSIFICATION_TYPE);
    
    // Initialize animation if not already done
    // Always reload data to ensure correct classification type is used
    loadTreeData().then(() => {
        updateVisualization();
    });
}

function exitAnimation() {
    document.getElementById('animationSection').style.display = 'none';
    document.getElementById('trainingSection').style.display = 'flex';
}
