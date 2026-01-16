A Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It has a hierarchical tree structure consisting of a root node, internal nodes, branches, and leaf nodes.

The main components of a Decision Tree are:
- **Internal Nodes:** Represent attribute tests
- **Branches:** Represent outcomes of attribute tests
- **Leaf Nodes:** Represent final decision or class labels

As shown in the below figure, a Decision Tree works by recursively subdividing the dataset into smaller groups based on feature values. At each step, the algorithm selects the best feature that separates the data into different classes using an appropriate splitting criterion. This splitting process continues until the data within each node becomes highly uniform or a predefined stopping condition is met. Internal nodes represent decision conditions used to split the data, while leaf nodes represent the final predicted class.

#### 1. Splitting Criteria

To determine the best split at each node, Decision Trees commonly use **Gini Index** and **Entropy** as splitting criteria. The Gini Index measures the impurity of a node and represents the probability of incorrectly classifying a randomly chosen sample if it were labeled according to the class distribution in that node. Lower Gini values indicate purer nodes, and the Gini value becomes zero when all samples belong to a single class. However, it is slightly biased toward dominant classes.

Entropy, on the other hand, measures the amount of uncertainty or disorder in a node and is derived from information theory. It is used to calculate Information Gain, which helps in selecting the most informative feature for splitting. Entropy is zero for a perfectly pure node, while higher entropy values indicate greater disorder. It is sensitive to small changes in class distribution and often produces balanced and informative splits.

<div style="text-align: center; margin: 15px 0;">
<img src="images/decision-trees.png" alt="Decision Tree" style="max-height: 300px; width: auto;">
</div>

#### 2. Merits of Decision Tree

- **Easy to Understand and Interpret:** The tree structure is simple and intuitive, making it easy to visualize and explain decisions.
- **Handles Both Numerical and Categorical Data:** Decision Trees can work with different types of data without much preprocessing.
- **No Need for Feature Scaling:** Normalization or standardization is not required, as trees are not affected by feature scaling.

#### 3. Demerits of Decision Tree

- **Prone to Overfitting:** A deep decision tree may fit noise in the training data, reducing generalization.
- **Unstable Model:** Small changes in data can lead to a completely different tree structure.
- **Lower Accuracy Compared to Ensembles:** Single decision trees often perform worse than ensemble methods like Random Forest.