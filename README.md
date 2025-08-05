# Wine Quality Classification: A Detailed Decision Tree Pipeline 🍷

This project develops a machine learning model to classify red wines into "good" or "bad" quality categories based on their physiochemical properties. It utilizes a **Decision Tree Classifier** and demonstrates a comprehensive machine learning workflow, emphasizing detailed data analysis, preprocessing, model building, and hyperparameter tuning for a robust and interpretable solution.

***

## Table of Contents
1.  **Project Overview**
2.  **Dataset Description**
3.  **Methodology & Key Steps**
4.  **Underlying Mathematical Concepts**
5.  **How to Run the Project**
6.  **Dependencies**

***

## 1. Project Overview

This project guides you through building a Decision Tree model for a binary classification task. The core objectives are:

* **Data Transformation:** Converting a multi-class regression-like problem (wine quality ratings 3-8) into a binary classification problem (Good vs. Bad wine).
* **Exploratory Data Analysis (EDA):** Understanding feature distributions, class imbalance, and correlations within the dataset.
* **Decision Tree Implementation:** Building and evaluating a Decision Tree Classifier.
* **Hyperparameter Tuning:** Optimizing the Decision Tree to prevent overfitting and improve generalization using `GridSearchCV`.
* **Model Interpretability:** Leveraging the inherent interpretability of Decision Trees through visualization.

***

## 2. Dataset Description

The dataset, `winequality-red.csv`, contains information on red variants of Portuguese "Vinho Verde" wine. It includes 11 physiochemical input variables and one output variable: `quality` (score between 0 and 10).

| Feature Name | Description |
| :--- | :--- |
| `fixed acidity` | Most acids involved with wine or fixed or nonvolatile |
| `volatile acidity` | The amount of acetic acid in wine, which at too high levels can lead to an unpleasant, vinegar taste |
| `citric acid` | Found in small quantities, citric acid can add 'freshness' and flavor to wines |
| `residual sugar` | The amount of sugar remaining after fermentation stops |
| `chlorides` | The amount of salt in the wine |
| `free sulfur dioxide` | The free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine |
| `total sulfur dioxide` | Amount of free and bound forms of S02 |
| `density` | The density of wine |
| `pH` | Describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale |
| `sulphates` | A wine additive that can contribute to sulfur dioxide gas (SO2) levels, which acts as an antimicrobial and antioxidant |
| `alcohol` | The percent alcohol content of the wine |
| `quality` | **Target:** Output quality score (0-10) |

***

## 3. Methodology & Key Steps

### Step 1: Data Loading & Initial Exploration
The project begins by loading the `winequality-red.csv` file into a Pandas DataFrame. Initial checks using `df.head()`, `df.info()`, and `df.describe()` provide a quick overview of the dataset's structure, data types, and basic statistics. This step confirms the absence of missing values and identifies the `quality` column as the original target.

### Step 2: Exploratory Data Analysis (EDA) & Data Preprocessing

* **Binary Target Creation:** The original `quality` ratings (3-8) are converted into a **binary classification problem**. Wines with a `quality` score of **7 or higher are labeled as 'Good' (1)**, while those with scores below 7 are labeled as 'Bad' (0). This simplifies the classification task.
* **Class Imbalance Check:** A count plot of the new `quality_label` reveals that the dataset is **imbalanced**, with significantly more "Bad" quality wines than "Good" ones. This insight is crucial for model evaluation, as high accuracy can be misleading in imbalanced datasets; **Precision** and **Recall** for the minority class become more important.
* **Correlation Analysis:** A **correlation heatmap** is generated for all features. This visualization helps identify relationships between features and the target. For Decision Trees, multicollinearity (highly correlated features) is generally less of a concern compared to linear models, but understanding these relationships still provides valuable insights into the data.

### Step 3: Feature Scaling

* **StandardScaler** is applied to all input features. This transforms the data to have a mean of 0 and a standard deviation of 1.
* **Why for Decision Trees?** While Decision Trees are inherently **not sensitive to feature scaling** (because their splitting criteria are based on thresholds, not distances or magnitudes), scaling is still a good practice. It ensures consistency if other algorithms (e.g., SVMs, Logistic Regression, Neural Networks) are later explored, which *are* sensitive to feature scales.

### Step 4: Decision Tree Model Building & Evaluation

* **Model Intuition:** A Decision Tree learns by recursively partitioning the dataset into smaller subsets based on feature values. At each step, it selects the feature and threshold that best separates the data points into different classes. This process creates a tree-like structure of decision rules.
* **Splitting Criterion (Gini Impurity):** The quality of a split is measured by a criterion like **Gini Impurity**. The algorithm aims to find splits that result in the greatest reduction in impurity.
* **Initial Model Training:** A default `DecisionTreeClassifier` is trained on the scaled training data to establish a baseline performance.
* **Evaluation Metrics:** The initial model's performance is assessed using:
    * **Accuracy Score:** Overall percentage of correct predictions.
    * **Classification Report:** Provides **Precision**, **Recall**, and **F1-score** for each class. In imbalanced datasets, **Recall for the minority class** (Good wine) is particularly important to ensure the model isn't missing positive cases.
    * **Confusion Matrix:** A visual table showing True Positives, True Negatives, False Positives, and False Negatives. This helps understand specific types of errors the model is making.

### Step 5: Hyperparameter Tuning

* **Overfitting in Decision Trees:** Decision Trees can easily **overfit** the training data by growing too deep and learning noise. This leads to poor generalization on unseen data.
* **Hyperparameters for Control:** To mitigate overfitting, key hyperparameters are tuned:
    * `max_depth`: Limits the maximum depth of the tree.
    * `criterion`: Specifies the function to measure the quality of a split (`gini` or `entropy`).
    * `min_samples_split`: The minimum number of samples required to split an internal node.
    * `min_samples_leaf`: The minimum number of samples required to be at a leaf node.
* **`GridSearchCV`:** This technique systematically searches through a predefined grid of hyperparameter values, training and evaluating the model for each combination using cross-validation. It then identifies the combination that yields the best performance.

### Step 6: Final Model Evaluation & Visualization

* The optimized Decision Tree model (from `GridSearchCV`) is used to make final predictions on the test set.
* Its performance is re-evaluated using the same metrics (Accuracy, Classification Report, Confusion Matrix) to confirm improvements from tuning.
* **Decision Tree Visualization:** A significant advantage of Decision Trees is their interpretability. The final, optimized tree is visualized, allowing us to see the exact features and thresholds the model uses to make its classification decisions. This provides clear insights into the learned rules.

***

## 4. Underlying Mathematical Concepts

### Gini Impurity
Gini impurity is a measure of the disorder or "impurity" of a set of samples. For a node in a Decision Tree, Gini impurity measures the probability of incorrectly classifying a randomly chosen element from the set if it were randomly labeled according to the class distribution in the node.

The formula for Gini impurity ($G$) for a node containing samples from $J$ classes is:
$$G = 1 - \sum_{j=1}^{J} p_j^2$$
where $p_j$ is the proportion of samples belonging to class $j$ in that node.

* A Gini impurity of **0** means the node is perfectly pure (all samples belong to the same class).
* A Gini impurity of **0.5** (for a binary classification) means the node is perfectly impure (samples are evenly split between classes).

The Decision Tree algorithm aims to find splits that **maximize the reduction in Gini impurity** (or information gain if using entropy) from the parent node to its child nodes.

***

## 5. How to Run the Project
1.  Ensure you have the `winequality-red.csv` file in the same directory as your Python script.
2.  Install the necessary Python libraries (listed in Section 6).
3.  Execute the Python script in your preferred environment (e.g., Jupyter Notebook, VS Code, or directly from the terminal).

The script will print various insights, model performance metrics, and display plots throughout the execution.

***

## 6. Dependencies
This project requires the following Python libraries. You can install them using pip:

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `statsmodels` (used for VIF, though not strictly necessary for DTs, it's included for comprehensive statistical analysis as requested)
