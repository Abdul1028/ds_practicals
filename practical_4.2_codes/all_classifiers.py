import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the diabetes dataset
data = pd.read_csv('diabetic.csv')

# Assuming the target variable is 'Outcome'
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# # Handle missing values in the target variable either by adding in  place of null values or transformation

# y.fillna(0,inplace=True)
# X.fillna(0,inplace=True)

# Impute missing values in the target variable 'y' with the most frequent value
imputer_target = SimpleImputer(strategy='most_frequent')
y = imputer_target.fit_transform(y.values.reshape(-1, 1)).ravel()

# Impute missing values in the feature matrix 'X' with the mean of each feature column
imputer_features = SimpleImputer(strategy='mean')
X = imputer_features.fit_transform(X)

# Manually provide the feature names
feature_names = data.columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)
y_pred_tree = decision_tree_classifier.predict(X_test)

# Visualize Decision Tree using plot_tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree_classifier, filled=True, feature_names=feature_names, class_names=['No Diabetes', 'Diabetes'])
plt.show()

# Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_scaled, y_train)
y_pred_linear = (linear_regression_model.predict(X_test_scaled) > 0.5).astype(int)

# Naive Bayes
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)
y_pred_naive_bayes = naive_bayes_classifier.predict(X_test)

# ROC Curve for Decision Tree
fpr_tree, tpr_tree, _ = roc_curve(y_test, decision_tree_classifier.predict_proba(X_test)[:, 1])
roc_auc_tree = auc(fpr_tree, tpr_tree)

# ROC Curve for Linear Regression
fpr_linear, tpr_linear, _ = roc_curve(y_test, linear_regression_model.predict(X_test_scaled))
roc_auc_linear = auc(fpr_linear, tpr_linear)

# ROC Curve for Naive Bayes
fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, naive_bayes_classifier.predict_proba(X_test)[:, 1])
roc_auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes)

# Plot ROC Curves
plt.figure(figsize=(8, 8))
plt.plot(fpr_tree, tpr_tree, label=f'Decision Tree (AUC = {roc_auc_tree:.2f})')
plt.plot(fpr_linear, tpr_linear, label=f'Linear Regression (AUC = {roc_auc_linear:.2f})')
plt.plot(fpr_naive_bayes, tpr_naive_bayes, label=f'Naive Bayes (AUC = {roc_auc_naive_bayes:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()


# Evaluate the performance of the classifiers
def evaluate_model(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)

    print(f'{model_name} Model:')
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
    print(f'Classification Report:\n{class_report}')
    print('\n')


evaluate_model('Decision Tree', y_test, y_pred_tree)
evaluate_model('Linear Regression', y_test, y_pred_linear)
evaluate_model('Naive Bayes', y_test, y_pred_naive_bayes)